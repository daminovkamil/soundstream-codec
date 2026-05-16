import argparse

import pytorch_lightning as pl
import torch
from pystoi import stoi
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CometLogger
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchmetrics.audio import NonIntrusiveSpeechQualityAssessment

from src.dataset import LibriSpeechCodec, eval_collate, train_collate
from src.discriminator import SoundStreamDisciminator
from src.losses import (
    DiscriminatorAdversarialLoss,
    FeatureLoss,
    GeneratorAdversarialLoss,
    ReconstructionLoss,
)
from src.model import Decoder, Encoder, ResidualVectorQuantizer

SAMPLE_RATE = 16_000
STRIDES = [2, 4, 5, 5]


class SoundStream(nn.Module):
    def __init__(
        self, channels=32, embedding_dim=128, num_quantizers=8, codebook_size=1024
    ):
        super().__init__()
        self.encoder = Encoder(channels, embedding_dim, STRIDES)
        self.decoder = Decoder(embedding_dim, channels, STRIDES)
        self.quantizer = ResidualVectorQuantizer(
            num_quantizers, codebook_size, embedding_dim
        )

    def forward(self, waveform):
        length = waveform.size(-1)
        encoded = self.encoder(waveform)
        quantized, indices, commitment_loss = self.quantizer(encoded)
        reconstructed = self.decoder(quantized)[..., :length]
        return reconstructed, indices, commitment_loss


class LibriSpeechDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=32, num_workers=8, crop_length=SAMPLE_RATE // 2):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.crop_length = crop_length

    def prepare_data(self):
        LibriSpeechCodec(self.data_dir, "train-clean-100", download=True)
        LibriSpeechCodec(self.data_dir, "test-clean", download=True)

    def setup(self, stage=None):
        train_full = LibriSpeechCodec(
            self.data_dir, "train-clean-100", crop_length=self.crop_length
        )

        generator = torch.Generator().manual_seed(42)
        self.train_dataset, self.val_dataset = random_split(
            train_full, [0.9, 0.1], generator=generator
        )

        self.test_dataset = LibriSpeechCodec(self.data_dir, "test-clean")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers,
            collate_fn=train_collate,
        )

    def _eval_loader(self, dataset):
        return DataLoader(
            dataset,
            batch_size=1,
            num_workers=self.num_workers,
            collate_fn=eval_collate,
        )

    def val_dataloader(self):
        return self._eval_loader(self.val_dataset)

    def test_dataloader(self):
        return self._eval_loader(self.test_dataset)


class SoundStreamTrainer(pl.LightningModule):
    def __init__(
        self,
        lr=2e-4,
        channels=32,
        embedding_dim=128,
        num_quantizers=8,
        codebook_size=1024,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.codec = SoundStream(channels, embedding_dim, num_quantizers, codebook_size)
        self.discriminator = SoundStreamDisciminator(
            win_length=1024, hop_length=256, n_channels=32
        )

        self.reconstruction_loss = ReconstructionLoss(SAMPLE_RATE)
        self.generator_adv_loss = GeneratorAdversarialLoss()
        self.discriminator_adv_loss = DiscriminatorAdversarialLoss()
        self.feature_loss = FeatureLoss()

        self._nisqa: list = []

    def forward(self, waveform):
        reconstructed, _, _ = self.codec(waveform)
        return reconstructed

    def configure_optimizers(self):
        generator_optimizer = torch.optim.Adam(
            self.codec.parameters(), lr=self.hparams.lr, betas=(0.5, 0.9)
        )
        discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=self.hparams.lr, betas=(0.5, 0.9)
        )
        return [generator_optimizer, discriminator_optimizer]

    def training_step(self, batch, batch_idx):
        real_audio, _ = batch
        generator_optimizer, discriminator_optimizer = self.optimizers()

        with torch.no_grad():
            reconstructed_audio_detached, _, _ = self.codec(real_audio)

        discriminator_loss = self.discriminator_adv_loss(
            self.discriminator(real_audio),
            self.discriminator(reconstructed_audio_detached),
        )
        discriminator_optimizer.zero_grad()
        self.manual_backward(discriminator_loss)
        discriminator_optimizer.step()

        reconstructed_audio, indices, commitment_loss = self.codec(real_audio)

        self.discriminator.requires_grad_(False)
        reconstructed_outputs = self.discriminator(reconstructed_audio)
        with torch.no_grad():
            real_outputs = self.discriminator(real_audio)

        reconstruction_loss = self.reconstruction_loss(real_audio, reconstructed_audio)
        adversarial_loss = self.generator_adv_loss(reconstructed_outputs)
        feature_loss = self.feature_loss(real_outputs, reconstructed_outputs)
        generator_loss = (
            reconstruction_loss
            + adversarial_loss
            + 100.0 * feature_loss
            + commitment_loss
        )

        generator_optimizer.zero_grad()
        self.manual_backward(generator_loss)
        generator_optimizer.step()
        self.discriminator.requires_grad_(True)

        self.log_dict(
            {
                "train/generator_loss": generator_loss,
                "train/discriminator_loss": discriminator_loss,
                "train/reconstruction_loss": reconstruction_loss,
                "train/generator_adv_loss": adversarial_loss,
                "train/feature_loss": feature_loss,
                "train/commitment_loss": commitment_loss,
                "train/codebook_perplexity": self._codebook_perplexity(indices),
            },
            prog_bar=True,
            on_step=True,
            on_epoch=False,
        )

        if batch_idx == 0 and self.global_step > 0 and self.global_step % 1000 == 0:
            self._log_audio(real_audio, reconstructed_audio.detach(), prefix="train")

    def _eval_step(self, batch, prefix):
        real_audio, lengths = batch
        reconstructed_audio, indices, commitment_loss = self.codec(real_audio)

        length = lengths[0].item()
        real_audio = real_audio[..., :length]
        reconstructed_audio = reconstructed_audio[..., :length]

        if not self._nisqa:
            self._nisqa.append(
                NonIntrusiveSpeechQualityAssessment(SAMPLE_RATE).to(
                    reconstructed_audio.device
                )
            )

        real_outputs = self.discriminator(real_audio)
        fake_outputs = self.discriminator(reconstructed_audio)

        reconstruction_loss = self.reconstruction_loss(real_audio, reconstructed_audio)
        adversarial_loss = self.generator_adv_loss(fake_outputs)
        feature_loss = self.feature_loss(real_outputs, fake_outputs)
        discriminator_loss = self.discriminator_adv_loss(real_outputs, fake_outputs)
        generator_loss = (
            reconstruction_loss
            + adversarial_loss
            + 100.0 * feature_loss
            + commitment_loss
        )
        codebook_perplexity = self._codebook_perplexity(indices)
        stoi_score = stoi(
            real_audio[0, 0].float().cpu().numpy(),
            reconstructed_audio[0, 0].float().cpu().numpy(),
            SAMPLE_RATE,
            extended=False,
        )
        nisqa_score = self._nisqa[0](reconstructed_audio.squeeze(1).float())[0]

        self.log_dict(
            {
                f"{prefix}/reconstruction_loss": reconstruction_loss,
                f"{prefix}/generator_loss": generator_loss,
                f"{prefix}/generator_adv_loss": adversarial_loss,
                f"{prefix}/feature_loss": feature_loss,
                f"{prefix}/discriminator_loss": discriminator_loss,
                f"{prefix}/commitment_loss": commitment_loss,
                f"{prefix}/codebook_perplexity": codebook_perplexity,
            },
            batch_size=1,
        )
        self.log(f"{prefix}/stoi", stoi_score, batch_size=1, prog_bar=True)
        self.log(f"{prefix}/nisqa", nisqa_score, batch_size=1, prog_bar=True)
        return real_audio, reconstructed_audio

    def validation_step(self, batch, batch_idx):
        real_audio, reconstructed_audio = self._eval_step(batch, prefix="val")
        if batch_idx == 0:
            self._log_audio(real_audio, reconstructed_audio, prefix="val")

    def test_step(self, batch, batch_idx):
        self._eval_step(batch, prefix="test")

    def _codebook_perplexity(self, indices):
        codebook_size = self.hparams.codebook_size
        per_quantizer = indices.reshape(indices.size(1), -1)
        probabilities = torch.stack(
            [
                torch.bincount(row, minlength=codebook_size).float() / row.numel()
                for row in per_quantizer
            ]
        )
        entropy = -(probabilities * (probabilities + 1e-10).log()).sum(dim=-1)
        return entropy.exp().mean()

    def _log_audio(self, real_audio, reconstructed_audio, prefix):
        max_length = SAMPLE_RATE * 4
        for tag, audio in [
            ("real", real_audio),
            ("reconstructed", reconstructed_audio),
        ]:
            self.logger.experiment.log_audio(
                audio[0, 0, :max_length].detach().float().cpu().numpy(),
                sample_rate=SAMPLE_RATE,
                file_name=f"{prefix}_{tag}_{self.global_step}.wav",
                step=self.global_step,
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--crop-length", type=int, default=SAMPLE_RATE // 2)
    parser.add_argument("--max-steps", type=int, default=18_000)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--val-check-interval", type=int, default=2000)
    parser.add_argument("--val-batches", type=int, default=64)
    parser.add_argument("--ckpt-dir", default="checkpoints")
    parser.add_argument("--precision", default="16-mixed")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    torch.set_float32_matmul_precision("medium")

    datamodule = LibriSpeechDataModule(
        args.data_dir, args.batch_size, args.num_workers, crop_length=args.crop_length
    )
    model = SoundStreamTrainer(lr=args.lr)

    checkpoint = ModelCheckpoint(
        dirpath=args.ckpt_dir,
        filename="best-{step}-{val/generator_loss:.3f}",
        monitor="val/generator_loss",
        mode="min",
        save_top_k=4,
        auto_insert_metric_name=False,
    )

    trainer = pl.Trainer(
        max_steps=args.max_steps,
        accelerator="auto",
        devices=1,
        logger=CometLogger(project_name="soundstream-codec"),
        callbacks=[checkpoint, LearningRateMonitor(logging_interval="step")],
        log_every_n_steps=50,
        val_check_interval=args.val_check_interval,
        check_val_every_n_epoch=None,
        limit_val_batches=args.val_batches,
        precision=args.precision,
        benchmark=True,
    )
    trainer.fit(model, datamodule, ckpt_path=args.resume)
    trainer.test(model, datamodule, ckpt_path=checkpoint.best_model_path)


if __name__ == "__main__":
    main()
