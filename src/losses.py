import torch
import torch.nn.functional as F
import torchaudio
from torch import nn

from src.discriminator import DiscOutputs


class ReconstructionLoss(nn.Module):
    def __init__(self, sample_rate: int, floor_level: float = 1e-5):
        super().__init__()
        self.transforms = nn.ModuleList()
        self.alphas: list[float] = []
        self.floor_level = floor_level

        for power in range(6, 12):
            win_length = 2**power
            self.transforms.append(
                torchaudio.transforms.MelSpectrogram(
                    sample_rate=sample_rate,
                    n_fft=win_length,
                    win_length=win_length,
                    hop_length=win_length // 4,
                    n_mels=64,
                    f_min=64,
                )
            )
            self.alphas.append((win_length / 2) ** 0.5)

    def forward(
        self,
        original: torch.FloatTensor,
        reconstructed: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        Args:
            original:      [B, 1, T] ground-truth waveform.
            reconstructed: [B, 1, T] generated waveform.

        Returns:
            scalar loss.
        """
        loss = torch.tensor(0.0, device=original.device)
        for alpha, transform in zip(self.alphas, self.transforms):
            orig_mel = transform(original)
            recon_mel = transform(reconstructed)

            loss = loss + (orig_mel - recon_mel).abs().mean()
            log_diff = torch.log(self.floor_level + orig_mel) - torch.log(
                self.floor_level + recon_mel
            )
            loss = loss + alpha * log_diff.pow(2).mean()

        return loss


class GeneratorAdversarialLoss(nn.Module):
    def forward(self, fake_outputs: DiscOutputs) -> torch.FloatTensor:
        """
        Args:
            fake_outputs: discriminator outputs on generated audio.

        Returns:
            scalar loss.
        """
        loss = torch.tensor(0.0, device=fake_outputs[0][0].device)
        for logits, _ in fake_outputs:
            loss += F.relu(1 - logits).mean()
        return loss / len(fake_outputs)


class DiscriminatorAdversarialLoss(nn.Module):
    def forward(
        self,
        real_outputs: DiscOutputs,
        fake_outputs: DiscOutputs,
    ) -> torch.FloatTensor:
        """
        Args:
            real_outputs: discriminator outputs on original audio.
            fake_outputs: discriminator outputs on generated audio.

        Returns:
            scalar loss.
        """
        loss = torch.tensor(0.0, device=real_outputs[0][0].device)
        for (real_logits, _), (fake_logits, _) in zip(real_outputs, fake_outputs):
            loss += F.relu(1 - real_logits).mean()
            loss += F.relu(1 + fake_logits).mean()
        return loss / len(real_outputs)


class FeatureLoss(nn.Module):
    def forward(
        self,
        real_outputs: DiscOutputs,
        fake_outputs: DiscOutputs,
    ) -> torch.FloatTensor:
        """
        Args:
            real_outputs: discriminator outputs on original audio.
            fake_outputs: discriminator outputs on generated audio.

        Returns:
            scalar loss.
        """
        loss = torch.tensor(0.0, device=real_outputs[0][0].device)
        num_layers = 0
        for (_, real_features), (_, fake_features) in zip(real_outputs, fake_outputs):
            for real_feat, fake_feat in zip(real_features, fake_features):
                loss = loss + (real_feat - fake_feat).abs().mean()
                num_layers += 1
        return loss / num_layers


class CommitmentLoss(nn.Module):
    def forward(
        self,
        initial: torch.FloatTensor,
        quantized: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        Args:
            initial:   [B, D, T'] encoder output.
            quantized: [B, D, T'] quantized vectors.

        Returns:
            scalar loss.
        """
        return F.mse_loss(initial, quantized.detach())
