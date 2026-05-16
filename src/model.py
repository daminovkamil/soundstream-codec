import typing as tp
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


class CausalConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.left_padding = dilation * (kernel_size - 1)
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """
        Args:
            x: [B, C, T] input.

        Returns:
            [B, C_out, T_out] output.
        """
        return self.conv(F.pad(x, (self.left_padding, 0)))


class CausalConvTranspose1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.stride = stride
        self.conv = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """
        Args:
            x: [B, C, T] input.

        Returns:
            [B, C_out, T * stride] output.
        """
        length = x.size(-1)
        x = self.conv(x)
        return x[..., : length * self.stride]


class ResidualUnit(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.conv1 = CausalConv1d(
            kernel_size=7,
            in_channels=channels,
            out_channels=channels,
            dilation=dilation,
        )
        self.conv2 = CausalConv1d(
            kernel_size=1,
            in_channels=channels,
            out_channels=channels,
        )

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """
        Args:
            x: [B, C, T] input.

        Returns:
            [B, C, T] output.
        """
        residual = x
        x = F.elu(x)
        x = self.conv1(x)
        x = F.elu(x)
        x = self.conv2(x)
        return x + residual


class EncoderBlock(nn.Module):
    def __init__(self, in_channels: int, stride: int):
        super().__init__()
        self.layers = nn.Sequential(
            ResidualUnit(in_channels, dilation=1),
            ResidualUnit(in_channels, dilation=3),
            ResidualUnit(in_channels, dilation=9),
            nn.ELU(),
            CausalConv1d(
                kernel_size=2 * stride,
                in_channels=in_channels,
                out_channels=in_channels * 2,
                stride=stride,
            ),
        )

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """
        Args:
            x: [B, C, T] input.

        Returns:
            [B, 2*C, T // stride] output.
        """
        return self.layers(x)


class Encoder(nn.Module):
    def __init__(self, n_channels: int, output_dim: int, strides: tp.List[int]):
        super().__init__()

        self.strides = strides

        self.layers = nn.Sequential(
            CausalConv1d(
                in_channels=1,
                kernel_size=7,
                out_channels=n_channels,
            ),
        )

        for idx, stride in enumerate(strides):
            self.layers.append(EncoderBlock(n_channels * 2**idx, stride))

        self.layers.append(nn.ELU())
        self.layers.append(
            CausalConv1d(
                kernel_size=3,
                in_channels=n_channels * 2 ** len(strides),
                out_channels=output_dim,
            )
        )

    def forward(self, waveform: torch.FloatTensor) -> torch.FloatTensor:
        """
        Args:
            waveform: [B, 1, T] raw audio.

        Returns:
            [B, output_dim, T // prod(strides)] latent embeddings.
        """
        return self.layers(waveform)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, stride: int):
        super().__init__()
        out_channels = in_channels // 2
        self.layers = nn.Sequential(
            nn.ELU(),
            CausalConvTranspose1d(
                kernel_size=2 * stride,
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
            ),
            ResidualUnit(out_channels, dilation=1),
            ResidualUnit(out_channels, dilation=3),
            ResidualUnit(out_channels, dilation=9),
        )

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """
        Args:
            x: [B, C, T] input.

        Returns:
            [B, C // 2, T * stride] output.
        """
        return self.layers(x)


class Decoder(nn.Module):
    def __init__(self, input_dim: int, n_channels: int, strides: tp.List[int]):
        super().__init__()

        self.strides = strides

        hidden_channels = n_channels * 2 ** len(strides)

        self.layers = nn.Sequential(
            CausalConv1d(
                in_channels=input_dim,
                kernel_size=7,
                out_channels=hidden_channels,
            ),
        )

        for stride in reversed(strides):
            self.layers.append(DecoderBlock(hidden_channels, stride))
            hidden_channels //= 2

        self.layers.append(nn.ELU())
        self.layers.append(
            CausalConv1d(
                in_channels=hidden_channels,
                kernel_size=7,
                out_channels=1,
            )
        )

    def forward(self, embeddings: torch.FloatTensor) -> torch.FloatTensor:
        """
        Args:
            embeddings: [B, input_dim, T_latent] quantized latent vectors.

        Returns:
            [B, 1, T] reconstructed waveform.
        """
        return self.layers(embeddings)


class VectorQuantizer(nn.Module):
    def __init__(
        self,
        codebook_size: int,
        dim: int,
        decay: float = 0.99,
        ema_threshold: float = 2.0,
    ):
        super().__init__()
        self.dim = dim
        self.codebook_size = codebook_size
        self.decay = decay
        self.ema_threshold = ema_threshold

        self.register_buffer("codebook", torch.randn(codebook_size, dim))
        self.register_buffer("counts", torch.zeros(codebook_size))
        self.register_buffer("sums", torch.zeros(codebook_size, dim))
        self.register_buffer("initialized", torch.tensor(False))

    def forward(
        self,
        x: torch.FloatTensor,
    ) -> tp.Tuple[torch.FloatTensor, torch.LongTensor]:
        """
        Args:
            x: [B, D, T] encoder output.

        Returns:
            quantized: [B, D, T] nearest codebook vectors.
            indices:   [B, T] codebook indices per frame.
        """
        x = x.transpose(-1, -2).contiguous()

        if self.training and not self.initialized:
            self._kmeans_init(x)
            self.initialized.fill_(True)

        distances = torch.cdist(x, self.codebook)
        indices = distances.argmin(dim=-1)
        quantized = F.embedding(indices, self.codebook)

        if self.training:
            self._ema_update(x.detach(), indices)

        quantized = quantized.transpose(-1, -2).contiguous()
        return quantized, indices

    @torch.no_grad()
    def _kmeans_init(self, x: torch.FloatTensor) -> None:
        flat = x.reshape(-1, self.dim).to(self.codebook.dtype)
        n = flat.shape[0]
        if n < self.codebook_size:
            return

        idx = torch.randperm(n, device=flat.device)[: self.codebook_size]
        centroids = flat[idx].clone()

        for _ in range(20):
            distances = torch.cdist(flat, centroids)
            assignments = distances.argmin(dim=-1)
            one_hot = F.one_hot(assignments, self.codebook_size).type_as(flat)
            counts = one_hot.sum(dim=0).clamp_min(1)
            centroids = (one_hot.T @ flat) / counts.unsqueeze(-1)

        self.codebook.copy_(centroids)
        self.sums.copy_(centroids * 1.0)
        self.counts.fill_(1.0)

    @torch.no_grad()
    def _ema_update(
        self,
        x: torch.FloatTensor,
        indices: torch.LongTensor,
    ) -> None:
        x = x.reshape(-1, self.dim).to(self.codebook.dtype)
        indices = indices.reshape(-1)

        indices_one_hot = F.one_hot(indices, self.codebook_size).type_as(x)

        counts = indices_one_hot.sum(dim=0)
        sums = indices_one_hot.T @ x

        self.counts.mul_(self.decay).add_(counts, alpha=1 - self.decay)
        self.sums.mul_(self.decay).add_(sums, alpha=1 - self.decay)

        dead = self.counts < self.ema_threshold

        if dead.any():
            random_indices = torch.randint(
                0,
                x.shape[0],
                (dead.sum().item(),),
                device=x.device,
            )
            replacement = x[random_indices]

            self.codebook[dead] = replacement
            self.sums[dead] = replacement
            self.counts[dead] = 1.0

        alive = ~dead
        self.codebook[alive] = self.sums[alive] / self.counts[alive].unsqueeze(-1)


class ResidualVectorQuantizer(nn.Module):
    def __init__(
        self,
        num_quantizer: int,
        codebook_size: int,
        dim: int,
        decay: float = 0.99,
        ema_threshold: float = 2.0,
    ):
        super().__init__()

        self.quantizers = nn.ModuleList()

        for _ in range(num_quantizer):
            self.quantizers.append(
                VectorQuantizer(
                    codebook_size,
                    dim,
                    decay,
                    ema_threshold,
                )
            )

    def forward(self, x: torch.FloatTensor):
        """
        Args:
            x: [B, D, T] encoder output.

        Returns:
            quantized:       [B, D, T] STE-quantized latent.
            indices:         [B, N_q, T] codebook indices per quantizer.
            commitment_loss: scalar, per-stage residual commitment loss.
        """
        collected = torch.zeros_like(x)
        residual = x
        indices = []
        commitment_loss = torch.tensor(0.0, device=x.device)

        for quantizer in self.quantizers:
            quantized, quantizer_indices = quantizer(residual)
            commitment_loss = (
                commitment_loss + (residual - quantized.detach()).pow(2).mean()
            )
            collected = collected + quantized
            residual = residual - quantized.detach()
            indices.append(quantizer_indices)

        commitment_loss = commitment_loss / len(self.quantizers)
        quantized_st = x + (collected - x).detach()
        return quantized_st, torch.stack(indices, dim=1), commitment_loss

    def quantize(self, x: torch.FloatTensor) -> torch.LongTensor:
        """
        Args:
            x: [B, D, T] encoder output.

        Returns:
            [B, N_q, T] codebook indices per quantizer.
        """
        residual = x
        indices = []
        for quantizer in self.quantizers:
            quantized, quantizer_indices = quantizer(residual)
            indices.append(quantizer_indices)
            residual = residual - quantized.detach()
        return torch.stack(indices, dim=1)

    def unquantize(self, indices: torch.LongTensor) -> torch.FloatTensor:
        """
        Args:
            indices: [B, N_q, T] codebook indices per quantizer.

        Returns:
            [B, D, T] summed quantized latent.
        """
        collected = torch.zeros(
            indices.size(0),
            self.quantizers[0].dim,
            indices.size(-1),
            device=indices.device,
            dtype=self.quantizers[0].codebook.dtype,
        )
        for i, quantizer in enumerate(self.quantizers):
            quantized = F.embedding(indices[:, i], quantizer.codebook)
            collected = collected + quantized.transpose(-1, -2).contiguous()
        return collected

class SoundStream(nn.Module):
    def __init__(
        self,
        channels: int,
        embedding_dim: int,
        num_quantizers: int,
        codebook_size: int,
        strides: tp.List[int],
    ):
        super().__init__()
        self.encoder = Encoder(channels, embedding_dim, strides)
        self.decoder = Decoder(embedding_dim, channels, strides)
        self.quantizer = ResidualVectorQuantizer(
            num_quantizers, codebook_size, embedding_dim
        )

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str | Path,
        channels: int,
        embedding_dim: int,
        num_quantizers: int,
        codebook_size: int,
        strides: tp.List[int],
        map_location: str | torch.device = "cpu",
    ) -> "SoundStream":
        """Load codec weights from a PyTorch Lightning checkpoint."""
        ckpt = torch.load(
            checkpoint_path, map_location=map_location, weights_only=False
        )
        model = cls(channels, embedding_dim, num_quantizers, codebook_size, strides)
        state = {
            k.removeprefix("codec."): v
            for k, v in ckpt["state_dict"].items()
            if k.startswith("codec.")
        }
        model.load_state_dict(state, strict=True)
        model.eval()
        return model

    def encode(self, waveform: torch.FloatTensor) -> torch.FloatTensor:
        """
        Args:
            waveform: [B, 1, T] raw audio.

        Returns:
            [B, D, T_latent] continuous encoder latent.
        """
        return self.encoder(waveform)

    def quantize(self, encoded: torch.FloatTensor) -> torch.LongTensor:
        """
        Args:
            encoded: [B, D, T_latent] encoder output.

        Returns:
            [B, N_q, T_latent] discrete codes.
        """
        return self.quantizer.quantize(encoded)

    def unquantize(self, indices: torch.LongTensor) -> torch.FloatTensor:
        """
        Args:
            indices: [B, N_q, T_latent] discrete codes.

        Returns:
            [B, D, T_latent] quantized latent for the decoder.
        """
        return self.quantizer.unquantize(indices)

    def decode(
        self, quantized: torch.FloatTensor, length: int | None = None
    ) -> torch.FloatTensor:
        """
        Args:
            quantized: [B, D, T_latent] quantized latent.
            length: optional waveform length to crop the output to.

        Returns:
            [B, 1, T] reconstructed audio.
        """
        reconstructed = self.decoder(quantized)
        if length is not None:
            reconstructed = reconstructed[..., :length]
        return reconstructed

    def forward(self, waveform: torch.FloatTensor):
        length = waveform.size(-1)
        encoded = self.encode(waveform)
        quantized, indices, commitment_loss = self.quantizer(encoded)
        reconstructed = self.decode(quantized, length)
        return reconstructed, indices, commitment_loss
