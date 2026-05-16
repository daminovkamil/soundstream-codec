import typing as tp

import torch
from torch import nn
from torch.nn.utils import weight_norm


class WaveDiscriminator(nn.Module):
    def __init__(
        self,
        negative_slope: float = 0.2,
    ):
        super().__init__()

        self.activation = nn.LeakyReLU(negative_slope)

        self.layers = nn.ModuleList(
            [
                weight_norm(nn.Conv1d(1, 16, 15, 1, padding=7)),
                weight_norm(nn.Conv1d(16, 64, 41, 4, padding=20, groups=4)),
                weight_norm(nn.Conv1d(64, 256, 41, 4, padding=20, groups=16)),
                weight_norm(nn.Conv1d(256, 1024, 41, 4, padding=20, groups=64)),
                weight_norm(nn.Conv1d(1024, 1024, 41, 4, padding=20, groups=256)),
                weight_norm(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
                weight_norm(nn.Conv1d(1024, 1, 3, 1, padding=1)),
            ]
        )

    def forward(
        self, x: torch.FloatTensor
    ) -> tp.Tuple[torch.FloatTensor, tp.List[torch.FloatTensor]]:
        """
        Args:
            x: [B, 1, T] raw audio waveform.

        Returns:
            logits:   [B, 1, T'] per-time-step scores.
            features: list of intermediate activations.
        """
        features: tp.List[torch.FloatTensor] = []

        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.activation(x)
            features.append(x)

        logits = self.layers[-1](x)

        return logits, features


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride_freq: int,
        stride_time: int,
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=(3, 3),
            padding=(1, 1),
        )

        self.conv2 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(stride_freq + 2, stride_time + 2),
            stride=(stride_freq, stride_time),
            padding=(stride_freq, stride_time),
        )

        self.skip = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, 1),
            stride=(stride_freq, stride_time),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C_in, F, T] input feature map.

        Returns:
            [B, C_out, F_new, T_new] output.
        """
        residual = self.skip(x)

        x = self.conv1(x)
        x = nn.functional.leaky_relu(x, negative_slope=0.2)
        x = self.conv2(x)
        x = nn.functional.leaky_relu(x, negative_slope=0.2)

        if x.shape != residual.shape:
            diff_freq = residual.shape[2] - x.shape[2]
            diff_time = residual.shape[3] - x.shape[3]
            if diff_freq > 0 or diff_time > 0:
                x = nn.functional.pad(x, (0, diff_time, 0, diff_freq))
            elif diff_freq < 0 or diff_time < 0:
                residual = nn.functional.pad(residual, (0, -diff_time, 0, -diff_freq))

        return x + residual


class STFTDiscriminator(nn.Module):
    def __init__(
        self,
        win_length: int = 1024,
        hop_length: int = 256,
        n_channels: int = 32,
    ) -> None:
        super().__init__()

        self.n_fft = win_length
        self.win_length = win_length
        self.hop_length = hop_length
        self.register_buffer("window", torch.hann_window(win_length), persistent=False)

        self.input_conv = nn.Sequential(
            nn.Conv2d(2, n_channels, kernel_size=7, padding=3),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.residual_blocks = nn.ModuleList(
            [
                ResidualBlock(
                    1 * n_channels, 2 * n_channels, stride_freq=2, stride_time=1
                ),
                ResidualBlock(
                    2 * n_channels, 4 * n_channels, stride_freq=2, stride_time=2
                ),
                ResidualBlock(
                    4 * n_channels, 4 * n_channels, stride_freq=2, stride_time=1
                ),
                ResidualBlock(
                    4 * n_channels, 8 * n_channels, stride_freq=2, stride_time=2
                ),
                ResidualBlock(
                    8 * n_channels, 8 * n_channels, stride_freq=2, stride_time=1
                ),
                ResidualBlock(
                    8 * n_channels, 16 * n_channels, stride_freq=2, stride_time=2
                ),
            ]
        )

        n = self.n_fft // 2 + 1
        for _ in range(6):
            n = (n - 1) // 2 + 1

        self.output_conv = nn.Conv2d(16 * n_channels, 1, kernel_size=(n, 1))

    def forward(self, x: torch.Tensor) -> tp.Tuple[torch.Tensor, tp.List[torch.Tensor]]:
        """
        Args:
            x: [B, 1, T] raw audio waveform.

        Returns:
            logits:   [B, 1, T'] per-time-step scores.
            features: list of [B, C_l, F_l, T_l] feature maps.
        """
        x = torch.squeeze(x, dim=1)
        x = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
        )

        x = torch.view_as_real(x)
        x = x.permute(0, 3, 1, 2)

        features: tp.List[torch.Tensor] = []

        x = self.input_conv(x)
        features.append(x)

        for block in self.residual_blocks:
            x = block(x)
            features.append(x)

        logits = self.output_conv(x).squeeze(-2)

        return logits, features


DiscOutputs = tp.List[tp.Tuple[torch.FloatTensor, tp.List[torch.FloatTensor]]]


class SoundStreamDisciminator(nn.Module):
    def __init__(
        self,
        win_length: int,
        hop_length: int,
        n_channels: int,
        negative_slope: float = 0.2,
    ):
        super().__init__()
        self.wave_discriminators = nn.ModuleList(
            [WaveDiscriminator(negative_slope=negative_slope) for _ in range(3)]
        )
        self.pool = nn.AvgPool1d(kernel_size=4, stride=2, padding=1)

        self.stft_discriminator = STFTDiscriminator(win_length, hop_length, n_channels)

    def forward(self, x: torch.FloatTensor) -> DiscOutputs:
        """
        Args:
            x: [B, 1, T] raw audio waveform.

        Returns:
            list of 4 (logits [B, 1, T_k], features [list of B, C_l, *]) tuples.
        """
        outputs: DiscOutputs = [self.stft_discriminator(x)]

        for discriminator in self.wave_discriminators:
            outputs.append(discriminator(x))
            x = self.pool(x)

        return outputs
