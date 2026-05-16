import torch
import torch.nn.functional as F
import torchaudio


class LibriSpeechCodec(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str,
        split: str,
        sample_rate: int = 16_000,
        crop_length: int | None = None,
        download: bool = True,
    ):
        self.dataset = torchaudio.datasets.LIBRISPEECH(
            root=root, url=split, download=download
        )
        self.sample_rate = sample_rate
        self.crop_length = crop_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        waveform, sr, *_ = self.dataset[idx]
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        waveform = waveform.clamp(-1.0, 1.0)

        if self.crop_length is not None:
            waveform = self._random_crop(waveform)

        return waveform, waveform.size(-1)

    def _random_crop(self, waveform):
        if waveform.size(-1) < self.crop_length:
            waveform = F.pad(
                waveform, (0, self.crop_length - waveform.size(-1)), mode="replicate"
            )
        start = torch.randint(0, waveform.size(-1) - self.crop_length + 1, (1,)).item()
        return waveform[..., start : start + self.crop_length]


def train_collate(batch):
    waveforms, lengths = zip(*batch)
    return torch.stack(waveforms), torch.tensor(lengths)


def eval_collate(batch):
    waveforms, lengths = zip(*batch)
    max_length = max(w.size(-1) for w in waveforms)
    padded = [F.pad(w, (0, max_length - w.size(-1))) for w in waveforms]
    return torch.stack(padded), torch.tensor(lengths)
