# SoundStream Neural Codec

PyTorch implementation of [SoundStream](https://arxiv.org/abs/2107.03312), trained on LibriSpeech at 16 kHz using [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/).

## Usage

Clone the repo, install dependencies, and set your Comet API key. Training logs to the `soundstream-codec` Comet project.

```bash
pip install -r requirements.txt
export COMET_API_KEY="YOUR_API_KEY"
python train.py
```

On the first run, LibriSpeech is downloaded to `data/` automatically. Checkpoints are saved to `checkpoints/` and ranked by `val/generator_loss`.

Common flags: `--data-dir`, `--batch-size`, `--num-workers`, `--max-steps`, `--crop-length`, `--lr`, `--ckpt-dir`, `--resume`, `--precision`. See `python train.py --help` for defaults.

Example — resume from a checkpoint with a smaller batch:

```bash
python train.py --resume checkpoints/your.ckpt --batch-size 16 --max-steps 20000
```

## Inference

Load the codec from a Lightning checkpoint (architecture must match training — see constants in `train.py`):

```python
import torch
import torchaudio

from src.model import SoundStream
from train import CHANNELS, CODEBOOK_SIZE, EMBEDDING_DIM, NUM_QUANTIZERS, SAMPLE_RATE, STRIDES

codec = SoundStream.from_pretrained(
    "checkpoints/your.ckpt",
    CHANNELS,
    EMBEDDING_DIM,
    NUM_QUANTIZERS,
    CODEBOOK_SIZE,
    STRIDES,
)

wav, sr = torchaudio.load("audio.wav")
if wav.size(0) > 1:
    wav = wav.mean(dim=0, keepdim=True)
if sr != SAMPLE_RATE:
    wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
wav = wav.unsqueeze(0).clamp(-1.0, 1.0)  # [B, 1, T], 16 kHz mono

with torch.no_grad():
    recon, indices, _ = codec(wav)
```

Step-by-step API (`encode` → `quantize` → `unquantize` → `decode`):

```python
with torch.no_grad():
    encoded = codec.encode(wav)
    indices = codec.quantize(encoded)
    recon = codec.decode(codec.unquantize(indices), length=wav.size(-1))
```

Save reconstruction:

```python
torchaudio.save("recon.wav", recon.squeeze(0).cpu(), SAMPLE_RATE)
```
