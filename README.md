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
