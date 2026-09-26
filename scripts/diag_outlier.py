"""Separate one piece several times and locate where the error energy sits.

Full-track nSDR per run (seeded random shift, plus shifts=0), and for each source the share of the
track's error energy that falls in its worst 1-s window. A normal run spreads error over the track;
a glitch puts most of it in a few seconds.
"""
import random
import sys
from pathlib import Path

import torch as th

from demucs.evaluate import new_sdr
from demucs.pretrained import get_model
from scripts.compare_stems import load, separate

piece_dir, repo, name = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
model = get_model(name, repo=Path(repo) if repo != "-" else None).eval()
device = "cuda" if th.cuda.is_available() else "cpu"
rate, channels = model.samplerate, model.audio_channels
mix = load(piece_dir / "mixture.wav", rate, channels)
refs = th.stack([load(piece_dir / f"{s}.wav", rate, channels) for s in model.sources])
print(f"{piece_dir.name}: {mix.shape[-1] / rate:.1f} s, mixture peak {mix.abs().max():.3f}, device {device}")

for label, shifts, seed in [("shifts=0", 0, None)] + [(f"shifts=1 seed={k}", 1, k) for k in range(4)]:
    if seed is not None:
        random.seed(seed)
    with th.no_grad():
        est = separate(model, mix.to(device), shifts, 0.25).cpu()
    n = min(est.shape[-1], refs.shape[-1])
    ref, est = refs[..., :n].double(), est[..., :n].double()
    nsdr = new_sdr(ref[None], est[None])[0]
    err = ((ref - est) ** 2).sum(1)                      # (sources, time)
    win = err[:, : n // rate * rate].reshape(len(model.sources), -1, rate).sum(-1)
    worst = win.argmax(-1)
    share = win.max(-1).values / err.sum(-1)
    print(f"{label:<16} finite={bool(th.isfinite(est).all())} est peak {est.abs().max():7.3f}")
    for i, s in enumerate(model.sources):
        print(f"    {s:<7} nSDR {float(nsdr[i]):6.2f}  worst 1-s window at {int(worst[i]):4d} s "
              f"holds {100 * float(share[i]):5.1f}% of error energy")
