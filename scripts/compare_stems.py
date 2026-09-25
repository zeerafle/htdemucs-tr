#!/usr/bin/env python3
"""
compare_stems.py: separate one piece and set each estimate beside its reference, for listening and
for figures.

Given a test-set folder (mixture.wav plus one wav per source), writes under OUT/<piece>/:
  separated/<source>.wav  the model's estimates, full length, float WAV (never clipped or rescaled)
  compare.png             per source: reference, estimate and residual (reference - estimate)
                          spectrograms of the excerpt, titled with the full-track nSDR
  ab_<source>.wav         the excerpt of the reference, a short gap, then the same excerpt of the
                          estimate, for A/B listening
  scores.json             full-track nSDR per source, same definition as demucs/evaluate.py

Given a plain audio file instead (a real recording, no references), writes the estimates and a
figure of the mixture and estimate spectrograms only.

Separation follows demucs/evaluate.py (normalised mixture, split, overlap 0.25, one random shift),
so a piece's nSDR here agrees with tools.test_pretrained to within that random shift.

  uv run --with matplotlib python -m scripts.compare_stems --repo release_models -n 10b843f9 \
      data/dataset_tr/test/<piece> -o demo/main
"""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch as th

from demucs.apply import apply_model
from demucs.audio import AudioFile, convert_audio
from demucs.evaluate import new_sdr
from demucs.pretrained import add_model_flags, get_model_from_args

# Slot mapping of the Turkish ensemble (conf/dset/tr_ensemble.yaml).
INSTRUMENTS = {"vocals": "bağlama", "other": "ney + kanun", "bass": "ud", "drums": "bendir"}
GAP_SECONDS = 0.5
DB_RANGE = 80


def load(path: Path, samplerate: int, channels: int) -> th.Tensor:
    """(channels, time) at the model's rate. WAVs via soundfile, as the evaluation reads them."""
    if path.suffix.lower() == ".wav":
        audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
        return convert_audio(th.from_numpy(audio.T), rate, samplerate, channels)
    return AudioFile(path).read(streams=0, samplerate=samplerate, channels=channels)


def separate(model, mix: th.Tensor, shifts: int, overlap: float) -> th.Tensor:
    """Same normalisation as demucs/evaluate.py: by the mono mixture's mean and std."""
    ref = mix.mean(dim=0)
    mix = (mix - ref.mean()) / ref.std()
    estimates = apply_model(model, mix[None], shifts=shifts, split=True, overlap=overlap)[0]
    return estimates * ref.std() + ref.mean()


def spectrogram_db(signal: th.Tensor, samplerate: int, n_fft: int = 2048, hop: int = 512):
    mono = signal.double().mean(dim=0)
    window = th.hann_window(n_fft, dtype=th.float64)
    magnitude = th.stft(mono, n_fft, hop_length=hop, window=window, return_complex=True).abs()
    times = np.arange(magnitude.shape[1]) * hop / samplerate
    freqs = np.fft.rfftfreq(n_fft, 1 / samplerate)
    return times, freqs, (20 * th.log10(magnitude + 1e-8)).numpy()


def label(source: str) -> str:
    return f"{INSTRUMENTS[source]} ({source})" if source in INSTRUMENTS else source


def plot(rows, samplerate: int, path: Path, title: str):
    """rows: list of (row title, [(panel title, signal), ...]). One colour scale per row, set by
    its first panel, so estimate and residual read against the reference."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ncols = max(len(panels) for _, panels in rows)
    fig, axes = plt.subplots(len(rows), ncols, figsize=(4.2 * ncols, 2.4 * len(rows)),
                             squeeze=False, sharex=True, sharey=True)
    for r, (row_title, panels) in enumerate(rows):
        vmax = None
        for c in range(ncols):
            ax = axes[r][c]
            if c >= len(panels):
                ax.axis("off")
                continue
            panel_title, signal = panels[c]
            times, freqs, db = spectrogram_db(signal, samplerate)
            if vmax is None:
                vmax = db.max()
            mesh = ax.pcolormesh(times, freqs[1:], db[1:], shading="auto", cmap="magma",
                                 vmin=vmax - DB_RANGE, vmax=vmax)
            ax.set_yscale("log")
            ax.set_ylim(40, samplerate / 2)
            ax.set_title(f"{row_title}: {panel_title}" if c == 0 else panel_title, fontsize=9)
            if c == 0:
                ax.set_ylabel("Hz")
            if r == len(rows) - 1:
                ax.set_xlabel("s")
        fig.colorbar(mesh, ax=list(axes[r]), label="dB", pad=0.01)
    fig.suptitle(title, fontsize=10)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser("compare_stems", description=__doc__.split("\n\n")[0])
    add_model_flags(parser)
    parser.add_argument("track", type=Path,
                        help="Test-set folder with mixture.wav and one wav per source, "
                             "or any audio file.")
    parser.add_argument("-o", "--out", type=Path, default=Path("demo"))
    parser.add_argument("--start", type=float, default=30.0,
                        help="Excerpt start in seconds, for the figure and A/B files.")
    parser.add_argument("--duration", type=float, default=15.0, help="Excerpt length in seconds.")
    parser.add_argument("--shifts", type=int, default=1)
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument("-d", "--device", default="cuda" if th.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model = get_model_from_args(args)
    model.eval()
    rate, channels = model.samplerate, model.audio_channels

    is_folder = args.track.is_dir()
    piece = args.track.name if is_folder else args.track.stem
    out = args.out / piece
    (out / "separated").mkdir(parents=True, exist_ok=True)

    mix = load(args.track / "mixture.wav" if is_folder else args.track, rate, channels)
    with th.no_grad():
        estimates = separate(model, mix.to(args.device), args.shifts, args.overlap).cpu()
    for source, estimate in zip(model.sources, estimates):
        sf.write(str(out / "separated" / f"{source}.wav"), estimate.numpy().T, rate,
                 subtype="FLOAT")

    references = None
    if is_folder and all((args.track / f"{s}.wav").exists() for s in model.sources):
        references = th.stack([load(args.track / f"{s}.wav", rate, channels)
                               for s in model.sources])
        length = min(references.shape[-1], estimates.shape[-1])
        references, estimates = references[..., :length], estimates[..., :length]

    total = mix.shape[-1] / rate
    start = min(max(args.start, 0.0), max(total - args.duration, 0.0))
    begin, end = int(start * rate), int(min(start + args.duration, total) * rate)
    excerpt = f"{start:.0f}–{end / rate:.0f} s"

    if references is None:
        rows = [("mixture", [("input", mix[:, begin:end])])]
        rows += [(label(s), [("estimate", e[:, begin:end])])
                 for s, e in zip(model.sources, estimates)]
        plot(rows, rate, out / "compare.png", f"{piece} — {excerpt}, no references")
        print(f"{piece}: estimates in {out / 'separated'}, figure {out / 'compare.png'}")
        return

    scores = new_sdr(references.double()[None], estimates.double()[None])[0]
    scores = {s: float(v) for s, v in zip(model.sources, scores)}
    (out / "scores.json").write_text(json.dumps(
        {"piece": piece, "model": args.name or args.sig, "shifts": args.shifts,
         "nsdr": scores, "instruments": {s: INSTRUMENTS.get(s, s) for s in model.sources}},
        indent=1, ensure_ascii=False))

    rows = []
    gap = th.zeros(channels, int(GAP_SECONDS * rate))
    for index, source in enumerate(model.sources):
        reference = references[index, :, begin:end]
        estimate = estimates[index, :, begin:end]
        rows.append((f"{label(source)}, nSDR {scores[source]:.2f} dB",
                     [("reference", reference), ("estimate", estimate),
                      ("residual", reference - estimate)]))
        sf.write(str(out / f"ab_{source}.wav"), th.cat([reference, gap, estimate], -1).numpy().T,
                 rate, subtype="FLOAT")
    plot(rows, rate, out / "compare.png", f"{piece} — {excerpt}")

    print(f"{piece} ({out}):")
    for source in model.sources:
        print(f"  {label(source):<22} nSDR {scores[source]:6.2f} dB")


if __name__ == "__main__":
    main()
