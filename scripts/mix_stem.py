#!/usr/bin/env python3
"""
mix_stem.py: Build the Turkish ensemble dataset from per-instrument synthesized renders.

Each track is one SymbTr piece rendered through every instrument soundfont, so all stems
carry the same melodic line in heterophony (see POST-ABSTRACT-PLAN.md §3.1).

  vocals = baglama_saz        (target)
  other  = ney + kanun
  bass   = ud
  drums  = bendir             (usul cycle)

Gain staging: stems are RMS-normalized, summed to form the mixture, then mixture *and*
every stem are divided by one shared scalar so that `mixture == sum(stems)` sample-for-
sample. The pilot dataset violated this (mixture was 0.5x the stem sum), which pinned the
SDR floor at 0 dB and the ceiling at ~6 dB -- see POST-ABSTRACT-PLAN.md §4.3.

  Output: data/dataset_tr/{train|valid|test}/{piece}/{vocals,other,bass,drums,mixture}.wav

The MUSDB-based pilot version of this script is in git history (pre-2026-09-20).
"""

import argparse
import random
from pathlib import Path

import numpy as np
import soundfile as sf

SYNTH_DIR = Path("data/synthesized")
OUT_DIR = Path("data/dataset_tr")

# Stem slot -> synthesized instrument directories summed into it.
STEM_SOURCES = {
    "vocals": ["baglama_saz"],
    "other": ["ney", "kanun"],
    "bass": ["ud"],
    "drums": ["bendir"],
}

TARGET_DBFS = -18.0
SAMPLERATE = 44100
SUBTYPE = "PCM_16"  # matches MUSDB18-HQ; see verify_track() on quantization tolerance

# chisle: every stem normalized to the same RMS. If the task proves too hard, boosting the
# lead is the first knob to turn -- but an even mix is the honest heterophonic default.


def load_wav(path: Path) -> np.ndarray:
    """Load a wav as float32 (samples, 2).

    Renders arrive centred -- generate_midi.center() handles soundfont pan at the source.
    """
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if sr != SAMPLERATE:
        raise ValueError(f"{path}: expected {SAMPLERATE} Hz, got {sr}")
    if data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)
    return data[:, :2]


def rms_normalize(audio: np.ndarray, target_dbfs: float = TARGET_DBFS) -> np.ndarray:
    rms = float(np.sqrt(np.mean(audio**2)))
    if rms == 0.0:
        return audio
    return audio * (10.0 ** (target_dbfs / 20.0) / rms)


def pad_to(audio: np.ndarray, n: int) -> np.ndarray:
    if len(audio) >= n:
        return audio[:n]
    return np.pad(audio, ((0, n - len(audio)), (0, 0)))


def build_track(piece: str, out_track: Path) -> None:
    """Assemble one track. Raises FileNotFoundError if any instrument render is missing."""
    # Load every constituent first so length can be reconciled across all of them.
    loaded = {
        slot: [load_wav(SYNTH_DIR / inst / f"{piece}.wav") for inst in insts]
        for slot, insts in STEM_SOURCES.items()
    }
    n = max(len(a) for parts in loaded.values() for a in parts)

    stems = {
        slot: rms_normalize(sum(pad_to(a, n) for a in parts))
        for slot, parts in loaded.items()
    }

    mixture = sum(stems.values())

    # One shared scalar, applied to the mixture and every stem, so the additive
    # relationship survives. Never scale stems independently after this point.
    #
    # Every stem gets a vote, not just the mixture. RMS-normalizing percussion against
    # melodic stems gives the bendir a far higher crest factor, so its transient peak can
    # exceed the mixture's -- the melodic stems happen to sit negative underneath it.
    # Scaling by the mixture alone then leaves that stem above 1.0, PCM_16 clips it on
    # write, and `mixture == sum(stems)` breaks by ~1e-2 on exactly those samples.
    k = max([1.0, float(np.max(np.abs(mixture)))]
            + [float(np.max(np.abs(stem))) for stem in stems.values()])
    mixture /= k

    out_track.mkdir(parents=True, exist_ok=True)
    for slot, audio in stems.items():
        sf.write(str(out_track / f"{slot}.wav"), audio / k, SAMPLERATE, subtype=SUBTYPE)
    sf.write(str(out_track / "mixture.wav"), mixture, SAMPLERATE, subtype=SUBTYPE)


def verify_track(out_track: Path, tol: float = 1e-3) -> float:
    """Return max |mixture - sum(stems)|.

    PCM_16 rounds each file on write, so four stems accumulate ~1e-4 of quantization
    noise (-80 dBFS, inaudible and far below any SDR effect). The pilot's gain defect
    (mixture = 0.5 * sum) produced errors around 5e-2, so `tol` sits between the two.
    """
    mixture = load_wav(out_track / "mixture.wav")
    total = sum(load_wav(out_track / f"{slot}.wav") for slot in STEM_SOURCES)
    err = float(np.max(np.abs(mixture - total)))
    if err > tol:
        raise AssertionError(f"{out_track.name}: mixture != sum(stems), max err {err:.2e}")
    return err


def available_pieces() -> list[str]:
    """Pieces rendered for every instrument -- a missing render silently drops a stem."""
    per_instrument = [
        {p.stem for p in (SYNTH_DIR / inst).glob("*.wav")}
        for insts in STEM_SOURCES.values()
        for inst in insts
    ]
    return sorted(set.intersection(*per_instrument))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build the Turkish ensemble dataset from synthesized stems."
    )
    parser.add_argument("--n_train", type=int, default=60)
    parser.add_argument("--n_valid", type=int, default=15)
    parser.add_argument("--n_test", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="After each track, assert mixture == sum(stems) (guards POST-ABSTRACT-PLAN §4.3)",
    )
    args = parser.parse_args()

    pieces = available_pieces()
    if not pieces:
        raise FileNotFoundError(
            f"No piece rendered for all of {[i for v in STEM_SOURCES.values() for i in v]} "
            f"under {SYNTH_DIR}"
        )
    print(f"{len(pieces)} pieces rendered for all instruments.")

    random.seed(args.seed)
    random.shuffle(pieces)

    needed = args.n_train + args.n_valid + args.n_test
    if needed > len(pieces):
        raise SystemExit(f"Need {needed} pieces, only {len(pieces)} available.")

    cursor = 0
    for split, count in [
        ("train", args.n_train),
        ("valid", args.n_valid),
        ("test", args.n_test),
    ]:
        chosen = pieces[cursor : cursor + count]
        cursor += count
        print(f"\n[{split}] {len(chosen)} tracks -> {OUT_DIR / split}")
        for i, piece in enumerate(chosen, 1):
            out_track = OUT_DIR / split / piece
            print(f"  [{i:>3}/{len(chosen)}] {piece}")
            build_track(piece, out_track)
            if args.verify:
                verify_track(out_track)
        print(f"[{split}] Done.")

    if args.verify:
        print("\nVerified: mixture == sum(stems) for every track.")
