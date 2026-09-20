#!/usr/bin/env python3
"""
build_dataset.py: render every instrument of the Turkish ensemble for one set of pieces.

Pieces are chosen once, up front, and staged into a single directory that all five renders
read from. Choosing per-instrument instead would let a piece render for the melodists but
not the bendir, and mix_stem.py intersects across instruments -- the track would vanish
from the dataset with nothing to say why.

A piece qualifies only if its usul has a pattern *and* its meter agrees with that usul, so
the percussion stem is guaranteed before a single second of audio is rendered.

  python scripts/build_dataset.py --n 90
  python scripts/mix_stem.py --verify

Renders land in data/synthesized/{instrument}/, the varied MIDIs in data/midis/fixed/
{instrument}/ -- keep those, they are the only record of what the seeds produced.
"""

import argparse
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import mido

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_midi import USULS, build_usul, usul_from_filename  # noqa: E402

SYMBTR = Path("data/midis/symbtr/mid_v3")
STAGED = Path("data/midis/selected")
FIXED = Path("data/midis/fixed")
SYNTH = Path("data/synthesized")
PROGRAMS = "data/soundfonts_programs_tr.json"

# name, octave, jitter_ms, velocity_sigma, seed -- one seed per instrument so a single
# stem can be re-rendered identically without touching the others.
MELODIC = [
    ("baglama_saz", 0, 0.0, 0.0, 1),  # lead, on the grid
    ("kanun", 0, 20.0, 6.0, 2),
    ("ney", 1, 25.0, 8.0, 3),
    ("ud", -1, 15.0, 5.0, 4),
]
PERCUSSION = ("bendir", 12.0, 6.0, 5)


def qualifies(path: Path, max_seconds: float) -> "tuple[bool, float]":
    """A piece must have a usul pattern, a meter that agrees with it, and a sane length."""
    usul = usul_from_filename(str(path))
    if usul not in USULS:
        return False, 0.0
    try:
        source = mido.MidiFile(str(path))
        length = source.length
    except Exception:
        return False, 0.0
    if not 30.0 <= length <= max_seconds:
        return False, length
    try:
        build_usul(source, usul, 0)
    except ValueError:
        return False, length
    return True, length


def select(n: int, seed: int, max_seconds: float) -> "list[Path]":
    candidates = sorted(SYMBTR.glob("*.mid"))
    random.Random(seed).shuffle(candidates)
    chosen, total = [], 0.0
    for path in candidates:
        ok, length = qualifies(path, max_seconds)
        if ok:
            chosen.append(path)
            total += length
        if len(chosen) == n:
            break
    if len(chosen) < n:
        raise SystemExit(f"Only {len(chosen)} qualifying pieces, wanted {n}.")
    # 5 stems + mixture, 44.1 kHz stereo 16-bit, rendered twice (synthesized + dataset).
    per_second = 44100 * 2 * 2
    print(
        f"{len(chosen)} pieces, {total / 60:.0f} min of audio "
        f"(mean {total / len(chosen):.0f}s). Projected disk: "
        f"{total * per_second * 5 / 2**30:.1f} GiB synthesized + "
        f"{total * per_second * 5 / 2**30:.1f} GiB dataset."
    )
    return chosen


def run(args: "list[str]") -> None:
    result = subprocess.run(args)
    if result.returncode != 0:
        raise SystemExit(f"failed: {' '.join(args)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--n", type=int, default=90, help="Pieces to render (60/15/15 split).")
    parser.add_argument("--seed", type=int, default=42, help="Piece-selection seed.")
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=300.0,
        help="Skip pieces longer than this; long pieces dominate disk for no extra variety.",
    )
    parser.add_argument("--keep", action="store_true", help="Keep existing staged/rendered output.")
    opts = parser.parse_args()

    if not SYMBTR.is_dir():
        raise SystemExit(f"SymbTr corpus not found at {SYMBTR}")

    pieces = select(opts.n, opts.seed, opts.max_seconds)

    if not opts.keep and STAGED.exists():
        shutil.rmtree(STAGED)
    STAGED.mkdir(parents=True, exist_ok=True)
    for path in pieces:
        shutil.copy2(path, STAGED / path.name)
    print(f"Staged {len(pieces)} pieces in {STAGED}\n")

    started = time.time()
    for name, octave, jitter, velocity, seed in MELODIC:
        print(f"=== {name} (octave {octave:+d}, jitter {jitter}ms, velocity sigma {velocity})")
        run(
            [sys.executable, "scripts/generate_midi.py", "fix-then-synthesize",
             str(STAGED), str(FIXED), str(SYNTH),
             "--instrument", name, "--octave", str(octave), "--jitter-ms", str(jitter),
             "--velocity-sigma", str(velocity), "--seed", str(seed),
             "--soundfont-programs", PROGRAMS]
        )

    name, jitter, velocity, seed = PERCUSSION
    print(f"=== {name} (usul cycle, jitter {jitter}ms, velocity sigma {velocity})")
    run(
        [sys.executable, "scripts/generate_midi.py", "usul-then-synthesize",
         str(STAGED), str(FIXED), str(SYNTH),
         "--instrument", name, "--jitter-ms", str(jitter),
         "--velocity-sigma", str(velocity), "--seed", str(seed),
         "--soundfont-programs", PROGRAMS]
    )

    rendered = {
        instrument.name: len(list(instrument.glob("*.wav")))
        for instrument in sorted(SYNTH.iterdir())
        if instrument.is_dir()
    }
    print(f"\nRendered in {(time.time() - started) / 60:.1f} min: {rendered}")
    if len(set(rendered.values())) != 1:
        print("WARNING: instruments disagree on piece count; mix_stem.py will drop the gaps.")
    print("Next: python scripts/mix_stem.py --verify")
