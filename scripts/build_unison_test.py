#!/usr/bin/env python3
"""
build_unison_test.py: the register-vs-timbre ablation test set (POST-ABSTRACT-PLAN.md §3.6).

The main dataset gives each melodic instrument its own octave (ney +1, ud -1), so a separator
could score well by splitting the spectrum rather than by recognising timbre. This variant holds
everything else fixed -- same 15 test pieces, same onset jitter, same velocity spread, same seeds
-- and puts every melodic instrument in the *same* octave. Whatever the score drops by is the
part of the result that was register, not timbre.

  python scripts/build_unison_test.py

Mirrors data/dataset_tr/test exactly; the two scores are only comparable piece for piece.
"""

import shutil
import subprocess
import sys
from pathlib import Path

SYMBTR = Path("data/midis/symbtr/mid_v3")
REFERENCE = Path("data/dataset_tr/test")
STAGED = Path("data/midis/selected_unison")
FIXED = Path("data/midis/fixed_unison")
SYNTH = Path("data/synthesized_unison")
OUT = Path("data/dataset_tr_unison")
PROGRAMS = "data/soundfonts_programs_tr.json"

# Same jitter, velocity and seed as build_dataset.MELODIC -- only the octave is collapsed to 0.
MELODIC = [
    ("baglama_saz", 0.0, 0.0, 1),
    ("kanun", 20.0, 6.0, 2),
    ("ney", 25.0, 8.0, 3),
    ("ud", 15.0, 5.0, 4),
]
PERCUSSION = ("bendir", 12.0, 6.0, 5)


def run(args: list) -> None:
    if subprocess.run([str(a) for a in args]).returncode != 0:
        raise SystemExit(f"failed: {' '.join(str(a) for a in args)}")


if __name__ == "__main__":
    if not REFERENCE.is_dir():
        raise SystemExit(f"{REFERENCE} not found -- build the main dataset first.")

    pieces = sorted(p.name for p in REFERENCE.iterdir() if p.is_dir())
    if STAGED.exists():
        shutil.rmtree(STAGED)
    STAGED.mkdir(parents=True)
    for piece in pieces:
        shutil.copy2(SYMBTR / f"{piece}.mid", STAGED / f"{piece}.mid")
    print(f"Staged {len(pieces)} test pieces in {STAGED}\n")

    for name, jitter, velocity, seed in MELODIC:
        print(f"=== {name} (octave 0, jitter {jitter}ms, velocity sigma {velocity})")
        run([sys.executable, "scripts/generate_midi.py", "fix-then-synthesize",
             STAGED, FIXED, SYNTH, "--instrument", name, "--octave", 0,
             "--jitter-ms", jitter, "--velocity-sigma", velocity, "--seed", seed,
             "--soundfont-programs", PROGRAMS])

    name, jitter, velocity, seed = PERCUSSION
    print(f"=== {name} (usul cycle, unchanged)")
    run([sys.executable, "scripts/generate_midi.py", "usul-then-synthesize",
         STAGED, FIXED, SYNTH, "--instrument", name, "--jitter-ms", jitter,
         "--velocity-sigma", velocity, "--seed", seed, "--soundfont-programs", PROGRAMS])

    run([sys.executable, "scripts/mix_stem.py", "--verify",
         "--synth-dir", SYNTH, "--out-dir", OUT, "--mirror", REFERENCE])
    print(f"\nUnison test set in {OUT / REFERENCE.name}")
