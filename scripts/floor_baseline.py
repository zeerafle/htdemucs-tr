#!/usr/bin/env python3
"""
floor_baseline.py: the trivial baseline for a results table -- estimate every source as the mixture.

`new_sdr` (demucs/evaluate.py:30) is 10*log10(|ref|^2 / |ref-est|^2), which returns exactly 0 dB
for an all-zero estimate. A zero-shot column full of ~0 therefore reads as "the floor" to anyone
skimming, when it actually means the head emitted silence. Reporting this baseline beside it is
what separates the two readings.

  python scripts/floor_baseline.py data/dataset_tr/test results/ensemble-eval/floor.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

SOURCES = ["drums", "bass", "other", "vocals"]


def new_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Same definition as demucs, delta included, so the numbers sit in one table."""
    delta = 1e-7
    num = float(np.sum(np.square(reference))) + delta
    den = float(np.sum(np.square(reference - estimate))) + delta
    return 10 * np.log10(num / den)


if __name__ == "__main__":
    root = Path(sys.argv[1])
    out = Path(sys.argv[2])

    scores = {source: [] for source in SOURCES}
    tracks = sorted(p for p in root.iterdir() if p.is_dir())
    for track in tracks:
        mixture, _ = sf.read(track / "mixture.wav")
        for source in SOURCES:
            reference, _ = sf.read(track / f"{source}.wav")
            scores[source].append(new_sdr(reference, mixture))

    results = {"note": f"mixture-as-estimate, new_sdr, {len(tracks)} tracks in {root}"}
    for source in SOURCES:
        results[f"nsdr_{source}"] = float(np.mean(scores[source]))
        results[f"nsdr_med_{source}"] = float(np.median(scores[source]))
    results["nsdr"] = float(np.mean([results[f"nsdr_{s}"] for s in SOURCES]))
    results["nsdr_med"] = float(np.median(sum(scores.values(), [])))

    out.write_text(json.dumps(results, indent=1))
    print(json.dumps(results, indent=1))
