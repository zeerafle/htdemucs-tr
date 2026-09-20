# Pilot evaluation, 20 Sep 2026

Test set: the pilot `valid` split (15 tracks), symlinked as `data/dataset/test` because the pilot
dataset has no held-out test split. Scored with `museval` via `demucs/evaluate.py`
(`tools/test_pretrained.py ... test.sdr=true`).

| file | what |
|---|---|
| `zeroshot-955717e8.json` | stock pretrained HTDemucs, no fine-tuning |
| `finetuned-a8b66669.json` | pilot fine-tune |
| `floor-mixture-as-estimate.json` | trivial baseline: estimate = mixture |
| `gain-check.json` | measured `mixture = 0.500 * sum(stems)` |

Read these with §4.2 and §4.3 of POST-ABSTRACT-PLAN.md. Two things are not optional context:

1. Zero-shot `sdr_vocals = -0.0006` is **silence**, not poor separation — `new_sdr` returns exactly
   0 for an all-zero estimate. Corroborated by `isr_vocals = 0.013` and `sir_vocals = -30.4`.
2. `gain-check.json` gives g = 0.500 across all 15 tracks, so `mixture != sum(stems)` and every
   number here is capped at -20*log10(1-g) = 6.02 dB. Absolute values are provisional; the
   zero-shot vs fine-tuned ranking is not affected, both were scored under the same handicap.

Per-track JSONs live in the teardown archive at `gs://htdemucs-tr/results/`, not here.
