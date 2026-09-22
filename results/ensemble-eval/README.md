# Ensemble evaluation, 21 Sep 2026

Test set: `data/dataset_tr/test`, 15 held-out tracks (proper held-out split, unlike the pilot).
Scored with `museval` via `demucs/evaluate.py`:

```
uv run python -m tools.test_pretrained -n 955717e8 dset=tr_ensemble \
    test.sdr=true test.workers=0 misc.num_workers=2
```

`test.workers=0` is required — see the 21 Sep correction in §4.1 POST-ABSTRACT-PLAN.md and commit
`09d53c7`. Without it the test stage is OOM-killed with no traceback.

| file | what |
|---|---|
| `zeroshot-955717e8.json` | stock pretrained HTDemucs, no fine-tuning — **the go/no-go gate** |

Slot mapping: `vocals` = bağlama (target), `other` = ney+kanun, `bass` = ud, `drums` = bendir.

Two things that are not optional context:

1. `nsdr_vocals = 0.00060`, `isr_vocals = 0.00087` — this is **silence**, not poor separation.
   `new_sdr` (`demucs/evaluate.py:30`) is `10·log10(‖ref‖²/‖ref−est‖²)`, so an all-zero estimate
   returns exactly 0 dB. Stock HTDemucs looks for singing voice, finds none, emits zero. Same
   signature as the pilot (§4.2). The headroom on the bağlama head is therefore total, and any
   "X dB vs zero-shot" headline measures **an unused head becoming a used head**, not makam
   adaptation. The mixture-as-estimate floor is the comparator that belongs beside it.
2. The other three heads are *worse than doing nothing*: `nsdr_bass = −2.02`, `nsdr_other = −3.13`,
   mean `nsdr = −1.05`. Only `nsdr_drums = +0.93` is positive, and its `sir_drums = 22.6` says the
   bendir is separable on a low-frequency shortcut (97% of its energy is sub-200 Hz, §3.7) rather
   than on timbre. `sir_bass = −23.8` says the stock bass head is grabbing that same sub-bass
   instead of the ud. This is direct evidence for the §2 claim that the ensemble supplies the
   headroom the pilot lacked.

Per-track JSONs live in the teardown archive `gs://htdemucs-tr/results/`, not here.

---

## Fine-tune `10b843f9` + unison ablation, 22 Sep 2026

| file | what |
|---|---|
| `finetuned-10b843f9.json` | fine-tuned model, `data/dataset_tr/test` |
| `finetuned-10b843f9-unison.json` | same model, `data/dataset_tr_unison/test` (§3.6 ablation) |

Both at `test.shifts=1`, matching the zero-shot gate, so all three columns are comparable. The
epoch-50 in-training test summary in `train.log` is **not** — `-f 955717e8` inherits
`test.shifts=0`.

nSDR, mean over 15 tracks:

| head | instrument | zero-shot | fine-tuned | unison | Δ ablation |
|---|---|---|---|---|---|
| `vocals` | **bağlama (target)** | 0.0006 | **16.24** | **8.68** | **−7.56** |
| `bass` | ud | −2.02 | 17.19 | 8.36 | −8.83 |
| `other` | ney + kanun | −3.13 | 19.60 | 10.29 | −9.32 |
| `drums` | bendir | +0.93 | 21.23 | 18.03 | −3.20 |
| mean | | −1.05 | 18.57 | 11.34 | −7.23 |

**The ablation is paired on the target and unpaired elsewhere.** `scripts/generate_midi.py:411`
builds one `random.Random(seed)` and draws from it across the whole staged set, so the jitter and
velocity realizations depend on how many files are staged — 90 for the main dataset, 15 for the
unison set. Every stem with nonzero jitter is therefore a *different draw* in the two sets, not the
same audio.

The bağlama is the exception, and it is the exception that matters: it renders at
`jitter=0, velocity_sigma=0`, so the RNG never touches it. Verified directly — regressing the
unison `vocals.wav` on the main one gives a residual of **3.0e-05** (the PCM_16 quantization floor)
at a per-track gain of 0.863–0.927, the gain being the shared normalizer in `mix_stem.py` responding
to a different mixture peak.

So for the target head the comparison is exact: **identical reference audio, only the surrounding
mixture moved.** −7.56 dB is a clean measurement. For `bass`, `other` and `drums` the reference
itself changed, so those rows are same-distribution comparisons, not paired ones — informative,
but do not report them as controls.

---

## The trivial floor, 22 Sep 2026

`floor-mixture-as-estimate.json`, `floor-mixture-as-estimate-unison.json`, from
`scripts/floor_baseline.py`. Estimate every source as the mixture.

**−4.78 dB**, near-identical across all four sources and both test sets. That is the analytic value
for four equal-power, mutually incoherent sources: `ref − est = −Σ(others)`, so the ratio is
`1/3` and `10·log10(1/3) = −4.77`. Getting it to two decimals is a check on the dataset — it
confirms the stems really are RMS-matched and that `mixture == Σ stems` holds.

This settles a reading that §4.2 got only half right. The pilot's floor came out at ≈ 0 dB, which
made the zero-shot `sdr_vocals` of −0.0006 look like it was sitting exactly on the floor. It wasn't
a coincidence and it wasn't the floor — the pilot mixtures had `mixture = 0.5 · Σ stems` (§4.3), and
with that gain the arithmetic gives `‖0.5·ref − 0.5·Σothers‖² = ‖ref‖²`, i.e. 0 dB. The defect
*was* the coincidence.

With the gain fixed, the two baselines separate by 4.77 dB and the results table has three distinct
reference points:

| head | instrument | floor (mixture) | silence | zero-shot | fine-tuned | unison |
|---|---|---|---|---|---|---|
| `vocals` | **bağlama** | −4.77 | 0.00 | 0.0006 | **16.24** | **8.68** |
| `bass` | ud | −4.77 | 0.00 | −2.02 | 17.19 | 8.36 |
| `other` | ney + kanun | −4.79 | 0.00 | −3.13 | 19.60 | 10.29 |
| `drums` | bendir | −4.79 | 0.00 | +0.93 | 21.23 | 18.03 |
| mean | | −4.78 | 0.00 | −1.05 | 18.57 | 11.34 |

Emitting **silence beats emitting the mixture by 4.77 dB** when the sources are equal-power. So
stock HTDemucs' 0.0006 on the bağlama is not "no better than trivial" — it is 4.77 dB better than
the trivial baseline, and it achieves that by outputting nothing. Say this explicitly; a reviewer
who assumes 0 dB is the floor will misread the entire zero-shot column.
