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
