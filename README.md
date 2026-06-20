# Demucs Music Source Separation for Turkish Music Instruments

## Setups

### Setup Env

```bash
uv sync
sudo apt install ffmpeg fluidsynth soundstretch
# optional install dvc for data versioning https://doc.dvc.org/install/linux
```

### Synthesize Midis

```bash
uv run scripts/generate_midi.py \
    synthesize-only \
    /teamspace/studios/turkish-music/anatolian-SAM/data/midis/fixed/baglama_saz \
    data/synthesized/ \
    --instrument baglama_saz \
    --soundfonts-dir /teamspace/studios/turkish-music/anatolian-SAM/data/soundfonts \
    --soundfont-programs data/soundfonts_programs_tr.json \
    --limit 75 \
    --randomize
```

### Download MUSDB180-HQ Dataset

```bash
uvx zenodo_get 3338373 -o data
unzip data/musdb18hq -d data/musdb18-hq
```

### Generate and Mix Stems

```bash
uv run scripts/mix_stem.py --n_train 60 --n_test 15 --shuffle_baglama
```

### Run Training Command

```bash
PYTHONWARNINGS="ignore" PYTORCH_NO_CUDA_MEMORY_CACHING=1 uv run dora run -f 955717e8 continue_pretrained="'955717e8'" dset=tr_pilot variant=tr_finetune batch_size=4 dset.segment=7.8 misc.num_workers=4
```

### Export Model

```bash
uv run python -m tools.export a8b66669
```

### Test Model

I don't know, still produces errors, but we can see the result already in the training log

```bash
uv run python -m tools.test_pretrained --repo release_models -n a8b66669 dset=tr_pilot
```

### Run the inference on the mixture audio

```bash
PYTHONPATH=. uv run python -m demucs --repo release_models -n a8b66669 "data/dataset/test/Al James - Schoolboy Facination/mixture.wav" -o separated_pilot/
```
