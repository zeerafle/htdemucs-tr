# Demucs Music Source Separation for Turkish Music Instruments

## Setups

### Setup Env

```bash
uv sync
sudo apt install ffmpeg fluidsynth
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
