# Demucs Music Source Separation for Turkish Music Instruments

## Setups

### Setup Env

```bash
# for new environment
# conda env create -f environment-cuda.yml
# for existing environment (e.g. lightning.ai machine) use update
conda env update -f environment-cuda.yml
```

### Synthesize Midis

```bash
python scripts/generate_midi.py \
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
