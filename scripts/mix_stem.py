#!/usr/bin/env python3
"""
mix_stem.py: Build training dataset by replacing MUSDB18 vocals with synthesized bağlama.

For each MUSDB track:
  - Replace vocals.wav with a bağlama wav (cycling through available files)
  - Copy and normalize bass, drums, other stems
  - Render mixture.wav from all four stems
  - Output to data/dataset/{train|valid}/{track_name}/
"""

import argparse
import random
from pathlib import Path

import numpy as np
import soundfile as sf
from pydub import AudioSegment

# --- Paths ---
MUSDB_DIR = Path("/teamspace/studios/htdemucs-tr-data/htdemucs-tr/data/musdb18-hq")
BAGLAMA_DIR = Path(
    "/teamspace/studios/htdemucs-tr-data/htdemucs-tr/data/synthesized/baglama_saz"
)
OUT_DIR = Path("data/dataset")

STEMS = ["bass", "drums", "other"]
TARGET_DBFS = -18.0


# --- Audio helpers ---


def normalize(audio: AudioSegment, target_dbfs: float = TARGET_DBFS) -> AudioSegment:
    """Normalize audio to target RMS level (dBFS)."""
    change = target_dbfs - audio.dBFS
    return audio + change


def match_duration(audio: AudioSegment, reference: AudioSegment) -> AudioSegment:
    """Trim or loop `audio` to exactly match `reference` duration."""
    ref_ms = len(reference)
    if len(audio) >= ref_ms:
        return audio[:ref_ms]
    # Loop until long enough, then trim
    loops = ref_ms // len(audio) + 1
    return (audio * loops)[:ref_ms]


def to_stereo(audio: AudioSegment) -> AudioSegment:
    """Ensure audio is stereo."""
    if audio.channels == 1:
        return audio.set_channels(2)
    return audio


def load(path: Path) -> AudioSegment:
    return to_stereo(AudioSegment.from_file(str(path), format="wav"))


# --- Core mixing ---


def mix_track(musdb_track: Path, baglama_path: Path, out_track: Path) -> None:
    out_track.mkdir(parents=True, exist_ok=True)

    # 1. Get EXACT sample count from the original MUSDB track
    original_bass = musdb_track / "bass.wav"
    exact_samples = sf.info(str(original_bass)).frames

    # Use bass.wav as duration reference for pydub
    reference = load(original_bass)

    # Bağlama → vocals slot
    baglama = load(baglama_path)
    # Ensure baglama is exactly 44100 if it isn't already
    if baglama.frame_rate != reference.frame_rate:
        baglama = baglama.set_frame_rate(reference.frame_rate)

    baglama = match_duration(baglama, reference)
    baglama = normalize(baglama)
    baglama.export(str(out_track / "vocals.wav"), format="wav")

    # Other stems
    mixture = baglama
    for stem in STEMS:
        audio = load(musdb_track / f"{stem}.wav")
        audio = normalize(audio)
        audio.export(str(out_track / f"{stem}.wav"), format="wav")
        mixture = mixture.overlay(audio)

    normalize(mixture).export(str(out_track / "mixture.wav"), format="wav")

    # --- FIX SAMPLE DRIFT ---
    # Read each exported wav and trim/pad it to the exact sample count
    for wav_file in out_track.glob("*.wav"):
        data, sr = sf.read(str(wav_file))

        if len(data) > exact_samples:
            data = data[:exact_samples]  # Trim excess samples
        elif len(data) < exact_samples:
            # Pad with silence if it's slightly too short
            pad_shape = list(data.shape)
            pad_shape[0] = exact_samples - len(data)
            padding = np.zeros(pad_shape, dtype=data.dtype)
            data = np.concatenate((data, padding))

        sf.write(str(wav_file), data, sr)


# --- Dataset builder ---


def process_split(split: str, n_tracks: int, baglama_files: list[Path]) -> None:
    musdb_split = MUSDB_DIR / split
    out_split = OUT_DIR / split

    tracks = sorted(musdb_split.iterdir())
    if n_tracks > 0:
        tracks = tracks[:n_tracks]

    print(f"\n[{split}] {len(tracks)} tracks → {out_split}")
    for i, track in enumerate(tracks):
        baglama = baglama_files[i % len(baglama_files)]
        print(f"  [{i + 1:>3}/{len(tracks)}] {track.name}  +  {baglama.stem}")
        mix_track(track, baglama, out_split / track.name)

    print(f"[{split}] Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Mix MUSDB18-HQ stems with synthesized bağlama audio."
    )
    parser.add_argument(
        "--n_train",
        type=int,
        default=0,
        help="Number of train tracks to render (0 = all, default: 0)",
    )
    parser.add_argument(
        "--n_valid",
        type=int,
        default=0,
        help="Number of valid (test) tracks to render (0 = all, default: 0)",
    )
    parser.add_argument(
        "--shuffle_baglama",
        action="store_true",
        help="Shuffle bağlama assignment instead of cycling in order",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    args = parser.parse_args()

    random.seed(args.seed)

    baglama_files = sorted(BAGLAMA_DIR.glob("*.wav"))
    if not baglama_files:
        raise FileNotFoundError(f"No bağlama WAVs found in {BAGLAMA_DIR}")

    if args.shuffle_baglama:
        random.shuffle(baglama_files)

    print(f"Found {len(baglama_files)} bağlama files.")

    for split, n in [("train", args.n_train), ("valid", args.n_valid)]:
        process_split(split, n, baglama_files)
