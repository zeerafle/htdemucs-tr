"""
generate_midi.py — Batch fix and/or synthesize MIDI files for Turkish instruments.

Modes:
  fix-only            : Rewrite program_change messages in a directory of MIDIs.
  fix-then-synthesize : Fix MIDIs, then render to WAVs via FluidSynth.
  synthesize-only     : Render a directory of (pre-fixed) MIDIs to WAVs without modification.

Example usage:
  # Fix only (first 10 files)
  python scripts/generate_midi.py fix-only input_dir fixed_midi_dir --instrument baglama_saz --limit 10

  # Fix then synthesize (all files)
  python scripts/generate_midi.py fix-then-synthesize input_dir fixed_midi_dir output_wav_dir \
      --instrument baglama_saz --soundfonts-dir data/soundfonts

  # Synthesize only (from pre-fixed MIDIs)
  python scripts/generate_midi.py synthesize-only fixed_midi_dir output_wav_dir \
      --instrument baglama_saz --soundfonts-dir data/soundfonts
"""

import argparse
import json
import os
import random
import sys

import mido
from midi2audio import FluidSynth


def load_instrument(soundfont_programs_path: str, name: str) -> dict:
    with open(soundfont_programs_path) as f:
        data = json.load(f)
    instruments = {e["name"]: e for e in data["soundfonts"]}
    if name not in instruments:
        sys.exit(f"Instrument '{name}' not found. Available: {list(instruments)}")
    return instruments[name]


def fix_midi(input_path: str, output_path: str, program: int) -> None:
    mid = mido.MidiFile(input_path)
    for track in mid.tracks:
        for msg in track:
            if msg.type == "program_change":
                msg.program = program
    mid.save(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch fix and/or synthesize MIDI files for Turkish instruments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "mode",
        choices=["fix-only", "fix-then-synthesize", "synthesize-only"],
    )
    parser.add_argument("input_dir", help="Input directory containing MIDI files.")
    parser.add_argument(
        "output_dir_1",
        help="For fix-only/fix-then-synthesize: directory for the fixed MIDIs. "
        "For synthesize-only: directory for the output WAVs.",
    )
    parser.add_argument(
        "output_dir_2",
        nargs="?",
        help="Output WAV directory (required for fix-then-synthesize).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of MIDI files to process.",
    )
    parser.add_argument(
        "--instrument",
        default="baglama_saz",
        help="Instrument name as defined in soundfonts_programs_tr.json (default: baglama_saz).",
    )
    parser.add_argument(
        "--soundfonts-dir",
        default="data/soundfonts",
        help="Directory containing .sf2 files (required for synthesis modes).",
    )
    parser.add_argument(
        "--soundfont-programs",
        default="soundfonts_programs_tr.json",
        help="Path to soundfonts_programs_tr.json.",
    )
    parser.add_argument(
        "--randomize",
        action="store_true",
        help="Randomize the MIDI track picking (default: False; alphabetically sorted).",
    )

    args = parser.parse_args()

    # 1. Load configuration
    instrument = load_instrument(args.soundfont_programs, args.instrument)
    program = int(instrument["program"])

    # 2. Validate input and ensure output directories exist
    if not os.path.isdir(args.input_dir):
        sys.exit(f"Input directory does not exist: {args.input_dir}")

    os.makedirs(os.path.join(args.output_dir_1, instrument["name"]), exist_ok=True)

    if args.mode == "fix-then-synthesize":
        if not args.output_dir_2:
            sys.exit(
                "fix-then-synthesize requires a third positional argument: output_wav_dir"
            )
        os.makedirs(os.path.join(args.output_dir_2, instrument["name"]), exist_ok=True)

    # 3. Setup FluidSynth once if synthesis is needed
    fs = None
    if args.mode in ["fix-then-synthesize", "synthesize-only"]:
        sf_path = os.path.join(args.soundfonts_dir, instrument["filename"])
        if not os.path.exists(sf_path):
            sys.exit(f"SoundFont not found: {sf_path}")
        fs = FluidSynth(sound_font=sf_path)

    # 4. Gather target MIDI files
    midi_files = [
        f for f in os.listdir(args.input_dir) if f.lower().endswith((".mid", ".midi"))
    ]

    if args.randomize:
        random.shuffle(midi_files)

    if args.limit is not None:
        midi_files = midi_files[: args.limit]

    print(f"Found {len(midi_files)} MIDI files to process in '{args.mode}' mode.")

    # 5. Process loop
    for i, filename in enumerate(midi_files, 1):
        input_path = os.path.join(args.input_dir, filename)
        base_name = os.path.splitext(filename)[0]

        print(f"[{i}/{len(midi_files)}] Processing {filename}...")

        if args.mode == "fix-only":
            out_midi = os.path.join(args.output_dir_1, instrument["name"], filename)
            fix_midi(input_path, out_midi, program)

        elif args.mode == "fix-then-synthesize":
            out_midi = os.path.join(args.output_dir_1, instrument["name"], filename)
            out_wav = os.path.join(
                args.output_dir_2, instrument["name"], f"{base_name}.wav"
            )
            fix_midi(input_path, out_midi, program)
            fs.midi_to_audio(out_midi, out_wav)

        elif args.mode == "synthesize-only":
            out_wav = os.path.join(
                args.output_dir_1, instrument["name"], f"{base_name}.wav"
            )
            fs.midi_to_audio(input_path, out_wav)

    print("Batch processing complete!")
