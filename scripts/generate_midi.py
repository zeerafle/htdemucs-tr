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

Heterophonic variation (--octave / --jitter-ms / --velocity-sigma, see vary_midi):
rendering one SymbTr line through several soundfonts unmodified gives stems with identical
pitch content, so the separation ground truth is ambiguous. Give each instrument its own
register, attack spread and seed instead -- roughly how the ensemble is scored in practice:

  baglama_saz  --octave  0 --jitter-ms  0 --velocity-sigma 0 --seed 1   (lead, on the grid)
  kanun        --octave  0 --jitter-ms 20 --velocity-sigma 6 --seed 2
  ney          --octave  1 --jitter-ms 25 --velocity-sigma 8 --seed 3
  ud           --octave -1 --jitter-ms 15 --velocity-sigma 5 --seed 4

  python scripts/generate_midi.py fix-then-synthesize input_dir fixed_midi_dir output_wav_dir \
      --instrument ney --octave 1 --jitter-ms 25 --velocity-sigma 8 --seed 3
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys

import mido
import numpy as np
import soundfile as sf


def render_wav(soundfont: str, midi_path: str, out_wav: str, samplerate: int = 44100) -> None:
    """Render `midi_path` through `soundfont` with FluidSynth.

    Deliberately not midi2audio: it passes `-F` after the file arguments, which FluidSynth
    2.x rejects ("'-F' is an illegal option at this place") while still exiting 0 -- so a
    whole batch reports success and writes no audio at all.
    """
    result = subprocess.run(
        ["fluidsynth", "-ni", "-F", out_wav, "-r", str(samplerate), soundfont, midi_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0 or not os.path.exists(out_wav):
        sys.exit(f"FluidSynth failed on {midi_path}:\n{result.stderr.decode(errors='replace')}")
    center(out_wav)


def center(wav_path: str) -> None:
    """Collapse a render to mono and duplicate it, so the instrument sits dead centre.

    Every soundfont here is a single mono instrument, but a preset may carry a pan generator
    that FluidSynth honours: `HAS bendir.sf2` is hard right, and rendered the whole percussion
    part with a digitally silent left channel while the melodic soundfonts came out L == R.
    A stem that lives on one side hands the separator a channel shortcut with nothing to do
    with timbre, and leaves the mixture's other side missing that instrument entirely.

    Centre here rather than at mix time: `data/synthesized/` is auditioned and reused on its
    own, so it should be correct on disk instead of relying on a downstream consumer to
    compensate. Sending CC10=0 in the MIDI would also centre this preset exactly, but only
    because it happens to be panned right -- the same message pushes a left-panned soundfont
    further left. Averaging the channels is soundfont-agnostic and a no-op for a centred one.
    """
    audio, samplerate = sf.read(wav_path, always_2d=True)
    if audio.shape[1] == 1:
        return
    mono = audio.mean(axis=1, keepdims=True)
    sf.write(wav_path, np.repeat(mono, 2, axis=1), samplerate, subtype=sf.info(wav_path).subtype)


def load_instrument(soundfont_programs_path: str, name: str) -> dict:
    with open(soundfont_programs_path) as f:
        data = json.load(f)
    instruments = {e["name"]: e for e in data["soundfonts"]}
    if name not in instruments:
        sys.exit(f"Instrument '{name}' not found. Available: {list(instruments)}")
    return instruments[name]


def _first_tempo(mid: mido.MidiFile) -> int:
    for track in mid.tracks:
        for msg in track:
            if msg.type == "set_tempo":
                return msg.tempo
    return 500000  # mido default, 120 bpm


def _jitter(events, max_shift: float, rng: random.Random) -> None:
    """Spread note onsets over `events` (a list of [abs_tick, message]), in place.

    A note_off carries its note_on's shift, so durations survive untouched. Two adjacent
    notes of the same pitch must not cross, though -- FluidSynth would read the earlier
    note's note_off as ending the later one, dropping it -- so each shift is clamped
    against the previous note of that pitch.

    SymbTr writes each pitchwheel immediately before the note_on it bends, so a bend travels
    with the next note it precedes. That keeps the makam microtonality attached to
    its note, and keeps a bend from firing after the note it was meant to tune -- including
    the bends that sit on SymbTr's silent rest markers (note 0, velocity 0).
    """
    paired, pending = [], {}
    for i, (when, msg) in enumerate(events):
        if msg.type == "note_on" and msg.velocity > 0:
            pending[(msg.channel, msg.note)] = (i, when)
        elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
            held = pending.pop((msg.channel, msg.note), None)
            if held is not None:
                paired.append((held[0], i, held[1], when, (msg.channel, msg.note)))
    paired.sort(key=lambda note: note[2])

    # Work out what travels with each note before shifting anything: a note may not move
    # earlier than the first of its own bends, or that bend would land on a negative tick.
    carried, upcoming = {}, {}
    for i in range(len(events) - 1, -1, -1):
        msg = events[i][1]
        if msg.type == "note_on" and msg.velocity > 0:
            upcoming[msg.channel] = i
        elif msg.type in ("pitchwheel", "control_change"):
            owner = upcoming.get(msg.channel)
            if owner is not None:
                carried.setdefault(owner, []).append(i)

    shifts, previous = {}, {}
    for on_i, off_i, on_t, off_t, key in paired:
        # 3-sigma lands on jitter_ms, so the flag reads as a practical bound.
        delta = int(round(rng.gauss(0.0, max_shift / 3.0)))
        delta = min(max(delta, -max_shift), max_shift)
        last = previous.get(key)
        if last is not None:
            delta = max(delta, last[1] - (on_t - last[0]))  # no crossing
        floor = min([on_t] + [events[j][0] for j in carried.get(on_i, ())])
        delta = int(max(delta, -floor))  # no negative absolute time
        shifts[on_i] = shifts[off_i] = delta
        previous[key] = (off_t, delta)

    for on_i, indices in carried.items():
        for i in indices:
            shifts[i] = shifts.get(on_i, 0)

    for i, delta in shifts.items():
        events[i][0] += delta
    events.sort(key=lambda event: event[0])  # stable: ties keep original order


def vary_midi(
    mid: mido.MidiFile,
    octave: int = 0,
    jitter_ms: float = 0.0,
    velocity_sigma: float = 0.0,
    rng: random.Random | None = None,
) -> None:
    """Apply per-instrument heterophonic variation to `mid`, in place.

    Several instruments rendered from one SymbTr line would otherwise be pitch-identical
    and differ only in timbre, which makes the separation ground truth ambiguous. This
    transposes by whole octaves, spreads note onsets, and varies velocities so the stems
    differ the way a real makam ensemble does.

    SymbTr v3 bakes makam microtonality into pitchwheel events, so a pitch bend set at a
    note's onset tick is shifted with that note and never independently -- otherwise the
    ornaments detune. Note-offs take their note-on's shift, preserving duration.
    """
    rng = rng or random.Random()
    max_shift = (
        mido.second2tick(jitter_ms / 1000.0, mid.ticks_per_beat, _first_tempo(mid))
        if jitter_ms
        else 0.0
    )

    for track in mid.tracks:
        events, now, end = [], 0, None
        for msg in track:
            now += msg.time
            if msg.type == "end_of_track":
                end = [now, msg]  # held aside; it must stay last
            else:
                events.append([now, msg])

        if max_shift:
            _jitter(events, max_shift, rng)

        for _, msg in events:
            if msg.type in ("note_on", "note_off"):
                if octave:
                    msg.note = max(0, min(127, msg.note + 12 * octave))
                if velocity_sigma and msg.type == "note_on" and msg.velocity > 0:
                    jittered = msg.velocity + rng.gauss(0.0, velocity_sigma)
                    msg.velocity = max(1, min(127, int(round(jittered))))

        if end is not None:
            # Keep the original trailing silence unless a note jittered past it, so the
            # instrument renders stay the same length.
            end[0] = max(end[0], events[-1][0] if events else 0)
            events.append(end)

        track.clear()
        previous = 0
        for when, msg in events:
            msg.time = when - previous
            previous = when
            track.append(msg)


# Usul (rhythmic cycle) patterns as (stroke, zaman) pairs, "D" = dum (strong, low),
# "T" = tek (weak, sharp). Vocables follow the standard readings: sofyan "Duum te ke",
# duyek "Dum teek tek duuum teek", curcuna "Duum te kaa duum teek tek"; aksak and
# aksaksemai from their 2+2+2+3 and 2+2+2+2+2 groupings. Agiraksak is aksak at 9/4.
#
# These eight cover 2014 of the 3000 SymbTr pieces. Each pattern's zaman total is checked
# against the piece's own time signature at build time -- 2005 of those 2014 agree, and
# the nine that do not are corpus disagreements between filename and meter, skipped rather
# than rendered against the wrong bar line.
USULS = {
    "nimsofyan": [("D", 1), ("T", 1)],
    "semai": [("D", 1), ("T", 1), ("T", 1)],
    "sofyan": [("D", 2), ("T", 1), ("T", 1)],
    "duyek": [("D", 1), ("T", 2), ("T", 1), ("D", 2), ("T", 2)],
    "aksak": [("D", 2), ("T", 2), ("T", 2), ("T", 3)],
    "agiraksak": [("D", 2), ("T", 2), ("T", 2), ("T", 3)],
    "aksaksemai": [("D", 2), ("T", 2), ("T", 2), ("T", 2), ("T", 2)],
    "curcuna": [("D", 2), ("T", 1), ("T", 2), ("D", 2), ("T", 2), ("T", 1)],
}

# Stroke keys in HAS bendir.sf2 (bank 0, preset 0).
DUM, TEK = 27, 29
DUM_VELOCITY, TEK_VELOCITY = 100, 72


def usul_from_filename(filename: str) -> str:
    """SymbTr names are {makam}--{form}--{usul}--{name}--{composer}."""
    parts = os.path.basename(filename).split("--")
    return parts[2] if len(parts) > 2 else ""


def build_usul(source: mido.MidiFile, usul: str, program: int) -> mido.MidiFile:
    """Return a one-track MIDI looping `usul` for the length of `source`.

    The usul is generated rather than sourced -- SymbTr carries only the melodic line, and
    the percussion stem needs to be rhythmically independent of it (POST-ABSTRACT-PLAN
    §3.4). Tempo and meter changes are copied across so the cycle tracks the piece.

    Raises ValueError if the piece's meter contradicts the usul named in its filename.
    """
    pattern = USULS[usul]
    zamanlar = sum(beats for _, beats in pattern)
    meter = next((m for tr in source.tracks for m in tr if m.type == "time_signature"), None)
    if meter is None:
        raise ValueError("no time_signature to place the cycle against")
    if meter.numerator != zamanlar:
        raise ValueError(
            f"{usul} is {zamanlar} zaman but the piece is "
            f"{meter.numerator}/{meter.denominator}"
        )
    zaman = source.ticks_per_beat * 4 // meter.denominator

    events = [(0, mido.Message("program_change", program=program, time=0))]
    for track in source.tracks:
        now = 0
        for msg in track:
            now += msg.time
            if msg.type in ("set_tempo", "time_signature"):
                events.append((now, msg.copy()))

    total = max(sum(msg.time for msg in track) for track in source.tracks)
    at = 0
    while at < total:
        for stroke, beats in pattern:
            if at >= total:
                break
            note = DUM if stroke == "D" else TEK
            velocity = DUM_VELOCITY if stroke == "D" else TEK_VELOCITY
            # Never overlap the next stroke, and never ring past the end of the piece --
            # an overhanging stroke would leave the mixture drums-only at the tail.
            gate = max(1, min(beats * zaman - 1, total - at))
            events.append((at, mido.Message("note_on", note=note, velocity=velocity, time=0)))
            events.append((at + gate, mido.Message("note_off", note=note, velocity=0, time=0)))
            at += beats * zaman

    events.append((total, mido.MetaMessage("end_of_track", time=0)))
    events.sort(key=lambda event: event[0])  # stable: ties keep insertion order
    out = mido.MidiFile(ticks_per_beat=source.ticks_per_beat)
    track = mido.MidiTrack()
    previous = 0
    for when, msg in events:
        msg.time = when - previous
        previous = when
        track.append(msg)
    out.tracks.append(track)
    return out


def fix_midi(
    input_path: str,
    output_path: str,
    program: int,
    octave: int = 0,
    jitter_ms: float = 0.0,
    velocity_sigma: float = 0.0,
    rng: random.Random | None = None,
) -> None:
    mid = mido.MidiFile(input_path)
    for track in mid.tracks:
        for msg in track:
            if msg.type == "program_change":
                msg.program = program
    if octave or jitter_ms or velocity_sigma:
        vary_midi(mid, octave, jitter_ms, velocity_sigma, rng)
    mid.save(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch fix and/or synthesize MIDI files for Turkish instruments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "mode",
        choices=[
            "fix-only",
            "fix-then-synthesize",
            "synthesize-only",
            "usul-only",
            "usul-then-synthesize",
        ],
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
        "--octave",
        type=int,
        default=0,
        help="Transpose by this many octaves (ney +1, ud -1, baglama/kanun 0).",
    )
    parser.add_argument(
        "--jitter-ms",
        type=float,
        default=0.0,
        help="Onset spread in milliseconds, ~3 sigma. Pitch bends move with their note.",
    )
    parser.add_argument(
        "--velocity-sigma",
        type=float,
        default=0.0,
        help="Per-note velocity jitter, in MIDI velocity units.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for the variation RNG, so a render is reproducible per instrument.",
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
    # Seeded per instrument, so re-rendering one stem reproduces it note for note.
    rng = random.Random(args.seed)

    # 2. Validate input and ensure output directories exist
    if not os.path.isdir(args.input_dir):
        sys.exit(f"Input directory does not exist: {args.input_dir}")

    os.makedirs(os.path.join(args.output_dir_1, instrument["name"]), exist_ok=True)

    if args.mode in ("fix-then-synthesize", "usul-then-synthesize"):
        if not args.output_dir_2:
            sys.exit(f"{args.mode} requires a third positional argument: output_wav_dir")
        os.makedirs(os.path.join(args.output_dir_2, instrument["name"]), exist_ok=True)

    # 3. Locate the soundfont once if synthesis is needed
    sf_path = None
    if args.mode in ["fix-then-synthesize", "synthesize-only", "usul-then-synthesize"]:
        sf_path = os.path.join(args.soundfonts_dir, instrument["filename"])
        if not os.path.exists(sf_path):
            sys.exit(f"SoundFont not found: {sf_path}")

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
    skipped = []
    for i, filename in enumerate(midi_files, 1):
        input_path = os.path.join(args.input_dir, filename)
        base_name = os.path.splitext(filename)[0]

        print(f"[{i}/{len(midi_files)}] Processing {filename}...")

        if args.mode == "fix-only":
            out_midi = os.path.join(args.output_dir_1, instrument["name"], filename)
            fix_midi(
                input_path,
                out_midi,
                program,
                args.octave,
                args.jitter_ms,
                args.velocity_sigma,
                rng,
            )

        elif args.mode == "fix-then-synthesize":
            out_midi = os.path.join(args.output_dir_1, instrument["name"], filename)
            out_wav = os.path.join(
                args.output_dir_2, instrument["name"], f"{base_name}.wav"
            )
            fix_midi(
                input_path,
                out_midi,
                program,
                args.octave,
                args.jitter_ms,
                args.velocity_sigma,
                rng,
            )
            render_wav(sf_path, out_midi, out_wav)

        elif args.mode == "synthesize-only":
            out_wav = os.path.join(
                args.output_dir_1, instrument["name"], f"{base_name}.wav"
            )
            render_wav(sf_path, input_path, out_wav)

        elif args.mode in ("usul-only", "usul-then-synthesize"):
            usul = usul_from_filename(filename)
            if usul not in USULS:
                print(f"    skipped: no pattern for usul '{usul}'")
                skipped.append(filename)
                continue
            try:
                usul_mid = build_usul(mido.MidiFile(input_path), usul, program)
            except ValueError as err:
                print(f"    skipped: {err}")
                skipped.append(filename)
                continue
            # A drummer is no more metronomic than the melodists; reuse the same variation.
            if args.jitter_ms or args.velocity_sigma:
                vary_midi(usul_mid, 0, args.jitter_ms, args.velocity_sigma, rng)
            out_midi = os.path.join(args.output_dir_1, instrument["name"], filename)
            usul_mid.save(out_midi)
            if args.mode == "usul-then-synthesize":
                out_wav = os.path.join(
                    args.output_dir_2, instrument["name"], f"{base_name}.wav"
                )
                render_wav(sf_path, out_midi, out_wav)

    if skipped:
        print(f"\nSkipped {len(skipped)} of {len(midi_files)} files (see reasons above).")
    print("Batch processing complete!")
