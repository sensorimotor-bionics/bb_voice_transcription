import argparse
import sys
import traceback
from pathlib import Path
from transcription import transcribe_and_diarize_folder, DEFAULT_SPEAKER_DISTANCE_THRESHOLD

# Mirrors the default extension list in transcribe_and_diarize_folder
MEDIA_EXTENSIONS = [".mp4", ".m4v", ".avi", ".mov", ".mkv", ".wav", ".mp3", ".flac", ".m4a", ".aac"]

TRANSCRIPTS_DIRNAME = "transcripts"


def has_media(folder: Path, extensions: list[str]) -> bool:
    """True if the folder directly contains at least one media file."""
    return any(f.is_file() and f.suffix.lower() in extensions for f in folder.iterdir())


def is_already_processed(folder: Path) -> bool:
    """
    True if the subfolder already has a non-empty transcripts folder.

    Emptiness matters: transcribe_and_diarize_folder creates its output directory
    before doing any work, so a run that crashed partway leaves an empty
    transcripts/ behind. Treating that as "done" would skip it forever.
    """
    transcripts = folder / TRANSCRIPTS_DIRNAME
    return transcripts.is_dir() and any(transcripts.iterdir())


def clear_previous_output(folder: Path) -> list[Path]:
    """
    Remove a subfolder's previous transcript output so a --force rerun cannot leave
    stale transcripts sitting next to fresh ones (they would then be concatenated
    together, and files dropped from the input would linger forever).

    Only files matching the pipeline's own output names are removed, never anything
    else that happens to live in the transcripts folder.
    """
    transcripts = folder / TRANSCRIPTS_DIRNAME
    if not transcripts.is_dir():
        return []

    patterns = ["*_transcript.txt", "*_transcript.json", "batch_summary.json",
                "concatenated_transcript.*", "*.relabeled.json", "*.relabeled.txt"]
    removed = []
    for pattern in patterns:
        for path in sorted(transcripts.glob(pattern)):
            if path.is_file():
                path.unlink()
                removed.append(path)
    return removed


def main():
    parser = argparse.ArgumentParser(
        description="Walk the immediate subfolders of a top-level folder and run transcription "
                    "+ diarization on any subfolder that does not already have a transcripts folder."
    )
    parser.add_argument("root", type=str, help="Path to the top-level folder containing subfolders")
    parser.add_argument("--whisper_size", type=str, default="small",
                        choices=["tiny", "base", "small", "medium", "large-v3"], help="Whisper model size")
    parser.add_argument("--global_num_speakers", type=int, default=None,
                        help="Known number of global speakers within each subfolder (optional). "
                             "When given it is honoured exactly, including for a single long file "
                             "whose speakers are diarized in chunks")
    parser.add_argument("--distance_threshold", type=float, default=DEFAULT_SPEAKER_DISTANCE_THRESHOLD,
                        help="Cosine distance below which two local speakers are merged when "
                             "--global_num_speakers is not given")
    parser.add_argument("--min_speech_duration", type=float, default=0.5,
                        help="Minimum seconds of speech required per file")
    parser.add_argument("--max_audio_length", type=int, default=600,
                        help="Diarize in chunks once a file is longer than this many seconds")
    parser.add_argument("--no_cleanup", action="store_true",
                        help="Keep the intermediate 16 kHz WAV files instead of deleting them")
    parser.add_argument("--force", action="store_true",
                        help="Reprocess subfolders even if they already have a transcripts folder, "
                             "clearing their previous transcript output first")
    parser.add_argument("--dry_run", action="store_true",
                        help="List what would be processed or skipped, then exit without transcribing")
    parser.add_argument("--verbose", action="store_true", help="Enable detailed logging during processing")

    args = parser.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        print(f"Error: Folder '{root}' does not exist or is not a directory.")
        sys.exit(1)

    # Immediate subfolders only. Skip a top-level 'transcripts' dir so output from a
    # prior flat run of batch_process.py is never mistaken for an input subfolder.
    subfolders = sorted([
        d for d in root.iterdir()
        if d.is_dir() and d.name != TRANSCRIPTS_DIRNAME
    ])

    if not subfolders:
        print(f"No subfolders found in {root.resolve()}")
        sys.exit(0)

    print(f"Found {len(subfolders)} subfolder(s) in {root.resolve()}")

    pending, skipped = [], []
    for folder in subfolders:
        print(f"\tChecking {folder.name} ...")
        if is_already_processed(folder) and not args.force:
            skipped.append((folder, "already has transcripts"))
        elif not has_media(folder, MEDIA_EXTENSIONS):
            skipped.append((folder, "no media files"))
        else:
            pending.append(folder)

    for folder, reason in skipped:
        print(f"  SKIP {folder.name} ({reason})")
    for folder in pending:
        print(f"  TODO {folder.name}")

    if args.dry_run:
        print(f"\nDry run: {len(pending)} to process, {len(skipped)} skipped. Nothing was written.")
        return

    if not pending:
        print("\nNothing to do.")
        return

    succeeded, failed = [], []
    for i, folder in enumerate(pending, start=1):
        print(f"\n=== [{i}/{len(pending)}] {folder.name} ===")
        try:
            if args.force:
                removed = clear_previous_output(folder)
                if removed:
                    print(f"  Cleared {len(removed)} stale output file(s) from "
                          f"{folder.name}/{TRANSCRIPTS_DIRNAME}")
            summary = transcribe_and_diarize_folder(
                folder_path=folder,
                output_dir=None,  # defaults to <folder>/transcripts
                whisper_size=args.whisper_size,
                global_num_speakers=args.global_num_speakers,
                distance_threshold=args.distance_threshold,
                min_speech_duration=args.min_speech_duration,
                max_audio_length=args.max_audio_length,
                cleanup=not args.no_cleanup,
                verbose=args.verbose
            )
            succeeded.append((folder, summary))
        except Exception as exc:
            # Keep going: one bad subfolder shouldn't abandon the rest of the batch.
            print(f"  FAILED {folder.name}: {type(exc).__name__}: {exc}")
            if args.verbose:
                traceback.print_exc()
            failed.append((folder, exc))

    print(f"\n{'=' * 60}")
    print(f"Processed {len(succeeded)} subfolder(s), {len(failed)} failed, {len(skipped)} skipped.")
    for folder, summary in succeeded:
        print(f"  OK   {folder.name}: {summary['files_with_speech']}/{summary['total_files']} file(s) with speech")
    for folder, exc in failed:
        print(f"  FAIL {folder.name}: {type(exc).__name__}: {exc}")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
