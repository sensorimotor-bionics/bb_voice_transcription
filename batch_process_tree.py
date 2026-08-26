import argparse
import sys
import traceback
from pathlib import Path
from transcription import (transcribe_and_diarize_folder, DEFAULT_SPEAKER_DISTANCE_THRESHOLD,
                           MIN_SPEAKER_SPEECH_SECONDS, SPEAKER_COUNT_POLICIES)
from speaker_roles import add_role_arguments, roles_requested, label_from_args
from audio_utils import (find_leftover_prepared_audio, remove_leftover_prepared_audio,
                         MEDIA_EXTENSIONS)

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
                "concatenated_transcript.*", "*.relabeled.json", "*.relabeled.txt",
                "*.speaker_roles.json"]
    removed = []
    for pattern in patterns:
        for path in sorted(transcripts.glob(pattern)):
            if path.is_file():
                path.unlink()
                removed.append(path)
    return removed


def sweep_intermediates(folders: list[Path], dry_run: bool = False, verbose: bool = False) -> int:
    """
    Remove intermediate 16 kHz WAVs left behind in the given folders.

    transcribe_and_diarize_folder deletes its own intermediates as it goes, but plenty of
    them survive that: a run killed mid-file, a run made with --no_cleanup, or a
    subfolder that is skipped as already processed and so never revisited. Those strays
    would otherwise sit next to the media forever, so the whole tree is swept once at the
    end regardless of whether this run touched each subfolder.

    Returns:
        int: How many files were removed, or would be removed when dry_run is set.
    """
    total = 0
    for folder in folders:
        if dry_run:
            affected = find_leftover_prepared_audio(folder, MEDIA_EXTENSIONS)
        else:
            affected = remove_leftover_prepared_audio(folder, MEDIA_EXTENSIONS)
        if not affected:
            continue
        total += len(affected)
        print(f"  {'Would remove' if dry_run else 'Removed'} {len(affected)} "
              f"intermediate file(s) in {folder.name}")
        if verbose:
            for path in affected:
                print(f"    {path.name}")
    return total


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
    parser.add_argument("--min_speaker_speech", type=float, default=MIN_SPEAKER_SPEECH_SECONDS,
                        help="Speech a local speaker needs before it may define a speaker of its own")
    parser.add_argument("--speaker_count_policy", choices=SPEAKER_COUNT_POLICIES, default="given",
                        help="When --global_num_speakers disagrees with the count the embeddings "
                             "support: ask, keep the given number, or take the predicted one. "
                             "Defaults to 'given' here, since a tree walk is usually unattended")
    parser.add_argument("--no_cleanup", action="store_true",
                        help="Keep the intermediate 16 kHz WAV files instead of deleting them, "
                             "and skip the final sweep for leftovers from earlier runs")
    parser.add_argument("--cleanup_only", action="store_true",
                        help="Only delete leftover intermediate 16 kHz WAV files from the tree, "
                             "then exit without transcribing anything")
    parser.add_argument("--force", action="store_true",
                        help="Reprocess subfolders even if they already have a transcripts folder, "
                             "clearing their previous transcript output first")
    parser.add_argument("--dry_run", action="store_true",
                        help="List what would be processed or skipped, then exit without transcribing")
    parser.add_argument("--verbose", action="store_true", help="Enable detailed logging during processing")
    add_role_arguments(parser)

    args = parser.parse_args()

    if args.cleanup_only and args.no_cleanup:
        print("Error: --cleanup_only and --no_cleanup ask for opposite things.")
        sys.exit(1)

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

    # The root itself is swept too, so intermediates from a flat batch_process.py run
    # over the same tree are not left behind either.
    sweep_targets = [root] + subfolders

    if args.cleanup_only:
        print(f"Sweeping {len(sweep_targets)} folder(s) under {root.resolve()} for "
              "leftover intermediate files...")
        total = sweep_intermediates(sweep_targets, dry_run=args.dry_run, verbose=True)
        if not total:
            print("  No leftover intermediate files found.")
        elif args.dry_run:
            print(f"\nDry run: {total} intermediate file(s) would be removed. Nothing was deleted.")
        else:
            print(f"\nRemoved {total} intermediate file(s).")
        return

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
        if roles_requested(args):
            print(f"Speaker roles would be labelled per subfolder with {args.role_model}.")
        if not args.no_cleanup:
            if not sweep_intermediates(sweep_targets, dry_run=True, verbose=args.verbose):
                print("No leftover intermediate files to clean up.")
        return

    if not pending:
        print("\nNothing to do.")
        return

    succeeded, failed, unlabelled = [], [], []
    for i, folder in enumerate(pending, start=1):
        print(f"\n=== [{i}/{len(pending)}] {folder.name} ===")
        try:
            if args.force:
                removed = clear_previous_output(folder)
                if removed and args.verbose:
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
                min_speaker_speech=args.min_speaker_speech,
                speaker_count_policy=args.speaker_count_policy,
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
            continue
        finally:
            # Sweep as we go rather than only at the end, so a batch that is interrupted
            # or crashes halfway does not leave the subfolders it already finished
            # littered with intermediates.
            if not args.no_cleanup:
                sweep_intermediates([folder], verbose=args.verbose)

        # Each subfolder is one session, so roles are decided per subfolder. This runs
        # after the transcripts are safely on disk and cannot undo them: a stopped Ollama
        # server costs the labels, not hours of transcription.
        if roles_requested(args):
            try:
                label_from_args(summary["output_dir"], args, verbose=args.verbose)
            except (RuntimeError, ValueError) as exc:
                print(f"  Speaker role labelling failed for {folder.name}: {exc}")
                unlabelled.append((folder, exc))

    print(f"\n{'=' * 60}")
    print(f"Processed {len(succeeded)} subfolder(s), {len(failed)} failed, {len(skipped)} skipped.")
    for folder, summary in succeeded:
        print(f"  OK   {folder.name}: {summary['files_with_speech']}/{summary['total_files']} file(s) with speech")
    for folder, exc in failed:
        print(f"  FAIL {folder.name}: {type(exc).__name__}: {exc}")
    if unlabelled:
        # Not counted as a failure: these subfolders have their transcripts, only the
        # interviewer/participant labels are missing, and label_speakers.py can add them.
        print(f"\n{len(unlabelled)} subfolder(s) were transcribed but not role-labelled:")
        for folder, exc in unlabelled:
            print(f"  UNLABELLED {folder.name}: {exc}")

    # Final pass over the whole tree, including the subfolders this run skipped: those are
    # never revisited, so leftovers from an earlier interrupted run would live there
    # forever otherwise.
    if not args.no_cleanup:
        total = sweep_intermediates(sweep_targets, verbose=args.verbose)
        if total:
            print(f"\nCleanup: removed {total} leftover intermediate file(s).")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
