import argparse
import sys
from pathlib import Path
from transcription import (transcribe_and_diarize_folder, DEFAULT_SPEAKER_DISTANCE_THRESHOLD,
                           MIN_SPEAKER_SPEECH_SECONDS)

def main():
    parser = argparse.ArgumentParser(
        description="Batch process a folder of video/audio files with speech detection, diarization, and cross-file speaker classification."
    )
    parser.add_argument("folder", type=str, help="Path to the folder containing video/audio files")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for transcripts and summary report")
    parser.add_argument("--whisper_size", type=str, default="small", choices=["tiny", "base", "small", "medium", "large-v3"], help="Whisper model size")
    parser.add_argument("--global_num_speakers", type=int, default=None, help="Known number of global speakers across all files (optional)")
    parser.add_argument("--distance_threshold", type=float, default=DEFAULT_SPEAKER_DISTANCE_THRESHOLD,
                        help="Cosine distance below which two local speakers are merged when --global_num_speakers is not given")
    parser.add_argument("--min_speech_duration", type=float, default=0.5, help="Minimum seconds of speech required per file")
    parser.add_argument("--max_audio_length", type=int, default=600,
                        help="Diarize in chunks once a file is longer than this many seconds")
    parser.add_argument("--min_speaker_speech", type=float, default=MIN_SPEAKER_SPEECH_SECONDS,
                        help="Speech a local speaker needs before it may define a speaker of its own")
    parser.add_argument("--no_cleanup", action="store_true",
                        help="Keep the intermediate 16 kHz WAV files instead of deleting them")
    parser.add_argument("--verbose", action="store_true", help="Enable detailed logging during processing")

    args = parser.parse_args()

    folder_path = Path(args.folder)
    if not folder_path.exists() or not folder_path.is_dir():
        print(f"Error: Folder '{folder_path}' does not exist or is not a directory.")
        sys.exit(1)

    print(f"Starting batch processing on: {folder_path.resolve()}")
    summary = transcribe_and_diarize_folder(
        folder_path=folder_path,
        output_dir=args.output_dir,
        whisper_size=args.whisper_size,
        global_num_speakers=args.global_num_speakers,
        distance_threshold=args.distance_threshold,
        min_speech_duration=args.min_speech_duration,
        max_audio_length=args.max_audio_length,
        min_speaker_speech=args.min_speaker_speech,
        cleanup=not args.no_cleanup,
        verbose=args.verbose
    )
    print(summary)
    print("Batch processing completed successfully.")

if __name__ == "__main__":
    main()
