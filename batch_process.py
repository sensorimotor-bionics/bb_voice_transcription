import argparse
import sys
from pathlib import Path
from transcription import transcribe_and_diarize_folder

def main():
    parser = argparse.ArgumentParser(
        description="Batch process a folder of video/audio files with speech detection, diarization, and cross-file speaker classification."
    )
    parser.add_argument("folder", type=str, help="Path to the folder containing video/audio files")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for transcripts and summary report")
    parser.add_argument("--whisper_size", type=str, default="small", choices=["tiny", "base", "small", "medium", "large-v3"], help="Whisper model size")
    parser.add_argument("--global_num_speakers", type=int, default=None, help="Known number of global speakers across all files (optional)")
    parser.add_argument("--distance_threshold", type=float, default=0.75, help="Agglomerative clustering distance threshold")
    parser.add_argument("--min_speech_duration", type=float, default=0.5, help="Minimum seconds of speech required per file")
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
        verbose=args.verbose
    )
    print("Batch processing completed successfully.")

if __name__ == "__main__":
    main()
