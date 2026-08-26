import argparse
import sys
from pathlib import Path
from speaker_roles import (label_speaker_roles, DEFAULT_MODEL, DEFAULT_HOST,
                           DEFAULT_CHUNK_CHARS, INTERVIEWER_LABEL, INTERVIEWEE_LABEL)


def main():
    parser = argparse.ArgumentParser(
        description="Label diarized speakers as interviewer or interviewee by reading the "
                    "transcript with a local LLM served by Ollama (Gemma by default)."
    )
    parser.add_argument("transcript", type=str,
                       help="Concatenated transcript JSON, batch summary JSON, or a transcripts folder")
    parser.add_argument("--num_interviewees", type=int, default=None,
                       help="Assert how many speakers are interviewees. Recommended whenever there "
                            "are more than two speakers; inferred from the model's votes if omitted")
    parser.add_argument("--num_interviewers", type=int, default=None,
                       help="Assert how many speakers are interviewers instead (must agree with "
                            "--num_interviewees when both are given)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                       help="Ollama model tag, e.g. gemma3:12b or gemma3:4b")
    parser.add_argument("--host", type=str, default=DEFAULT_HOST, help="Ollama base URL")
    parser.add_argument("--chunk_chars", type=int, default=DEFAULT_CHUNK_CHARS,
                       help="Transcript characters sent per LLM call")
    parser.add_argument("--interviewer_label", type=str, default=INTERVIEWER_LABEL,
                       help="Name given to interviewers (numbered when there are several)")
    parser.add_argument("--interviewee_label", type=str, default=INTERVIEWEE_LABEL,
                       help="Name given to interviewees (numbered when there are several)")
    parser.add_argument("--output_path", type=str, default=None,
                       help="Where to write the relabeled transcript; defaults to beside the input")
    parser.add_argument("--dry_run", action="store_true",
                       help="Report the roles without writing a relabeled transcript")
    parser.add_argument("--verbose", action="store_true", help="Print per-chunk votes and pooled scores")

    args = parser.parse_args()

    transcript_path = Path(args.transcript)
    if not transcript_path.exists():
        print(f"Error: '{transcript_path}' does not exist.")
        sys.exit(1)

    try:
        _, report = label_speaker_roles(
            transcript_path,
            num_interviewees=args.num_interviewees,
            num_interviewers=args.num_interviewers,
            model=args.model,
            host=args.host,
            chunk_chars=args.chunk_chars,
            interviewer_label=args.interviewer_label,
            interviewee_label=args.interviewee_label,
            output_path=args.output_path,
            relabel=not args.dry_run,
            verbose=args.verbose,
        )
    except (RuntimeError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)

    print("\nSpeaker roles:")
    for spk, role in report["roles"].items():
        name = report["mapping"].get(spk, spk)
        print(f"  {spk} -> {name} ({role}, score {report['scores'][spk]:.2f})"
              f"{': ' + report['evidence'][spk] if report['evidence'].get(spk) else ''}")


if __name__ == "__main__":
    main()
