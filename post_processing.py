import os
import json
from pathlib import Path

def export_transcript_by_speaker(transcript:list, out_path:os.PathLike):
    """
    Format the transcript segments by speaker name.
    """
    with open(out_path, "w", encoding="utf-8") as f:
        last = None
        for row in transcript:
            if row["speaker"] != last:
                f.write(f"\n[{row['speaker']}]\n")
                last = row["speaker"]
            f.write(f"({row['start']:.1f}-{row['end']:.1f}) {row['text']}\n")


def relabel_transcript(transcript_path:os.PathLike, speaker_ids:list[str]):
    """
    Relabels generic speaker_0, speaker_N to desired labels.
    Assumes sequential speaker numbering and requires equal numbers of original and new ids.
    """
    transcript_path = Path(transcript_path)
    with open(transcript_path, 'r', encoding='utf8') as f: 
        transcript = json.load(f)

    # Get list of unique speakers and confirm lengths match
    speaker_id = [int(t['speaker'].split('_')[-1]) for t in transcript]
    speaker_list = list(set(speaker_id))
    assert len(speaker_list) == len(speaker_ids), f"{len(speaker_list)} speakers in transcript but only {len(speaker_ids)} provided"

    # Iterate through the transcript and relabel according to speaker id
    for (idx, t) in zip(speaker_id, transcript):
        t['speaker'] = speaker_ids[idx]

    # Dump relabeled transcript
    export_transcript_by_speaker(transcript, transcript_path.with_suffix(".relabeled.txt"))