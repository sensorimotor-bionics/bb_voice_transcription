import os
import json
from pathlib import Path
from typing import Union


def export_transcript_by_speaker(transcript: list, out_path: os.PathLike):
    """
    Format the transcript segments by speaker name.
    """
    out_path = Path(out_path)
    with open(out_path, "w", encoding="utf-8") as f:
        last = None
        for row in transcript:
            if row["speaker"] != last:
                f.write(f"\n[{row['speaker']}]\n")
                last = row["speaker"]
            f.write(f"({row['start']:.1f}-{row['end']:.1f}) {row['text']}\n")


def export_concatenated_transcript(transcript: list[dict], out_path: os.PathLike):
    """
    Format concatenated transcript segments grouped by video/file origin and speaker name.
    """
    out_path = Path(out_path)
    with open(out_path, "w", encoding="utf-8") as f:
        last_file = None
        last_speaker = None
        for row in transcript:
            file_id = row.get("video") or row.get("file") or row.get("file_name") or ""
            if file_id and file_id != last_file:
                f.write(f"\n=== Video: {file_id} ===\n")
                last_file = file_id
                last_speaker = None
            if row.get("speaker") != last_speaker:
                f.write(f"\n[{row.get('speaker')}]\n")
                last_speaker = row.get("speaker")
            f.write(f"({row['start']:.1f}-{row['end']:.1f}) {row['text']}\n")


def concatenate_batch_transcripts(
    batch_input: Union[dict, os.PathLike, list],
    output_path: Union[os.PathLike, None] = None
) -> list[dict]:
    """
    Concatenates individual transcripts from batch processing into a single sequence
    while keeping track of which video each transcript segment came from.

    Args:
        batch_input: Batch summary dictionary (from transcribe_and_diarize_folder),
                    directory path containing transcript JSON files, path to a summary JSON file,
                    or a list of file records / segment dicts.
        output_path: Optional file path or directory path to save concatenated transcript JSON/TXT.

    Returns:
        list[dict]: List of segment dicts with added 'video' and 'file' origin fields.
    """
    file_records = []
    records_dir = None  # where per-file transcripts referenced by a summary live

    if isinstance(batch_input, (str, Path)):
        p = Path(batch_input)
        if p.is_dir():
            records_dir = p
            summary_file = p / "batch_summary.json"
            if not summary_file.exists():
                summary_file = p / "summary.json"

            if summary_file.exists():
                with open(summary_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict) and "files" in data:
                        file_records = data["files"]
            if not file_records:
                json_files = sorted([
                    f for f in p.glob("*_transcript.json")
                    if f.name not in ("batch_summary.json", "summary.json", "concatenated_transcript.json", "combined_transcript.json")
                ])
                if not json_files:
                    json_files = sorted([
                        f for f in p.glob("*.json")
                        if f.name not in ("batch_summary.json", "summary.json", "concatenated_transcript.json", "combined_transcript.json")
                    ])

                for jf in json_files:
                    stem = jf.stem.removesuffix("_transcript").removesuffix(".16k")
                    with open(jf, "r", encoding="utf-8") as f:
                        seg_data = json.load(f)
                        if isinstance(seg_data, list):
                            file_records.append({
                                "file_name": jf.name,
                                "stem": stem,
                                "has_speech": True,
                                "transcript": seg_data
                            })
        elif p.is_file():
            records_dir = p.parent
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict) and "files" in data:
                    file_records = data["files"]
                elif isinstance(data, list):
                    if data and isinstance(data[0], dict) and "transcript" in data[0]:
                        file_records = data
                    else:
                        concatenated = []
                        for seg in data:
                            row = dict(seg)
                            v_id = row.get("video") or row.get("file") or row.get("file_name") or p.stem
                            row["video"] = v_id
                            row["file"] = v_id
                            concatenated.append(row)
                        dump_transcript(concatenated, output_path)
                        return concatenated

    elif isinstance(batch_input, dict):
        if "files" in batch_input:
            file_records = batch_input["files"]
        if "output_dir" in batch_input and batch_input["output_dir"]:
            records_dir = Path(batch_input["output_dir"])

    elif isinstance(batch_input, list):
        if batch_input and isinstance(batch_input[0], dict) and "transcript" in batch_input[0]:
            file_records = batch_input
        else:
            concatenated = []
            for seg in batch_input:
                row = dict(seg)
                v_id = row.get("video") or row.get("file") or row.get("file_name") or "unknown"
                row["video"] = v_id
                row["file"] = v_id
                concatenated.append(row)
            dump_transcript(concatenated, output_path)
            return concatenated

    concatenated = []
    for rec in file_records:
        if not rec.get("has_speech", True):
            continue
        v_id = rec.get("file_name") or rec.get("stem") or "unknown"
        segments = rec.get("transcript")
        # A summary written to disk points at each transcript instead of inlining it
        if not segments and rec.get("transcript_file"):
            segments = _load_referenced_transcript(rec["transcript_file"], records_dir)
        for seg in (segments or []):
            row = dict(seg)
            row["video"] = v_id
            row["file"] = v_id
            concatenated.append(row)

    dump_transcript(concatenated, output_path)
    return concatenated


def _load_referenced_transcript(transcript_file: os.PathLike,
                                records_dir: Union[Path, None]) -> list[dict]:
    """Load a transcript that a batch summary references rather than inlines."""
    path = Path(transcript_file)
    if not path.is_absolute() and records_dir is not None:
        path = Path(records_dir) / path
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"Could not read referenced transcript {path}: {e}")
        return []
    return data if isinstance(data, list) else []


def dump_transcript(concatenated: list[dict], 
                         output_path: Union[os.PathLike, None]):
    if output_path is None:
        return
    out_p = Path(output_path)
    if out_p.is_dir():
        out_p = out_p / "concatenated_transcript"

    json_path = out_p.with_suffix(".json")
    txt_path = out_p.with_suffix(".txt")

    os.makedirs(json_path.parent, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(concatenated, f, indent=4)
    export_concatenated_transcript(concatenated, txt_path)
    print(f"Exported concatenated transcript to:\n  JSON: {json_path}\n  TXT:  {txt_path}")


# Alias for convenience
concatenate_transcripts = concatenate_batch_transcripts


def _speaker_sort_key(speaker: str):
    """
    Order speaker tags by their trailing number when they have one, so speaker_2 sorts
    before speaker_10 and both sort before any non-numbered label.
    """
    tail = str(speaker).rsplit('_', 1)[-1]
    try:
        return (0, int(tail), "")
    except ValueError:
        return (1, 0, str(speaker))


def relabel_transcript(transcript_path: os.PathLike, speaker_ids: list[str]):
    """
    Relabels generic speaker_0, speaker_N to desired labels.
    Assumes sequential speaker numbering and requires equal numbers of original and new ids.
    """
    transcript_path = Path(transcript_path)
    with open(transcript_path, 'r', encoding='utf8') as f:
        transcript = json.load(f)

    if isinstance(transcript, dict) and "files" in transcript:
        return relabel_batched_transcript(transcript, speaker_ids, output_path=transcript_path)
    elif isinstance(transcript, list) and transcript and ("video" in transcript[0] or "file" in transcript[0]):
        return relabel_batched_transcript(transcript, speaker_ids, output_path=transcript_path)

    # Map the speakers actually present onto the provided names by position. Speaker
    # numbers are not necessarily 0..N-1 (clustering can return speaker_0 and speaker_2)
    # and are not necessarily numeric at all, so indexing by the number would either
    # run off the end of speaker_ids or fail to parse.
    present = sorted({t['speaker'] for t in transcript}, key=_speaker_sort_key)
    assert len(present) == len(speaker_ids), \
        f"{len(present)} speakers in transcript ({', '.join(present)}) but {len(speaker_ids)} provided"

    mapping = dict(zip(present, speaker_ids))
    for t in transcript:
        t['speaker'] = mapping[t['speaker']]

    # Dump relabeled transcript
    export_transcript_by_speaker(transcript, transcript_path.with_suffix(".relabeled.txt"))
    with open(transcript_path.with_suffix(".relabeled.json"), "w", encoding="utf-8") as f:
        json.dump(transcript, f, indent=4)
    return transcript


def relabel_batched_transcript(
    transcript_input: Union[dict, os.PathLike, list],
    speaker_ids: Union[list[str], dict[Union[str, int], str]],
    output_path: Union[os.PathLike, None] = None
) -> list[dict]:
    """
    Relabels generic speaker names (speaker_0, speaker_1, etc.) across a batched transcript.

    Args:
        transcript_input: Batched transcript (list of segment dicts with 'video' field),
                         summary dictionary from transcribe_and_diarize_folder,
                         path to combined/concatenated JSON, or folder path containing transcripts.
        speaker_ids: List of replacement speaker names (indexed by speaker number)
                    or a mapping dict (e.g. {"speaker_0": "Interviewer", "speaker_1": "Participant"}).
        output_path: Optional file path or directory path for saving relabeled outputs.

    Returns:
        list[dict]: Relabeled concatenated transcript segments.
    """
    target_path = None
    if isinstance(transcript_input, (str, Path)):
        p = Path(transcript_input)
        if output_path is None:
            target_path = p
        if p.is_file() and p.suffix == ".json":
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list) and data and ("video" in data[0] or "file" in data[0]):
                concatenated = [dict(s) for s in data]
            else:
                concatenated = concatenate_batch_transcripts(p)
        else:
            concatenated = concatenate_batch_transcripts(p)
    elif isinstance(transcript_input, dict):
        concatenated = concatenate_batch_transcripts(transcript_input)
        if output_path is None and "output_dir" in transcript_input and transcript_input["output_dir"]:
            target_path = Path(transcript_input["output_dir"]) / "concatenated_transcript"
    elif isinstance(transcript_input, list):
        if transcript_input and isinstance(transcript_input[0], dict) and "transcript" in transcript_input[0]:
            concatenated = concatenate_batch_transcripts(transcript_input)
        else:
            concatenated = [dict(s) for s in transcript_input]
    else:
        raise ValueError(f"Unsupported transcript_input type: {type(transcript_input)}")

    if output_path is not None:
        target_path = Path(output_path)

    # Build speaker mapping
    mapping = {}
    if isinstance(speaker_ids, dict):
        for k, v in speaker_ids.items():
            mapping[str(k)] = v
            if isinstance(k, int) or (isinstance(k, str) and k.isdigit()):
                mapping[f"speaker_{k}"] = v
    elif isinstance(speaker_ids, (list, tuple)):
        # Assign names by position over the speakers actually present. Indexing by the
        # trailing number instead would hand the same name to two speakers whenever the
        # numbering has a gap (speaker_0 takes ids[0], then unmapped speaker_2 also
        # takes ids[0]), silently merging two people into one.
        unique_speakers = sorted({t["speaker"] for t in concatenated if "speaker" in t},
                                 key=_speaker_sort_key)
        if len(unique_speakers) > len(speaker_ids):
            print(f"Warning: {len(unique_speakers)} speakers present "
                  f"({', '.join(map(str, unique_speakers))}) but only "
                  f"{len(speaker_ids)} name(s) given; the rest keep their labels.")
        for spk, name in zip(unique_speakers, speaker_ids):
            mapping[spk] = name

    # Perform relabeling
    for row in concatenated:
        spk = row.get("speaker")
        if spk in mapping:
            row["speaker"] = mapping[spk]
        elif isinstance(spk, str) and spk.startswith("speaker_"):
            num_part = spk.split("_")[-1]
            if num_part in mapping:
                row["speaker"] = mapping[num_part]

    # Export if target_path is determined
    if target_path:
        out_p = Path(target_path)
        if out_p.is_dir():
            out_p = out_p / "concatenated_transcript"

        stem = out_p.stem
        if stem.endswith(".relabeled"):
            base_stem = stem
        else:
            base_stem = f"{stem}.relabeled"

        json_out = out_p.parent / f"{base_stem}.json"
        txt_out = out_p.parent / f"{base_stem}.txt"

        os.makedirs(json_out.parent, exist_ok=True)
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump(concatenated, f, indent=4)
        export_concatenated_transcript(concatenated, txt_out)
        print(f"Saved relabeled batched transcript to:\n  JSON: {json_out}\n  TXT:  {txt_out}")

    return concatenated


# Convenient aliases
relabel_speaker_batch = relabel_batched_transcript
relabel_batch_transcript = relabel_batched_transcript
relabel_speaker = relabel_transcript