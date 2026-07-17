import os
from pathlib import Path
import json
import torch
from faster_whisper import WhisperModel
from pydub import AudioSegment
from nemo.collections.asr.models import SortformerEncLabelModel

whisper_size = "small" # tiny / base / small / medium / large-v3 (small fits on my Nvidia 1080 GPU)
# Check if CUDA is installed properly and update settings. Cannot do diarization without CUDA.
device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "auto" if device == "cuda" else "int8" # Some systems to float16 or float32, auto lets it do either
print(f"Using device: {device} ({compute_type})")

# test audio file
audio_path = Path(r"C:\Users\Somlab\Downloads\audio1365983309_trunc.16k.wav")

# Segmentation
model = WhisperModel(whisper_size, device=device, compute_type=compute_type)
segments, info = model.transcribe(str(audio_path), beam_size=5, word_timestamps=True)

# transcribe() returns a generator; materialize it so we can reuse the segments.
segments = list(segments)
print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))

# Dump a config
manifest_path = audio_path.with_suffix(".json")
with open(manifest_path, "w", encoding="utf-8") as f:
    f.write(json.dumps({"audio_filepath": str(audio_path), "offset": 0, "duration": 100000, "label": "infer", "text": "-"}) + "\n")

# Diarize
print("Running NeMo Sortformer diarization...")
diar_model = SortformerEncLabelModel.from_pretrained("nvidia/diar_sortformer_4spk-v1")
diar_model.eval()

# diarize() returns one list of "start end speaker_label" strings per input file.
prediction = diar_model.diarize(audio=[str(manifest_path)], batch_size=1)

speaker_turns = []
for line in prediction[0]:
    start, end, speaker = line.split()
    speaker_turns.append({"start": float(start), "end": float(end), "speaker": speaker})

speakers = sorted({t["speaker"] for t in speaker_turns})
print(f"{len(speaker_turns)} speaker turns across {len(speakers)} speakers: {speakers}")


# --- 3. Assign a speaker to each transcript segment ---
def _speaker_for(start, end):
    """Speaker whose turns overlap the [start, end] interval the most."""
    best, best_overlap = "unknown", 0.0
    for t in speaker_turns:
        overlap = max(0.0, min(end, t["end"]) - max(start, t["start"]))
        if overlap > best_overlap:
            best_overlap, best = overlap, t["speaker"]
    return best


def assign_speaker(seg):
    """Majority speaker over a segment's words (falls back to segment span)."""
    votes = {}
    for w in (seg.words or []):
        spk = _speaker_for(w.start, w.end)
        votes[spk] = votes.get(spk, 0.0) + (w.end - w.start)
    return max(votes, key=votes.get) if votes else _speaker_for(seg.start, seg.end)


transcript = [
    {"start": s.start, "end": s.end, "speaker": assign_speaker(s), "text": s.text.strip()}
    for s in segments
]

# Print, grouping consecutive segments from the same speaker.
last = None
for row in transcript:
    if row["speaker"] != last:
        print(f"\n[{row['speaker']}]")
        last = row["speaker"]
    print(f"  ({row['start']:.1f}-{row['end']:.1f}) {row['text']}")

out_path = audio_path.with_suffix(".transcript.txt")
with open(out_path, "w", encoding="utf-8") as f:
    last = None
    for row in transcript:
        if row["speaker"] != last:
            f.write(f"\n[{row['speaker']}]\n")
            last = row["speaker"]
        f.write(f"({row['start']:.1f}-{row['end']:.1f}) {row['text']}\n")
print("Saved transcript to", out_path)