from pathlib import Path
import json
from nemo.collections.asr.models import SortformerEncLabelModel


audio_path = Path(r"C:\Users\Somlab\Downloads\audio1365983309_trunc.16k.wav")

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
print(speaker_turns)