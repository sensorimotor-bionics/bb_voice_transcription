import os
from pathlib import Path
import json
import torch
from faster_whisper import WhisperModel
from pydub import AudioSegment
from nemo.collections.asr.models import SortformerEncLabelModel
from audio_utils import trim_wav, prepare_audio
import numpy as np

def process_prediction(prediction):
    speaker_turns = []
    for line in prediction:
        start, end, speaker = line.split()
        speaker_turns.append({"start": float(start), "end": float(end), "speaker": speaker})

    return speaker_turns

def transcribe_and_diarize_audio(audio_path: os.PathLike,
                                 whisper_size: str = "small",
                                 transcription_path: str | None = None,
                                 max_audio_length: int = 600,
                                 segment_gap: int | float = 5,
                                 verbose: bool = False):
    
    # Check that the audio path exists
    assert os.path.isfile(audio_path), f"{audio_path} is not a file"
    
    # Ensure whisper model is valid
    valid_whisper_models = ["tiny", "base", "small", "medium", "large-v3"]
    assert whisper_size in valid_whisper_models, f"'{whisper_size}' was not found in {valid_whisper_models}"
    
    # Parse the transcription path
    if transcription_path is None:
        transcription_path = Path(audio_path).stem + "_transcript.txt"
    else:
        if not isinstance(transcription_path, (str, bytes, os.PathLike)):
            raise TypeError("transcription_path must be a PathLike")
        # Check if the path exists
        if os.path.exists(transcription_path):
            user_input = input(f"{transcription_path}. Do you want to overwrite? (y/n): ").strip().lower()
            if user_input not in ('y', 'yes'):
                return
    print(f"Processing {audio_path}")

    ### Pre-format audio file for quicker processing
    audio_duration = prepare_audio(audio_path)
    if verbose:
        print(f"\tDuration = {audio_duration} seconds")

    ### FasterWhisper transcription
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "auto" if device == "cuda" else "int8"
    print("\tInitalizing transcription")
    model = WhisperModel(whisper_size, device=device, compute_type=compute_type)
    segments, _ = model.transcribe(audio_path, beam_size=5, word_timestamps=True)
    segments = list(segments)
    
    # Identify gaps in segments
    segment_times = np.zeros((len(segments), 2))
    for i, segment in enumerate(segments):
        segment_times[i,0] = segment.start
        segment_times[i,1] = segment.end
        if verbose:
            print("\t\t[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))

    ### NeMo diarization & merging
    print("\tInitalizing diarization")
    diar_model = SortformerEncLabelModel.from_pretrained("nvidia/diar_sortformer_4spk-v1")
    diar_model.eval()

    # If audio duration exceeds max then chunk
    if audio_duration > max_audio_length:
        pass
    
    else: # Otherwise process the whole file
        # Create the manifest config for the model
        manifest_path = audio_path.with_suffix(".json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"audio_filepath": str(audio_path),
                                "offset": 0,
                                "duration": audio_duration,
                                "label": "infer",
                                "text": "-"}) + "\n")
        
        # Run the diarization
        prediction = diar_model.diarize(audio=[str(manifest_path)], batch_size=1)
        speaker_turns = process_prediction(prediction[0])

    # Sort all turn by speaker
    speakers = sorted({t["speaker"] for t in speaker_turns})
    if verbose:
        print(f"{len(speaker_turns)} speaker turns across {len(speakers)} speakers: {speakers}")


    # Check audio file length

    # Merge speaker labels with transcript