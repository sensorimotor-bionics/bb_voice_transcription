import os
from pathlib import Path
import json
import torch
from faster_whisper import WhisperModel
from pydub import AudioSegment
from nemo.collections.asr.models import SortformerEncLabelModel, EncDecSpeakerLabelModel
from audio_utils import trim_wav, prepare_audio
import numpy as np

def format_diarization(prediction):
    speaker_turns = []
    for line in prediction:
        start, end, speaker = line.split()
        speaker_turns.append({"start": float(start), "end": float(end), "speaker": speaker})

    return speaker_turns


def diarize(diarizatrion_model: SortformerEncLabelModel, 
            audio_path: os.PathLike,
            audio_duration: int):

    # Create the manifest config for the model
    manifest_path = Path(audio_path).stem + ".json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"audio_filepath": str(audio_path),
                            "offset": 0,
                            "duration": audio_duration,
                            "label": "infer",
                            "text": "-"}) + "\n")
    
    # Run the diarization
    prediction = diarizatrion_model.diarize(audio=[str(manifest_path)], batch_size=1)

    # Delete the manifest file
    os.remove(manifest_path)

    # Return the processed prediction
    return prediction


def assign_speaker(segment, speaker_turns):
    votes = {}
    # For each word in the segment
    for w in (segment.words or []):
        # Check 
        best, best_overlap = "unknown", 0.0
        for t in speaker_turns:
            overlap = max(0.0, min(w.end, t["end"]) - max(w.start, t["start"]))
            if overlap > best_overlap:
                best_overlap, best = overlap, t["speaker"]

        votes[best] = votes.get(best, 0.0) + (w.end - w.start)
    return max(votes, key=votes.get) if votes else best


def create_transcript(segments, speaker_times):
    transcript = []
    for s in segments:
        transcript.append({"start": s.start,
                           "end": s.end,
                           "speaker": assign_speaker(s, speaker_times),
                           "text": s.text.strip()})
    
    return transcript

def transcribe_and_diarize_audio(audio_path: os.PathLike,
                                 whisper_size: str = "small",
                                 transcription_path: str | None = None,
                                 max_audio_length: int = 600,
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
    print("\tInitalizing transcription")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "auto" if device == "cuda" else "int8"
    model = WhisperModel(whisper_size, device=device, compute_type=compute_type)
    segments, _ = model.transcribe(audio_path, beam_size=5, word_timestamps=True)
    segments = list(segments) # Full list of word segments for whole file

    ### NeMo diarization
    print("\tInitalizing diarization")
    diar_model = SortformerEncLabelModel.from_pretrained("nvidia/diar_sortformer_4spk-v1").eval()
    
    # If audio duration exceeds max then chunk
    if audio_duration > max_audio_length:
        # Initalize an encoding model to identify the speaker across segments
        enc_model = EncDecSpeakerLabelModel.from_pretrained(model_name="titanet_small").eval()

        # Identify gaps in segments
        segment_times = np.zeros((len(segments), 3))
        for i, segment in enumerate(segments):
            segment_times[i,0] = segment.start
            segment_times[i,1] = segment.end

        segment_times[1:,2] = segment_times[1:,1] - segment_times[:-1,0]

        # Loop through segments to find gaps and diarize in chunks

    
    else: # Otherwise process the whole file
        diar_prediction = diarize(diar_model, audio_path, audio_duration)
        speaker_times = format_diarization(diar_prediction[0])
        transcript = create_transcript(segments, speaker_times)
    
    if verbose: # Print the transcript
        last = None
        for row in transcript:
            if row["speaker"] != last:
                print(f"\n[{row['speaker']}]") # Identify the speaker on speaker change
                last = row["speaker"]
            print(f"  ({row['start']:.1f}-{row['end']:.1f}) {row['text']}")

    ### Check speaker embeddings for 