import os
from pathlib import Path
import json
import torch
from faster_whisper import WhisperModel
from pydub import AudioSegment
from nemo.collections.asr.models import SortformerEncLabelModel
from audio_utils import trim_wav, prepare_audio

def transcribe_and_diarize_audio(audio_path: os.PathLike,
                                 whisper_size: str = "small",
                                 transcription_path: str | None = None,
                                 max_audio_length: int = 90,
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
    segments, info = model.transcribe(audio_path, beam_size=5, word_timestamps=True)
    segments = list(segments)
    
    for segment in segments:
        if verbose:
            print("\t\t[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))

    # Identify gaps in segments

    ### NeMo diarization & merging
    # Format input file for NeMo (mono 16k)
    # Check audio file length

    # Merge speaker labels with transcript