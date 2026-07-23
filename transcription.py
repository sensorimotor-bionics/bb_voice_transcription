import os
import json
import torch
import librosa
import numpy as np
from faster_whisper import WhisperModel
from faster_whisper.transcribe import Segment
from pathlib import Path
from nemo.collections.asr.models import SortformerEncLabelModel, EncDecSpeakerLabelModel
from audio_utils import prepare_audio
from sklearn.cluster import AgglomerativeClustering
from post_processing import export_transcript_by_speaker

## Disable logging stuff
from nemo.utils import logging as nemo_logging
os.environ["NEMO_LOG_LEVEL"] = "40"
nemo_logging.set_verbosity(nemo_logging.ERROR)


def transcribe(audio_path: os.PathLike,
               whisper_size: str = "small",
               model: WhisperModel | None = None) -> list[Segment]:
    """
    Wrapper for using WhisperModel to transcribe an audio file.

    Args:
        audio_path (os.PathLike): Path to file.
        whisper_size (str): WhisperModel size ["tiny", "small", "large-v3", etc]
        model (WhisperModel | None): Pre-loaded WhisperModel instance to reuse across calls.

    Returns:
        list[Segment]: List of word segments
    """
    if model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "auto" if device == "cuda" else "int8"
        model = WhisperModel(whisper_size, device=device, compute_type=compute_type)
    
    segments, _ = model.transcribe(str(audio_path), beam_size=5, word_timestamps=True)
    
    return list(segments) # Full list of word segments for whole file


def detect_speech(segments: list[Segment], min_speech_duration: float = 0.5) -> bool:
    """
    Determines if there is speech present in the transcribed segments.

    Args:
        segments (list[Segment]): List of Whisper segments.
        min_speech_duration (float): Minimum total duration of speech segments in seconds.

    Returns:
        bool: True if speech is detected, False otherwise.
    """
    if not segments:
        return False
    speech_duration = sum(s.end - s.start for s in segments if s.text and s.text.strip())
    return speech_duration >= min_speech_duration


def format_diarization(prediction: list[str],
                       speaker_offset: int = 0) -> list[dict]:
    """
    Convert diarization output into easier to process dictionary
    """
    speaker_turns = []
    for line in prediction:
        start, end, speaker = line.split()
        if speaker_offset > 0:
            spk_int = int(speaker.split('_')[-1])
            speaker = f"speaker_{spk_int+speaker_offset}"
        speaker_turns.append({"start": float(start), "end": float(end), "speaker": speaker})

    return speaker_turns


def _diarize(diarizatrion_model: SortformerEncLabelModel, 
            audio_path: os.PathLike,
            offset: int | float,
            audio_duration: int | float) -> list[list[str]]:
    """
    Wrapper function for diarize an audio file. Handles dynamic manifests.

    Args:
        diarizatrion_model (SortformerEncLabelModel): Which model to use for diarization.
        audio_path (os.PathLike): Path to audio file.
        offset (int | float): Where in audio file to start diarization.
        audio_duration (int | float): What duration of the audio file to process.

    Returns:
        list[list[str]]: Speaker timings
    """

    # Create the manifest config for the model - had permission issues with the default config
    manifest_path = Path(audio_path).stem + ".json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"audio_filepath": str(audio_path),
                            "offset": offset,
                            "duration": audio_duration,
                            "label": "infer",
                            "text": "-"}) + "\n")
    
    # Run the diarization
    prediction = diarizatrion_model.diarize(audio=[str(manifest_path)],
                                            batch_size=1,
                                            verbose=False)

    # Delete the manifest file
    if os.path.exists(manifest_path):
        os.remove(manifest_path)

    # Return the processed prediction
    return prediction # type: ignore


def diarize_audio(diar_model: SortformerEncLabelModel,
                  audio_path: os.PathLike,
                  segments: list[Segment],
                  audio_duration: int | float,
                  max_audio_length: int | float = 600, 
                  verbose: bool = False):

    # Determine if we can fit the audio in the available memory
    chunked_diarization = audio_duration > max_audio_length

    # If audio duration exceeds max then chunk
    if chunked_diarization:
        num_chunks = np.ceil(audio_duration / max_audio_length)
        optimal_chunk_length = audio_duration / num_chunks
        # Identify gaps in segments
        segment_times = np.zeros((len(segments), 3))
        for i, segment in enumerate(segments):
            segment_times[i,0] = segment.start
            segment_times[i,1] = segment.end

        segment_times[1:,2] = segment_times[1:, 0] - segment_times[:-1, 1]

        # Loop through segments to find gaps and diarize in chunks
        start_time, stop_time = -1, optimal_chunk_length
        speaker_times = []
        chunk_counter = 1
        while start_time < segment_times[-1,0]:
            if verbose:
                print(f"\t- Chunk {chunk_counter}: {start_time:0.1f}-{stop_time:0.1f}")

            # Check that we don't leave a section at the end hanging
            if (audio_duration - start_time) < max_audio_length:
                sub_seg_idx = np.where((segment_times[:,0] > start_time))[0]
                start_idx = sub_seg_idx[0]
                stop_idx = -1
            else:
                # Find the indices of the segments in the range of the start and stop time
                sub_seg_idx = np.where(((segment_times[:,0] > start_time) & 
                                        (segment_times[:,1] < stop_time) & 
                                        (segment_times[:,2] > 1)))[0]

                start_idx = sub_seg_idx[0]
                stop_idx = sub_seg_idx[-1]
            
            # Diarize audio chunk
            diar_prediction = _diarize(diar_model,
                                      audio_path,
                                      segment_times[start_idx,0] - 0.1,
                                      segment_times[stop_idx,1] - segment_times[start_idx,0] + 0.2)
            speaker_times += format_diarization(diar_prediction[0], chunk_counter*10) # Append speaker times for the chunk

            # Update 
            start_time = segment_times[stop_idx,1]
            stop_time = start_time + optimal_chunk_length
            chunk_counter += 1        
    
    else: # Otherwise process the whole file
        diar_prediction = _diarize(diar_model, audio_path, 0, audio_duration)
        speaker_times = format_diarization(diar_prediction[0])

    return speaker_times, chunked_diarization


def assign_speaker(segment: Segment,
                   speaker_turns: list[dict]) -> str:
    """
    Assign speaker to word segment based on maximum overlap from diarization output.
    """

    best = "unknown"
    votes = {}
    # For each word in the segment
    for w in (segment.words or []): # type: ignore
        # Check 
        best, best_overlap = "unknown", 0.0
        for t in speaker_turns:
            overlap = max(0.0, min(w.end, t["end"]) - max(w.start, t["start"]))
            if overlap > best_overlap:
                best_overlap, best = overlap, t["speaker"]

        votes[best] = votes.get(best, 0.0) + (w.end - w.start)

    return max(votes, key=votes.get) if votes else best # type: ignore


def create_transcript(segments: list[Segment],
                      speaker_times: list[dict]) -> list[dict]:
    """
    Create a transcript by assigning speaker times to word segments.
    """
    transcript = []
    for s in segments:
        transcript.append({"start": s.start,
                           "end": s.end,
                           "speaker": assign_speaker(s, speaker_times),
                           "text": s.text.strip()})
    
    return transcript


def get_transcript_speakers(transcript: list[dict]):
    # Get segments split by predicted speaker
    if not transcript:
        return [], [], []
    speaker = transcript[0]['speaker']
    num_segments = len(transcript)
    speaker_start_times, speaker_stop_times, speakers = [0], [], [speaker]
    for i, t in enumerate(transcript):
        if t['speaker'] != speaker:
            speaker_start_times.append(t['start'])
            if i == 0 or i == num_segments:
                speaker_stop_times.append(t['end'])
            else:
                speaker_stop_times.append(transcript[i-1]['end'])
            speaker = t['speaker']
            speakers.append(speaker)

    # Add the final timestamp
    speaker_stop_times.append(transcript[-1]['end'])

    return speakers, speaker_start_times, speaker_stop_times


def extract_unique_speaker_embeddings(transcript: list[dict],
                                      unique_speakers: list[str],
                                      audio: np.ndarray,
                                      encoding_model: EncDecSpeakerLabelModel,
                                      device: str,
                                      sample_frequency: int = 16000) -> np.ndarray:
    """
    Extracts, averages, and normalizes speaker embeddings for each unique speaker tag in the transcript.
    Useful as a standalone function for debugging speaker representations.
    """
    speaker_embeddings = []
    for spk in unique_speakers:
        spk_segments = [t for t in transcript if t['speaker'] == spk]
        
        seg_embeddings = []
        for seg in spk_segments:
            start_idx = int(seg['start'] * sample_frequency)
            stop_idx = int(seg['end'] * sample_frequency)
            if stop_idx - start_idx < 160:  # Skip empty or sub-10ms audio slices
                continue
            audio_tensor = torch.tensor(audio[start_idx:stop_idx], dtype=torch.float32).unsqueeze(0).to(device)
            audio_len = torch.tensor([audio_tensor.shape[1]], dtype=torch.int32).to(device)
            
            with torch.no_grad():
                _, emb = encoding_model.forward(input_signal=audio_tensor, input_signal_length=audio_len)
            emb = emb.squeeze(0).cpu().numpy()
            seg_embeddings.append(emb / np.linalg.norm(emb))

        if seg_embeddings:
            # Average embeddings across all segments for this chunk speaker
            mean_emb = np.mean(seg_embeddings, axis=0)
            speaker_embeddings.append(mean_emb / np.linalg.norm(mean_emb))
        else:
            speaker_embeddings.append(np.zeros(encoding_model.d_model))

    if not speaker_embeddings:
        return np.empty((0, encoding_model.d_model))

    # Denoise and normalize the speaker embedding matrix
    embedding_matrix = np.vstack(speaker_embeddings)
    if len(speaker_embeddings) > 1:
        embedding_matrix = embedding_matrix - np.mean(embedding_matrix, axis=0, keepdims=True)

    norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12 
    embedding_matrix = embedding_matrix / norms

    return embedding_matrix


def cluster_embeddings(distance_mat: np.ndarray,
                       num_speakers: int | None = None,
                       distance_threshold: float = 0.75):
    if len(distance_mat) <= 1:
        return np.zeros(len(distance_mat), dtype=int)
        
    if num_speakers is None:
        feature_clusterer = AgglomerativeClustering(n_clusters=None,
                                                    metric='precomputed',
                                                    linkage='average',
                                                    distance_threshold=distance_threshold)
    elif isinstance(num_speakers, int):
        feature_clusterer = AgglomerativeClustering(n_clusters=min(num_speakers, len(distance_mat)),
                                                    metric='precomputed',
                                                    linkage='average')

    return feature_clusterer.fit_predict(distance_mat)
    

def post_hoc_diarization(transcript: list[dict], 
                         audio_path: os.PathLike,
                         num_speakers: int | None = None,
                         enc_model: EncDecSpeakerLabelModel | None = None):
    
    # Identify all unique chunk-level speakers in the transcript
    unique_speakers = list(dict.fromkeys(t['speaker'] for t in transcript))
    if not unique_speakers:
        return transcript

    # Extract averaged speaker embeddings for each unique speaker
    if enc_model is None:
        enc_model = EncDecSpeakerLabelModel.from_pretrained(model_name="titanet_small").eval() # type: ignore
    
    device = next(enc_model.parameters()).device
    sample_frequency = 16000
    audio, _ = librosa.load(audio_path, sr=sample_frequency)

    embedding_matrix = extract_unique_speaker_embeddings(
        transcript=transcript,
        unique_speakers=unique_speakers,
        audio=audio,
        encoding_model=enc_model,
        device=device,
        sample_frequency=sample_frequency
    )

    if len(embedding_matrix) == 0:
        return transcript

    # Cluster chunk speakers into global speaker IDs
    similarity_mat = np.dot(embedding_matrix, embedding_matrix.T)
    distance_mat = np.clip(1.0 - similarity_mat, 0.0, 2.0)
    speaker_labels = cluster_embeddings(distance_mat, num_speakers)

    # Create mapping dict (e.g. {'speaker_10': 'speaker_0', 'speaker_20': 'speaker_0'})
    spk_to_global = {spk: f"speaker_{label}" for spk, label in zip(unique_speakers, speaker_labels)}

    # Remap transcript
    for t in transcript:
        t['speaker'] = spk_to_global[t['speaker']]

    return transcript


def transcribe_and_diarize_audio(audio_path: os.PathLike,
                                 whisper_size: str = "small",
                                 transcription_path: str | None = None,
                                 max_audio_length: int = 600,
                                 verbose: bool = False,
                                 num_speakers: int | None = 2,
                                 cleanup: bool = True,
                                 whisper_model: WhisperModel | None = None,
                                 diar_model: SortformerEncLabelModel | None = None,
                                 enc_model: EncDecSpeakerLabelModel | None = None):
    """
    High level function to process, transcribe, and diarize a single audio file.
    """
    # Check that the audio path exists
    assert os.path.isfile(audio_path), f"{audio_path} is not a file"
    audio_path = Path(audio_path)
    audio_path_bk = audio_path

    # Ensure whisper model is valid
    valid_whisper_models = ["tiny", "base", "small", "medium", "large-v3"]
    assert whisper_size in valid_whisper_models, f"'{whisper_size}' was not found in {valid_whisper_models}"
    
    # Parse the transcription path
    if transcription_path is None:
        transcription_path = audio_path.with_name(audio_path.stem + "_transcript")
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
    audio_path = audio_path.with_suffix(".16k.wav")

    if verbose:
        print(f"\tDuration = {audio_duration} seconds")

    ### FasterWhisper transcription
    print("\tInitializing transcription")
    segments = transcribe(audio_path, whisper_size, model=whisper_model)
    
    if not detect_speech(segments):
        print("\tNo speech detected in audio file.")
        if cleanup and os.path.exists(audio_path) and audio_path.suffix == ".16k.wav" and audio_path_bk != audio_path:
            os.remove(audio_path)
        return []

    ### NeMo diarization
    print("\tInitializing diarization")
    if diar_model is None:
        diar_model = SortformerEncLabelModel.from_pretrained("nvidia/diar_sortformer_4spk-v1").eval() # type: ignore
    
    speaker_times, needs_post_hoc = diarize_audio(diar_model,
                                                  audio_path,
                                                  segments,
                                                  audio_duration,
                                                  max_audio_length,
                                                  verbose=verbose)

    # Create the transcript from segments and speaker times
    transcript = create_transcript(segments, speaker_times)

    # Harmonize diarization across chunks
    if needs_post_hoc:
        transcript = post_hoc_diarization(transcript, audio_path, num_speakers, enc_model=enc_model)

    # Dump the transcript as .txt
    export_transcript_by_speaker(transcript, transcription_path.with_suffix(".txt"))

    # Dump the transcript as .json for later parsing
    out_path = transcription_path.with_suffix(".json")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(transcript, indent=4) + "\n")

    print("\tSaved transcript to", out_path)

    if cleanup:
        if (os.path.exists(audio_path) and audio_path.suffix == ".16k.wav" and audio_path_bk != audio_path):
            os.remove(audio_path)

    return transcript


def transcribe_and_diarize_folder(
    folder_path: os.PathLike,
    output_dir: os.PathLike | None = None,
    whisper_size: str = "small",
    video_extensions: list[str] | None = None,
    max_audio_length: int = 600,
    min_speech_duration: float = 0.5,
    global_num_speakers: int | None = None,
    distance_threshold: float = 0.75,
    verbose: bool = False,
    cleanup: bool = True
) -> dict:
    """
    Processes a folder of video/audio files:
    1. Transcribes each file and determines if speech is present.
    2. For files with speech, diarizes speaker turns and extracts speaker embeddings.
    3. Performs global cross-file speaker classification across all files.
    4. Saves transcript text, JSON, and a batch summary report.

    Args:
        folder_path (os.PathLike): Folder containing video/audio files.
        output_dir (os.PathLike | None): Target output directory. Defaults to <folder_path>/transcripts.
        whisper_size (str): Model size for FasterWhisper ("tiny", "small", "large-v3", etc.).
        video_extensions (list[str] | None): Allowed file extensions.
        max_audio_length (int): Chunk threshold in seconds for diarization.
        min_speech_duration (float): Minimum duration of speech to qualify as having speech.
        global_num_speakers (int | None): Known total number of global speakers across all files (or None for threshold-based).
        distance_threshold (float): Clustering distance threshold when global_num_speakers is None.
        verbose (bool): Extra logging.
        cleanup (bool): Automatically remove intermediate 16k WAV files.

    Returns:
        dict: Batch summary dictionary containing statistics and file details.
    """
    folder_path = Path(folder_path)
    assert folder_path.is_dir(), f"{folder_path} is not a valid directory"

    if output_dir is None:
        output_dir = folder_path / "transcripts"
    else:
        output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    if video_extensions is None:
        video_extensions = [".mp4", ".m4v", ".avi", ".mov", ".mkv", ".wav", ".mp3", ".flac", ".m4a", ".aac"]
    video_extensions = [ext.lower() for ext in video_extensions]

    # Discover candidate files
    media_files = sorted([
        f for f in folder_path.iterdir()
        if f.is_file() and f.suffix.lower() in video_extensions
    ])

    if not media_files:
        print(f"No media files with extensions {video_extensions} found in {folder_path}")
        return {"total_files": 0, "files_with_speech": 0, "files_without_speech": 0, "files": []}

    print(f"Found {len(media_files)} media file(s) in {folder_path}")

    # Initialize models once
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "auto" if device == "cuda" else "int8"
    print(f"Loading models on {device} ({compute_type})...")

    whisper_model = WhisperModel(whisper_size, device=device, compute_type=compute_type)
    diar_model = SortformerEncLabelModel.from_pretrained("nvidia/diar_sortformer_4spk-v1").eval() # type: ignore
    enc_model = EncDecSpeakerLabelModel.from_pretrained(model_name="titanet_small").eval() # type: ignore

    file_records = []
    all_speaker_embeddings = []
    embedding_metadata = [] # stores {"speech_file_idx": int, "local_speaker": str}

    speech_file_count = 0

    for idx, file_path in enumerate(media_files, start=1):
        print(f"\n--- [{idx}/{len(media_files)}] Processing: {file_path.name} ---")
        
        # Prepare 16kHz WAV audio
        wav_path = file_path.with_suffix(".16k.wav")
        audio_duration = prepare_audio(file_path, wav_path)

        if audio_duration < 0:
            print(f"Error processing audio for {file_path.name}. Skipping.")
            file_records.append({
                "file_name": file_path.name,
                "has_speech": False,
                "duration": 0.0,
                "speaker_count": 0,
                "error": "Audio conversion failed"
            })
            continue

        # Transcribe
        segments = transcribe(wav_path, whisper_size=whisper_size, model=whisper_model)
        has_speech = detect_speech(segments, min_speech_duration=min_speech_duration)

        if not has_speech:
            print(f"\tNo speech detected in {file_path.name}.")
            file_records.append({
                "file_name": file_path.name,
                "stem": file_path.stem,
                "has_speech": False,
                "duration": audio_duration,
                "speaker_count": 0,
                "transcript": [],
                "speakers": []
            })
            if cleanup and os.path.exists(wav_path) and wav_path != file_path:
                os.remove(wav_path)
            continue

        print(f"\tSpeech detected! ({audio_duration:.1f}s) Running diarization...")
        speaker_times, _ = diarize_audio(diar_model, wav_path, segments, audio_duration, max_audio_length, verbose=verbose)
        transcript = create_transcript(segments, speaker_times)

        # Extract speaker embeddings for local unique speakers
        unique_speakers = list(dict.fromkeys(t['speaker'] for t in transcript))
        audio, _ = librosa.load(wav_path, sr=16000)

        emb_matrix = extract_unique_speaker_embeddings(
            transcript=transcript,
            unique_speakers=unique_speakers,
            audio=audio,
            encoding_model=enc_model,
            device=device,
            sample_frequency=16000
        )

        for spk_idx, local_spk in enumerate(unique_speakers):
            if spk_idx < len(emb_matrix):
                all_speaker_embeddings.append(emb_matrix[spk_idx])
                embedding_metadata.append({
                    "speech_file_idx": speech_file_count,
                    "local_speaker": local_spk
                })

        file_records.append({
            "file_name": file_path.name,
            "stem": file_path.stem,
            "has_speech": True,
            "duration": audio_duration,
            "speech_file_idx": speech_file_count,
            "local_speakers": unique_speakers,
            "transcript": transcript
        })
        speech_file_count += 1

        if cleanup and os.path.exists(wav_path) and wav_path != file_path:
            os.remove(wav_path)

    # Perform global speaker classification across all speech files
    total_global_speakers = 0
    if all_speaker_embeddings:
        print(f"\nRunning cross-file speaker clustering across {len(all_speaker_embeddings)} local speaker representation(s)...")
        global_matrix = np.vstack(all_speaker_embeddings)
        if len(all_speaker_embeddings) > 1:
            global_matrix = global_matrix - np.mean(global_matrix, axis=0, keepdims=True)
        
        norms = np.linalg.norm(global_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1e-12
        global_matrix = global_matrix / norms

        similarity_mat = np.dot(global_matrix, global_matrix.T)
        distance_mat = np.clip(1.0 - similarity_mat, 0.0, 2.0)

        global_labels = cluster_embeddings(distance_mat, num_speakers=global_num_speakers, distance_threshold=distance_threshold)
        unique_global_labels = sorted(list(set(global_labels)))
        total_global_speakers = len(unique_global_labels)

        # Build mapping: speech_file_idx -> {local_speaker -> global_speaker_label}
        file_spk_mappings = {}
        for meta, g_label in zip(embedding_metadata, global_labels):
            sf_idx = meta["speech_file_idx"]
            loc_spk = meta["local_speaker"]
            if sf_idx not in file_spk_mappings:
                file_spk_mappings[sf_idx] = {}
            file_spk_mappings[sf_idx][loc_spk] = f"speaker_{g_label}"

        # Remap transcripts and export files
        for rec in file_records:
            if not rec.get("has_speech", False):
                continue
            sf_idx = rec["speech_file_idx"]
            spk_map = file_spk_mappings.get(sf_idx, {})
            
            for t in rec["transcript"]:
                t["speaker"] = spk_map.get(t["speaker"], t["speaker"])
            
            # Update speakers present list
            rec["speakers"] = sorted(list(set(t["speaker"] for t in rec["transcript"])))
            rec["speaker_count"] = len(rec["speakers"])

            # Save txt transcript
            txt_out = output_dir / f"{rec['stem']}_transcript.txt"
            export_transcript_by_speaker(rec["transcript"], txt_out)

            # Save json transcript
            json_out = output_dir / f"{rec['stem']}_transcript.json"
            with open(json_out, "w", encoding="utf-8") as f:
                f.write(json.dumps(rec["transcript"], indent=4) + "\n")
            
            print(f"\tSaved global transcript for {rec['file_name']} -> {txt_out.name}")

    # Build summary
    summary_file_details = []
    for rec in file_records:
        detail = {
            "file_name": rec["file_name"],
            "has_speech": rec["has_speech"],
            "duration_seconds": round(rec["duration"], 2)
        }
        if rec["has_speech"]:
            detail["speaker_count"] = rec.get("speaker_count", 0)
            detail["speakers"] = rec.get("speakers", [])
        else:
            detail["speaker_count"] = 0
            detail["speakers"] = []
        summary_file_details.append(detail)

    batch_summary = {
        "folder_path": str(folder_path),
        "total_files": len(media_files),
        "files_with_speech": speech_file_count,
        "files_without_speech": len(media_files) - speech_file_count,
        "total_global_speakers": total_global_speakers,
        "file_details": summary_file_details
    }

    summary_path = output_dir / "batch_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(batch_summary, indent=4) + "\n")
    print(f"\nBatch processing complete! Summary written to {summary_path}")

    return batch_summary