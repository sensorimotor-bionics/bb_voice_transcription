import os
import json
import torch
import librosa
import numpy as np
from faster_whisper import WhisperModel
from pathlib import Path
from nemo.collections.asr.models import SortformerEncLabelModel, EncDecSpeakerLabelModel
from audio_utils import prepare_audio
from sklearn.cluster import AgglomerativeClustering

## Disable logging stuff
from nemo.utils import logging as nemo_logging
os.environ["NEMO_LOG_LEVEL"] = "40"
nemo_logging.set_verbosity(nemo_logging.ERROR)


def transcribe(audio_path: os.PathLike, whisper_size: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "auto" if device == "cuda" else "int8"
    model = WhisperModel(whisper_size, device=device, compute_type=compute_type)
    segments, _ = model.transcribe(audio_path, beam_size=5, word_timestamps=True)
    
    return list(segments) # Full list of word segments for whole file


def format_diarization(prediction, speaker_offset = 0):
    speaker_turns = []
    for line in prediction:
        start, end, speaker = line.split()
        if speaker_offset > 0:
            spk_int = int(speaker[-1])
            speaker = f"speaker_{spk_int+speaker_offset}"
        speaker_turns.append({"start": float(start), "end": float(end), "speaker": speaker})

    return speaker_turns


def diarize(diarizatrion_model: SortformerEncLabelModel, 
            audio_path: os.PathLike,
            offset: int,
            audio_duration: int):

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


def export_transcript_by_speaker(transcript:list, out_path:os.PathLike):
    with open(out_path, "w", encoding="utf-8") as f:
        last = None
        for row in transcript:
            if row["speaker"] != last:
                f.write(f"\n[{row['speaker']}]\n")
                last = row["speaker"]
            f.write(f"({row['start']:.1f}-{row['end']:.1f}) {row['text']}\n")


def relabel_transcript(transcript_path:os.PathLike, speaker_ids:list[str]):
    with open(transcript_path, 'r', encoding='utf8') as f: 
        transcript = json.load(f)

    # Get list of unique speakers and confirm lengths match
    speaker_id = [t['speaker'].split('_')[-1] for t in transcript]
    speaker_list = list(set(speaker_id))
    assert len(speaker_list) == len(speaker_ids), f"{len(speaker_list)} speakers in transcript but only {len(speaker_ids)} provided"

    # Iterate through the transcript and relabel according to speaker id

def transcribe_and_diarize_audio(audio_path: os.PathLike,
                                 whisper_size: str = "small",
                                 transcription_path: str | None = None,
                                 max_audio_length: int = 600,
                                 verbose: bool = False,
                                 num_speakers: int | None = 2):
    
    # Check that the audio path exists
    assert os.path.isfile(audio_path), f"{audio_path} is not a file"
    audio_path = Path(audio_path)

    # Ensure whisper model is valid
    valid_whisper_models = ["tiny", "base", "small", "medium", "large-v3"]
    assert whisper_size in valid_whisper_models, f"'{whisper_size}' was not found in {valid_whisper_models}"
    
    # Parse the transcription path
    if transcription_path is None:
        transcription_path = audio_path.stem + "_transcript.txt"
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
    chunked_diarization = audio_duration > max_audio_length # Bool for whether we chunk or not

    if verbose:
        print(f"\tDuration = {audio_duration} seconds")

    ### FasterWhisper transcription
    print("\tInitalizing transcription")
    segments = transcribe(audio_path, whisper_size)
    
    ### NeMo diarization
    print("\tInitalizing diarization")
    diar_model = SortformerEncLabelModel.from_pretrained("nvidia/diar_sortformer_4spk-v1").eval()
    
    # If audio duration exceeds max then chunk
    if chunked_diarization:
        # Identify gaps in segments
        segment_times = np.zeros((len(segments), 3))
        for i, segment in enumerate(segments):
            segment_times[i,0] = segment.start
            segment_times[i,1] = segment.end

        segment_times[1:,2] = segment_times[1:,1] - segment_times[:-1,0]

        # Loop through segments to find gaps and diarize in chunks
        start_time, stop_time = -1, max_audio_length
        speaker_times = []
        chunk_counter = 1
        while start_time < segment_times[-1,1]:
            print(f"\t- Chunk {chunk_counter}")
            # Find the indices of the segments in the range of the start and stop time
            sub_seg_idx = np.where(((segment_times[:,0] > start_time) & 
                            (segment_times[:,1] < stop_time) & 
                            (segment_times[:,2] > 1)))[0]
            start_idx = sub_seg_idx[0]
            stop_idx = sub_seg_idx[-1]
            
            # Diarize audio chunk
            diar_prediction = diarize(diar_model,
                                      audio_path,
                                      segment_times[start_idx,0] - 0.1,
                                      segment_times[stop_idx,1] - segment_times[start_idx,0] + 0.2)
            speaker_times += format_diarization(diar_prediction[0], chunk_counter*10) # Append speaker times for the chunk

            # Update 
            start_time = segment_times[stop_idx,1]
            stop_time = start_time + max_audio_length
            chunk_counter += 1        
    
    else: # Otherwise process the whole file
        diar_prediction = diarize(diar_model, audio_path, 0, audio_duration)
        speaker_times = format_diarization(diar_prediction[0])

    # Create the transcript from segments and speaker times
    transcript = create_transcript(segments, speaker_times)

    # Harmonize diarization across chunks
    if chunked_diarization:
        # Get segments split by predicted speaker
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

        # Get an encoding model for feature extraction
        enc_model = EncDecSpeakerLabelModel.from_pretrained(model_name="titanet_small").eval()
        device = next(enc_model.parameters()).device
        # load the whole audio file
        sample_frequency = 16000
        audio, _ = librosa.load(audio_path, sr=sample_frequency)

        # Manually compute embeddings
        embeddings = []
        for (start, stop) in zip(speaker_start_times, speaker_stop_times):
            # Get segment indices for audio file
            start_idx = int(start * sample_frequency)
            stop_idx = int(stop * sample_frequency)
            audio_tensor = torch.tensor(audio[start_idx:stop_idx], dtype=torch.float32).unsqueeze(0).to(device)
            audio_len = torch.tensor([audio_tensor.shape[1]], dtype=torch.int32).to(device)

            # Get the embedding from the encoding model
            with torch.no_grad():
                _, embedding = enc_model.forward(input_signal=audio_tensor, input_signal_length=audio_len)

            # Normalize and return to CPU
            embedding = embedding.squeeze(0).cpu().numpy()
            embeddings.append(embedding / np.linalg.norm(embedding))

        # Compute similarity between each pair of embeddings
        similarity_mat = np.zeros((len(embeddings), len(embeddings)))
        for i in range(len(embeddings)):
            for j in range(len(embeddings)):
                similarity_mat[i,j] = np.dot(embeddings[i], embeddings[j]) / ((np.dot(embeddings[i], embeddings[i]) * np.dot(embeddings[j], embeddings[j])) ** 0.5)
        
        # Convert to distance matrix for linkage assessement
        distance_mat = 1-similarity_mat

        # Cluster to get harmonized speaker labels
        if num_speakers is None:
            feature_clusterer = AgglomerativeClustering(n_clusters=None, metric='precomputed', linkage='average', distance_threshold=0.75)
        elif isinstance(num_speakers, int):
            feature_clusterer = AgglomerativeClustering(n_clusters=2, metric='precomputed', linkage='average')
        feature_labels = feature_clusterer.fit_predict(distance_mat)
        print(np.unique(feature_labels))

        export_transcript_by_speaker(transcript, audio_path.with_suffix(".transcript_raw.txt"))

        # Update the transcript
        speaker_map = {name: f"speaker_{feature_labels[idx]}" for idx, name in enumerate(speakers)}
        for t in transcript:
            t['speaker'] = speaker_map[t['speaker']]

        export_transcript_by_speaker(transcript, audio_path.with_suffix(".transcript_relabeled.txt"))

    # Dump the transcript as .txt
    export_transcript_by_speaker(transcript, audio_path.with_suffix(".transcript.txt"))

    # Dump the transcript as .json for later parsing
    out_path = audio_path.with_suffix(".transcript.json")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(transcript, indent=4) + "\n")

    print("\tSaved transcript to", out_path)