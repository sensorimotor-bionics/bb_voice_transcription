import os
## Disable logging stuff - must be set before nemo is imported to have any effect
os.environ["NEMO_LOG_LEVEL"] = "40"

import json
import shutil
import tempfile
import traceback
import torch
import librosa
import numpy as np
from faster_whisper import WhisperModel
from faster_whisper.transcribe import Segment
from pathlib import Path
from nemo.collections.asr.models import SortformerEncLabelModel, EncDecSpeakerLabelModel
from nemo.utils import logging as nemo_logging
from audio_utils import prepare_audio, prepared_audio_path, PREPARED_AUDIO_SUFFIX
from sklearn.cluster import AgglomerativeClustering
from post_processing import export_transcript_by_speaker, concatenate_batch_transcripts

nemo_logging.set_verbosity(nemo_logging.ERROR)

# Cosine distance below which two speaker embeddings are taken to be the same person.
# It applies to raw (uncentered) L2-normalized TitaNet embeddings, where same-speaker
# pairs typically sit well under 0.3 and different speakers well above it. This is a
# starting point, not a tuned value: run with verbose=True to print the pairwise
# distances for your own recordings (report_speaker_distances) and adjust. Only used
# when the speaker count is unknown; passing num_speakers/global_num_speakers ignores it.
DEFAULT_SPEAKER_DISTANCE_THRESHOLD = 0.35


def _remove_prepared_audio(prepared_path: os.PathLike, original_path: os.PathLike) -> None:
    """
    Delete an intermediate 16 kHz WAV, without ever deleting the caller's input file.

    Path.suffix only ever returns the final suffix (".wav"), so the ".16k.wav" test
    has to be done against the full file name.
    """
    prepared_path, original_path = Path(prepared_path), Path(original_path)
    if prepared_path == original_path:
        return
    if not prepared_path.name.endswith(PREPARED_AUDIO_SUFFIX):
        return
    if prepared_path.exists():
        os.remove(prepared_path)


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
    
    segments, _ = model.transcribe(str(audio_path),
                                   beam_size=5,
                                   word_timestamps=True,
                                   vad_filter=True,
                                   condition_on_previous_text=False,
                                   language="en")
    
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

    # Create the manifest config for the model - had permission issues with the default
    # config. It goes in a temp dir rather than the CWD so that two files with the same
    # stem in different folders, or two runs at once, cannot clobber each other's
    # manifest, and so a failed diarization does not leave one behind.
    manifest_dir = tempfile.mkdtemp(prefix="diar_manifest_")
    manifest_path = Path(manifest_dir) / (Path(audio_path).stem + ".json")
    try:
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
    finally:
        shutil.rmtree(manifest_dir, ignore_errors=True)

    # Return the processed prediction
    return prediction # type: ignore


def diarize_audio(diar_model: SortformerEncLabelModel,
                  audio_path: os.PathLike,
                  segments: list[Segment],
                  audio_duration: int | float,
                  max_audio_length: int | float = 600,
                  min_chunk_gap: float = 1.0,
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

            # Every remaining segment, so the chunk always starts on real speech
            remaining_idx = np.where(segment_times[:,0] > start_time)[0]
            if len(remaining_idx) == 0:
                break
            start_idx = remaining_idx[0]

            # Check that we don't leave a section at the end hanging
            if (audio_duration - start_time) < max_audio_length:
                stop_idx = len(segments) - 1
            else:
                # Prefer to end the chunk on a segment preceded by a real pause so
                # boundaries land in silence rather than mid-utterance.
                in_window = remaining_idx[segment_times[remaining_idx,1] < stop_time]
                gap_idx = in_window[segment_times[in_window,2] > min_chunk_gap]
                if len(gap_idx) > 0:
                    stop_idx = gap_idx[-1]
                elif len(in_window) > 0:
                    # Continuous speech with no pause over min_chunk_gap in this
                    # window: fill the chunk instead of cutting it short.
                    stop_idx = in_window[-1]
                else:
                    # A single segment spans the whole window; keep it intact.
                    stop_idx = start_idx
                stop_idx = max(stop_idx, start_idx)

            # Diarize audio chunk, padding 0.1s either side without running off the file
            chunk_start = max(0.0, segment_times[start_idx,0] - 0.1)
            chunk_stop = min(float(audio_duration), segment_times[stop_idx,1] + 0.1)
            diar_prediction = _diarize(diar_model,
                                      audio_path,
                                      chunk_start,
                                      chunk_stop - chunk_start)
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
        # Diarization labels are per-file, so this is the local speaker. Callers that
        # remap to global labels overwrite "speaker" and leave "local_speaker" intact.
        speaker = assign_speaker(s, speaker_times)
        transcript.append({"start": s.start,
                           "end": s.end,
                           "speaker": speaker,
                           "local_speaker": speaker,
                           "text": s.text.strip()})

    return transcript


def get_transcript_speakers(transcript: list[dict]):
    # Get segments split by predicted speaker
    if not transcript:
        return [], [], []
    speaker = transcript[0]['speaker']
    # Start from the first segment's own timestamp rather than 0: the recording can
    # open with silence, and a run's start time is where its speech starts.
    speaker_start_times, speaker_stop_times, speakers = [transcript[0]['start']], [], [speaker]
    for i, t in enumerate(transcript):
        if t['speaker'] != speaker:
            speaker_start_times.append(t['start'])
            # i is never 0 here (the first segment defines the first speaker), so the
            # previous segment always exists and is where the previous run ended.
            speaker_stop_times.append(transcript[i-1]['end'])
            speaker = t['speaker']
            speakers.append(speaker)

    # Add the final timestamp
    speaker_stop_times.append(transcript[-1]['end'])

    return speakers, speaker_start_times, speaker_stop_times


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """L2-normalize each row, leaving all-zero rows alone rather than dividing by 0."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    return matrix / norms


def speaker_distance_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    Pairwise cosine distance between speaker embeddings: 0 is identical, 1 orthogonal,
    2 opposite. This is the only place speaker similarity is defined, so every caller
    compares embeddings on the same scale as DEFAULT_SPEAKER_DISTANCE_THRESHOLD.
    """
    embeddings = np.asarray(embeddings, dtype=float)
    if len(embeddings) == 0:
        return np.empty((0, 0))
    normalized = _normalize_rows(embeddings)
    return np.clip(1.0 - np.dot(normalized, normalized.T), 0.0, 2.0)


def report_speaker_distances(labels: list[str],
                             distance_mat: np.ndarray,
                             distance_threshold: float = DEFAULT_SPEAKER_DISTANCE_THRESHOLD) -> None:
    """
    Print the pairwise distances behind a clustering decision, for tuning
    distance_threshold on your own recordings: pairs you know are the same person
    should land below the threshold, pairs you know differ should land above it.
    """
    if len(labels) < 2:
        print(f"\tOnly {len(labels)} speaker embedding(s); no pairs to compare.")
        return

    width = max(len(str(label)) for label in labels)
    print(f"\tPairwise speaker distances (threshold {distance_threshold:.2f}):")
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            distance = distance_mat[i, j]
            verdict = "same" if distance <= distance_threshold else "different"
            print(f"\t  {str(labels[i]):<{width}} <-> {str(labels[j]):<{width}}  {distance:.3f}  {verdict}")


def _embed_segments(encoding_model: EncDecSpeakerLabelModel,
                    audio: np.ndarray,
                    spans: list[tuple[int, int]],
                    device: str | torch.device,
                    batch_size: int = 16) -> list[np.ndarray]:
    """
    Embed audio spans in batches, returning one L2-normalized embedding per span.

    Shorter spans are zero-padded up to the longest in their batch and the true lengths
    are passed alongside, so the encoder masks the padding out rather than treating it
    as silence the speaker produced.
    """
    embeddings = []
    for batch_start in range(0, len(spans), batch_size):
        batch = spans[batch_start:batch_start + batch_size]
        lengths = [stop - start for start, stop in batch]
        padded = np.zeros((len(batch), max(lengths)), dtype=np.float32)
        for row, (start, stop) in enumerate(batch):
            padded[row, :stop - start] = audio[start:stop]

        signal = torch.from_numpy(padded).to(device)
        signal_lengths = torch.tensor(lengths, dtype=torch.int32).to(device)
        with torch.no_grad():
            _, emb = encoding_model.forward(input_signal=signal, input_signal_length=signal_lengths)
        emb = emb.cpu().numpy()

        for row in range(len(batch)):
            norm = np.linalg.norm(emb[row])
            if norm > 0:
                embeddings.append(emb[row] / norm)

    return embeddings


def extract_unique_speaker_embeddings(transcript: list[dict],
                                      unique_speakers: list[str],
                                      audio: np.ndarray,
                                      encoding_model: EncDecSpeakerLabelModel,
                                      device: str | torch.device,
                                      sample_frequency: int = 16000,
                                      center: bool = False,
                                      batch_size: int = 16) -> tuple[np.ndarray, list[str]]:
    """
    Averages per-segment speaker embeddings into one L2-normalized embedding per speaker.
    Useful as a standalone function for debugging speaker representations.

    Args:
        transcript (list[dict]): Segments carrying 'speaker', 'start' and 'end'.
        unique_speakers (list[str]): Speaker tags to embed, in the desired row order.
        audio (np.ndarray): Mono waveform at sample_frequency.
        encoding_model (EncDecSpeakerLabelModel): Speaker encoder (e.g. TitaNet).
        device (str | torch.device): Device the encoder lives on.
        sample_frequency (int): Sample rate of `audio`.
        center (bool): Subtract the mean of *this call's* embeddings before normalizing.
            This spreads distances inside one pool, but makes the result depend on the
            pool it was computed in: with two speakers it forces them exactly antipodal,
            and embeddings from separate calls stop being comparable at all. Leave it off
            whenever the output is compared against embeddings from another call.
        batch_size (int): Segments embedded per forward pass.

    Returns:
        tuple[np.ndarray, list[str]]: (embeddings, speakers), where row i of embeddings
        belongs to speakers[i]. Speakers with no usable audio are dropped from both, so
        the two always stay aligned.
    """
    speaker_embeddings, valid_speakers = [], []
    for spk in unique_speakers:
        spans = []
        for seg in (t for t in transcript if t['speaker'] == spk):
            start_idx = int(seg['start'] * sample_frequency)
            stop_idx = int(seg['end'] * sample_frequency)
            if stop_idx - start_idx < 160:  # Skip empty or sub-10ms audio slices
                continue
            spans.append((start_idx, stop_idx))

        seg_embeddings = _embed_segments(encoding_model, audio, spans, device, batch_size)

        # A speaker with no usable audio is dropped rather than represented by a zero
        # row: a zero row survives normalization as garbage and drags a real cluster
        # towards it. Callers keep such speakers on their original label instead.
        if seg_embeddings:
            mean_emb = np.mean(seg_embeddings, axis=0)
            norm = np.linalg.norm(mean_emb)
            if norm > 0:
                speaker_embeddings.append(mean_emb / norm)
                valid_speakers.append(spk)

    if not speaker_embeddings:
        return np.empty((0, 0)), []

    embedding_matrix = np.vstack(speaker_embeddings)
    if center and len(speaker_embeddings) > 1:
        embedding_matrix = embedding_matrix - np.mean(embedding_matrix, axis=0, keepdims=True)

    return _normalize_rows(embedding_matrix), valid_speakers


def cluster_embeddings(distance_mat: np.ndarray,
                       num_speakers: int | None = None,
                       distance_threshold: float = DEFAULT_SPEAKER_DISTANCE_THRESHOLD):
    """
    Group speaker embeddings by cosine distance.

    Args:
        distance_mat (np.ndarray): Square pairwise distance matrix from
            speaker_distance_matrix().
        num_speakers (int | None): Known speaker count, or None to decide from
            distance_threshold.
        distance_threshold (float): Distance at which speakers stop being merged. Also
            used as a sanity check when num_speakers is given: if every pair is closer
            than this, the embeddings all look like one person and no split is made,
            since forcing one would cut a monologue into imaginary speakers.

    Returns:
        np.ndarray: Integer cluster label per row of distance_mat.
    """
    if len(distance_mat) <= 1:
        return np.zeros(len(distance_mat), dtype=int)

    if num_speakers is not None and len(distance_mat) > 1:
        off_diagonal = distance_mat[~np.eye(len(distance_mat), dtype=bool)]
        if off_diagonal.size and off_diagonal.max() <= distance_threshold:
            print(f"\tAll speaker embeddings are within {distance_threshold} of each other "
                  f"(max {off_diagonal.max():.3f}); treating them as one speaker instead of "
                  f"forcing {num_speakers}.")
            return np.zeros(len(distance_mat), dtype=int)

    if num_speakers is None:
        feature_clusterer = AgglomerativeClustering(n_clusters=None,
                                                    metric='precomputed',
                                                    linkage='average',
                                                    distance_threshold=distance_threshold)
    else:
        # Accept anything integral (including numpy ints) but reject 2.5 or "two"
        # rather than silently truncating or leaving the clusterer undefined.
        try:
            n_clusters = int(num_speakers)
        except (TypeError, ValueError):
            raise ValueError(f"num_speakers must be an integer or None, got {num_speakers!r}")
        if n_clusters != num_speakers or n_clusters < 1:
            raise ValueError(f"num_speakers must be a positive integer or None, got {num_speakers!r}")
        feature_clusterer = AgglomerativeClustering(n_clusters=min(n_clusters, len(distance_mat)),
                                                    metric='precomputed',
                                                    linkage='average')

    return feature_clusterer.fit_predict(distance_mat)


def post_hoc_diarization(transcript: list[dict],
                         audio_path: os.PathLike,
                         num_speakers: int | None = None,
                         enc_model: EncDecSpeakerLabelModel | None = None,
                         distance_threshold: float = DEFAULT_SPEAKER_DISTANCE_THRESHOLD,
                         verbose: bool = False):

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

    embedding_matrix, valid_speakers = extract_unique_speaker_embeddings(
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
    distance_mat = speaker_distance_matrix(embedding_matrix)
    if verbose:
        report_speaker_distances(valid_speakers, distance_mat, distance_threshold)
    speaker_labels = cluster_embeddings(distance_mat, num_speakers, distance_threshold)

    # Create mapping dict (e.g. {'speaker_10': 'speaker_0', 'speaker_20': 'speaker_0'})
    spk_to_global = {spk: f"speaker_{label}" for spk, label in zip(valid_speakers, speaker_labels)}

    # Remap transcript. Chunk speakers that produced no embedding aren't in the mapping,
    # so they keep their local label rather than being folded into an arbitrary cluster.
    for t in transcript:
        local_speaker = t.get('local_speaker', t['speaker'])
        t.setdefault('local_speaker', local_speaker)
        t['speaker'] = spk_to_global.get(local_speaker, local_speaker)

    return transcript


def transcribe_and_diarize_audio(audio_path: os.PathLike,
                                 whisper_size: str = "small",
                                 transcription_path: str | os.PathLike | None = None,
                                 max_audio_length: int = 600,
                                 min_speech_duration: float = 0.5,
                                 verbose: bool = False,
                                 num_speakers: int | None = 2,
                                 distance_threshold: float = DEFAULT_SPEAKER_DISTANCE_THRESHOLD,
                                 cleanup: bool = True,
                                 whisper_model: WhisperModel | None = None,
                                 diar_model: SortformerEncLabelModel | None = None,
                                 enc_model: EncDecSpeakerLabelModel | None = None):
    """
    High level function to process, transcribe, and diarize a single audio file.

    Returns:
        list[dict] | None: The transcript segments on success, an empty list if the file
        held no speech or could not be converted, and None if the caller declined to
        overwrite an existing transcript.
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
        transcription_path = Path(os.fsdecode(transcription_path))
        # Check the files we are actually about to write, not the extension-less base
        existing = [p for p in (transcription_path.with_suffix(".txt"),
                                transcription_path.with_suffix(".json")) if p.exists()]
        if existing:
            existing_str = ", ".join(str(p) for p in existing)
            user_input = input(f"{existing_str} already exists. Do you want to overwrite? (y/n): ").strip().lower()
            if user_input not in ('y', 'yes'):
                return None
    print(f"Processing {audio_path}")

    ### Pre-format audio file for quicker processing
    audio_duration = prepare_audio(audio_path)
    if audio_duration < 0:
        print(f"\tFailed to prepare audio for {audio_path}. Skipping.")
        if cleanup:
            _remove_prepared_audio(prepared_audio_path(audio_path), audio_path_bk)
        return []
    audio_path = prepared_audio_path(audio_path)

    if verbose:
        print(f"\tDuration = {audio_duration} seconds")

    ### FasterWhisper transcription
    print("\tInitializing transcription")
    segments = transcribe(audio_path, whisper_size, model=whisper_model)
    
    if not detect_speech(segments, min_speech_duration=min_speech_duration):
        print("\tNo speech detected in audio file.")
        if cleanup:
            _remove_prepared_audio(audio_path, audio_path_bk)
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
        transcript = post_hoc_diarization(transcript,
                                          audio_path,
                                          num_speakers,
                                          enc_model=enc_model,
                                          distance_threshold=distance_threshold,
                                          verbose=verbose)

    # Dump the transcript as .txt
    export_transcript_by_speaker(transcript, transcription_path.with_suffix(".txt"))

    # Dump the transcript as .json for later parsing
    out_path = transcription_path.with_suffix(".json")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(transcript, indent=4) + "\n")

    print("\tSaved transcript to", out_path)

    if cleanup:
        _remove_prepared_audio(audio_path, audio_path_bk)

    return transcript


def transcribe_and_diarize_folder(
    folder_path: os.PathLike,
    output_dir: os.PathLike | None = None,
    whisper_size: str = "small",
    video_extensions: list[str] | None = None,
    max_audio_length: int = 600,
    min_speech_duration: float = 0.5,
    global_num_speakers: int | None = None,
    distance_threshold: float = DEFAULT_SPEAKER_DISTANCE_THRESHOLD,
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
        distance_threshold (float): Cosine distance below which two local speakers are
            merged into one global speaker. Only used when global_num_speakers is None;
            see DEFAULT_SPEAKER_DISTANCE_THRESHOLD, and pass verbose=True to print the
            pairwise distances it is compared against.
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

    # Drop leftover 16 kHz intermediates whose source file is also here (from a run that
    # crashed or ran with cleanup=False), so one recording is not transcribed twice.
    sources = {f.name for f in media_files}
    redundant = [f for f in media_files
                 if f.name.endswith(PREPARED_AUDIO_SUFFIX)
                 and any(f"{f.name[:-len(PREPARED_AUDIO_SUFFIX)]}{ext}" in sources
                         for ext in video_extensions)]
    if redundant:
        print(f"Ignoring {len(redundant)} leftover intermediate file(s): "
              f"{', '.join(f.name for f in redundant)}")
        media_files = [f for f in media_files if f not in redundant]

    if not media_files:
        print(f"No media files with extensions {video_extensions} found in {folder_path}")
        return {"total_files": 0, "files_with_speech": 0, "files_without_speech": 0, "files": []}

    print(f"Found {len(media_files)} media file(s) in {folder_path}")

    # Initialize models once
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "auto" if device == "cuda" else "int8"
    print(f"Loading models on {device} ({compute_type})...")

    whisper_model = WhisperModel(whisper_size, device=device, compute_type=compute_type)
    diar_model = SortformerEncLabelModel.from_pretrained("nvidia/diar_sortformer_4spk-v1").to(device).eval() # type: ignore
    enc_model = EncDecSpeakerLabelModel.from_pretrained(model_name="titanet_small").to(device).eval() # type: ignore
    # Read the device back off the encoder rather than trusting the string: the audio
    # tensors have to be created wherever its weights actually ended up.
    enc_device = next(enc_model.parameters()).device

    file_records = []
    all_speaker_embeddings = []
    embedding_metadata = [] # stores {"speech_file_idx": int, "local_speaker": str}

    speech_file_count = 0

    for idx, file_path in enumerate(media_files, start=1):
        print(f"\n--- [{idx}/{len(media_files)}] Processing: {file_path.name} ---")
        wav_path = prepared_audio_path(file_path)

        # One unusual file shouldn't abandon a folder that may already have hours of
        # transcription in it, so failures are recorded and the batch carries on.
        try:
            # Prepare 16kHz WAV audio
            audio_duration = prepare_audio(file_path, wav_path)

            if audio_duration < 0:
                print(f"Error processing audio for {file_path.name}. Skipping.")
                file_records.append({
                    "file_name": file_path.name,
                    "stem": file_path.stem,
                    "has_speech": False,
                    "duration": 0.0,
                    "speaker_count": 0,
                    "transcript": [],
                    "speakers": [],
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
                continue

            print(f"\tSpeech detected! ({audio_duration:.1f}s) Running diarization...")
            speaker_times, _ = diarize_audio(diar_model, wav_path, segments, audio_duration, max_audio_length, verbose=verbose)
            transcript = create_transcript(segments, speaker_times)

            # Extract speaker embeddings for local unique speakers
            unique_speakers = list(dict.fromkeys(t['speaker'] for t in transcript))
            audio, _ = librosa.load(wav_path, sr=16000)

            if verbose:
                print(transcript)

            # center=False: these embeddings are pooled with every other file's below, so
            # they have to stay in one shared geometry rather than each file's own.
            emb_matrix, valid_speakers = extract_unique_speaker_embeddings(
                transcript=transcript,
                unique_speakers=unique_speakers,
                audio=audio,
                encoding_model=enc_model,
                device=enc_device,
                sample_frequency=16000,
                center=False
            )

            for local_spk, embedding in zip(valid_speakers, emb_matrix):
                all_speaker_embeddings.append(embedding)
                embedding_metadata.append({
                    "speech_file_idx": speech_file_count,
                    "stem": file_path.stem,
                    "local_speaker": local_spk
                })

            skipped_speakers = [s for s in unique_speakers if s not in valid_speakers]
            if skipped_speakers:
                print(f"\tNo usable audio for {', '.join(skipped_speakers)}; "
                      "keeping their local labels.")

            file_records.append({
                "file_name": file_path.name,
                "stem": file_path.stem,
                "has_speech": True,
                "duration": audio_duration,
                "speech_file_idx": speech_file_count,
                "transcript": transcript
            })
            speech_file_count += 1

        except Exception as exc:
            print(f"\tFAILED {file_path.name}: {type(exc).__name__}: {exc}")
            if verbose:
                traceback.print_exc()
            file_records.append({
                "file_name": file_path.name,
                "stem": file_path.stem,
                "has_speech": False,
                "duration": 0.0,
                "speaker_count": 0,
                "transcript": [],
                "speakers": [],
                "error": f"{type(exc).__name__}: {exc}"
            })

        finally:
            if cleanup:
                _remove_prepared_audio(wav_path, file_path)

    # Perform global speaker classification across all speech files. Clustering is kept
    # separate from exporting: if there is nothing to cluster, or the clustering itself
    # fails, transcripts are still written with their per-file labels rather than the
    # whole folder's transcription work being discarded.
    file_spk_mappings = {}
    if all_speaker_embeddings:
        print(f"\nRunning cross-file speaker clustering across {len(all_speaker_embeddings)} local speaker representation(s)...")
        try:
            global_matrix = np.vstack(all_speaker_embeddings)
            distance_mat = speaker_distance_matrix(global_matrix)

            if verbose:
                report_speaker_distances([f"{m['stem']}:{m['local_speaker']}" for m in embedding_metadata],
                                         distance_mat,
                                         distance_threshold)

            global_labels = cluster_embeddings(distance_mat,
                                               num_speakers=global_num_speakers,
                                               distance_threshold=distance_threshold)
            if global_num_speakers is None:
                print(f"\tFound {len(set(global_labels))} global speaker(s) "
                      f"at distance_threshold={distance_threshold}")
            else:
                print(f"\tSplit into {len(set(global_labels))} global speaker(s) as requested")

            # Build mapping: speech_file_idx -> {local_speaker -> global_speaker_label}
            for meta, g_label in zip(embedding_metadata, global_labels):
                sf_idx = meta["speech_file_idx"]
                loc_spk = meta["local_speaker"]
                if sf_idx not in file_spk_mappings:
                    file_spk_mappings[sf_idx] = {}
                file_spk_mappings[sf_idx][loc_spk] = f"speaker_{g_label}"
        except Exception as exc:
            print(f"\tCross-file speaker clustering failed ({type(exc).__name__}: {exc}); "
                  "keeping per-file speaker labels.")
            file_spk_mappings = {}
    else:
        print("\nNo speaker embeddings to cluster; keeping per-file speaker labels.")

    # Remap transcripts and export files
    for rec in file_records:
        if not rec.get("has_speech", False):
            continue
        spk_map = file_spk_mappings.get(rec["speech_file_idx"], {})

        # Key off local_speaker, not speaker, so the remap stays correct even if
        # a global label collides with a local one (both are "speaker_<n>").
        for t in rec["transcript"]:
            t.setdefault("local_speaker", t["speaker"])
            t["speaker"] = spk_map.get(t["local_speaker"], t["speaker"])

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

        print(f"\tSaved transcript for {rec['file_name']} -> {txt_out.name}")

    summary = {
        "total_files": len(media_files),
        "files_with_speech": speech_file_count,
        "files_without_speech": len(media_files) - speech_file_count,
        "output_dir": str(output_dir),
        "files": file_records,
    }

    # Dump the summary, pointing at each per-file transcript instead of repeating it.
    # concatenate_batch_transcripts follows "transcript_file" when it reads this back.
    summary_path = output_dir / "batch_summary.json"
    disk_summary = dict(summary)
    disk_summary["files"] = [
        {**{k: v for k, v in rec.items() if k != "transcript"},
         **({"transcript_file": f"{rec['stem']}_transcript.json"} if rec.get("has_speech") else {})}
        for rec in file_records
    ]
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(disk_summary, f, indent=4)

    print(f"\tSaved batch summary to {summary_path}")
    # Dump concatenated transcript
    concatenate_batch_transcripts(summary, output_path=output_dir)

    print("Batch processing complete!")
    return summary
