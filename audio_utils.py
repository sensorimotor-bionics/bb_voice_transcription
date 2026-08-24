import json
import subprocess
from os import PathLike
from pathlib import Path

# Suffix of the intermediate 16 kHz mono WAV that the rest of the pipeline reads.
PREPARED_AUDIO_SUFFIX = ".16k.wav"


def get_audio_duration(input_file: PathLike) -> float:
    """
    Duration of a media file in seconds via ffprobe, or -1.0 if it cannot be read.

    ffprobe only reads container metadata, so this stays cheap on hour-long video
    where decoding the whole stream just to measure it would cost hundreds of MB.
    """
    command = ["ffprobe", "-v", "error",
               "-show_entries", "format=duration:stream=duration",
               "-of", "json", str(input_file)]
    try:
        probe = subprocess.run(command, check=True, capture_output=True, text=True)
        data = json.loads(probe.stdout)
    except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError) as e:
        stderr = (getattr(e, "stderr", "") or "").strip()
        print(f"Error reading duration of {input_file}: {stderr or e}")
        return -1.0

    # Some containers only carry the duration on the stream, not the format
    candidates = [data.get("format", {}).get("duration")]
    candidates += [s.get("duration") for s in data.get("streams", [])]
    for value in candidates:
        try:
            return float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue

    print(f"Error reading duration of {input_file}: no duration in ffprobe output")
    return -1.0


def prepared_audio_path(input_file: PathLike) -> Path:
    """
    Where prepare_audio writes the 16 kHz mono WAV for this input.

    A file that is already a prepared WAV maps to itself. Deriving the name blindly
    would give "clip.16k.16k.wav", which prepare_audio never writes, leaving callers
    to read a file that does not exist.
    """
    input_file = Path(input_file)
    if input_file.name.endswith(PREPARED_AUDIO_SUFFIX):
        return input_file
    return input_file.with_suffix(PREPARED_AUDIO_SUFFIX)


def prepare_audio(input_file: PathLike,
                  output_file: PathLike | None = None) -> float:
    """
    Uses ffmpeg to ensure a mono 16 kHz WAV exists for input_file.

    Args:
        input_file (PathLike): Source audio or video file.
        output_file (PathLike | None): Destination WAV. Defaults to prepared_audio_path().

    Returns:
        float: Duration of the prepared audio in seconds, or -1.0 if the file could not
        be read or converted. On success the returned duration always describes a file
        that exists on disk.
    """
    input_file = Path(input_file)
    output_file = prepared_audio_path(input_file) if output_file is None else Path(output_file)

    # Already a prepared WAV: nothing to convert, just report its duration
    if output_file == input_file:
        return get_audio_duration(input_file)

    command = ["ffmpeg", "-y", "-loglevel", "error",
               "-i", str(input_file),
               "-vn",                  # drop any video stream
               "-ac", "1",             # mono
               "-ar", "16000",         # 16 kHz
               "-c:a", "pcm_s16le",
               str(output_file)]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        stderr = (getattr(e, "stderr", "") or "").strip()
        print(f"Error converting {input_file} with ffmpeg: {stderr or e}")
        return -1.0

    if not output_file.exists() or output_file.stat().st_size == 0:
        print(f"Error converting {input_file}: ffmpeg wrote no output to {output_file}")
        return -1.0

    # Measure the converted file rather than the source: a video container can report a
    # duration longer than its audio track, and everything downstream indexes the WAV.
    return get_audio_duration(output_file)
