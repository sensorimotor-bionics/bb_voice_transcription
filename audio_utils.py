import subprocess
from pydub import AudioSegment
from os import PathLike
from pathlib import Path

def trim_wav(input_file: PathLike, 
             output_file: PathLike, 
             start_time: int | float = 0,
             stop_time: int | float = 0):
    """
    Trims a WAV file to a specified duration using ffmpeg.
    """
    command = [
        'ffmpeg',
        '-ss', str(start_time),           # Start at the very beginning
        '-i', input_file,     # Input file
        '-t', str(stop_time),  # Duration in seconds
        '-c', 'copy',         # Copy audio without re-encoding (preserves quality)
        output_file
    ]
    
    try:
        subprocess.run(command, check=True)
        print(f"Successfully trimmed {input_file} to {output_file}.")
    except subprocess.CalledProcessError as e:
        print(f"Error trimming audio: {e}")


def prepare_audio(input_file: PathLike,
                  output_file: PathLike | None = None) -> float:
    # Simple rename
    if output_file is None:
        output_file = Path(input_file).with_suffix(".16k.wav")

    try:
        audio = AudioSegment.from_file(input_file)
        audio.export(output_file, format="wav", parameters=["-ac", "1", "-ar", "16000"])
        return len(audio)/1000 # 
    except Exception as e:
        print(f"Error converting {input_file} with FFMPEG: {e}")