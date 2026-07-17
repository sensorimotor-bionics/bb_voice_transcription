import subprocess

def trim_wav(input_file, output_file, duration=90):
    """
    Trims a WAV file to a specified duration using ffmpeg.
    """
    command = [
        'ffmpeg',
        '-ss', '0',           # Start at the very beginning
        '-i', input_file,     # Input file
        '-t', str(duration),  # Duration in seconds
        '-c', 'copy',         # Copy audio without re-encoding (preserves quality)
        output_file
    ]
    
    try:
        subprocess.run(command, check=True)
        print(f"Successfully trimmed {input_file} to {output_file}.")
    except subprocess.CalledProcessError as e:
        print(f"Error trimming audio: {e}")

# Example usage:
trim_wav(r"C:\Users\Somlab\Downloads\audio1365983309.16k.wav", r"C:\Users\Somlab\Downloads\audio1365983309_trunc.16k.wav")