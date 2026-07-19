import time
init_time = time.time()

from transcription import transcribe_and_diarize_audio
transcribe_and_diarize_audio(r"C:\Users\Somlab\Downloads\audio1365983309.wav",
                             verbose=True,
                             num_speakers=2)

print("Execution time:", time.time() - init_time)