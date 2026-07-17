from transcription import transcribe_and_diarize_audio
import time
init_time = time.time()
transcribe_and_diarize_audio(r"C:\Users\Somlab\Downloads\audio1365983309_trunc.wav", verbose=True)
print("Execution time:", time.time() - init_time)