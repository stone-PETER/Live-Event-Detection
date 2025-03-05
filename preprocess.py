import os
import librosa
import soundfile as sf
import numpy as np
import sounddevice as sd
import wave

def preprocess_audio_files(audio_dir):
    categories = [d for d in os.listdir(audio_dir) if os.path.isdir(os.path.join(audio_dir, d))]
    
    for category in categories:
        category_path = os.path.join(audio_dir, category)
        for file_name in os.listdir(category_path):
            if file_name.endswith('.wav'):
                file_path = os.path.join(category_path, file_name)
                y, sr = librosa.load(file_path, sr=None)
                # Example preprocessing: normalize audio
                y = librosa.util.normalize(y)
                # Save the preprocessed audio
                sf.write(file_path, y, sr)

def record_environment_sounds(output_dir, num_clips=20, duration=6, sample_rate=44100):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print("Recording environment sounds...")
    for i in range(num_clips):
        frames = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
        sd.wait()  # Wait until recording is finished
        
        output_file = os.path.join(output_dir, f'env_clip_{i+1}.wav')
        wf = wave.open(output_file, 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 2 bytes for int16
        wf.setframerate(sample_rate)
        wf.writeframes(frames.tobytes())
        wf.close()
        print(f"Saved {output_file}")
    
    print("Finished recording environment sounds.")

if __name__ == "__main__":
    audio_dir = 'audio'
    env_audio_dir = os.path.join(audio_dir, 'env_audio')
    
    preprocess_audio_files(audio_dir)
    record_environment_sounds(env_audio_dir)