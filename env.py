import os
import time
import uuid
import numpy as np
import sounddevice as sd
import soundfile as sf
import wave
import threading
import queue

# Configuration
SAMPLE_RATE = 44100  # 44.1 kHz
CHANNELS = 1  # Mono recording
TOTAL_DURATION = 100  # Total recording duration in seconds
CLIP_DURATION = 5  # Each clip duration in seconds

OUTPUT_DIR = "env_audio"  # Output directory
FORMAT = "wav"  # Audio format

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Create a queue to store audio data
audio_queue = queue.Queue()

def record_audio():
    """Record audio for the specified duration"""
    print(f"Starting audio recording for {TOTAL_DURATION} seconds...")
    
    # Calculate buffer size
    buffer_size = int(SAMPLE_RATE * TOTAL_DURATION)
    
    # Create buffer to hold all recorded data
    buffer = np.zeros((buffer_size, CHANNELS), dtype=np.float32)
    
    # Create counter to track position in buffer
    counter = [0]
    
    # Callback function to process audio data
    def callback(indata, frames, time_info, status):
        if status:
            print(f"Audio callback status: {status}")
        
        # Calculate remaining space in buffer
        remaining = buffer_size - counter[0]
        
        if remaining <= 0:
            # Buffer is full, stop recording
            raise sd.CallbackStop()
        
        # Calculate how many frames to copy
        frames_to_copy = min(frames, remaining)
        
        # Copy data to buffer
        buffer[counter[0]:counter[0] + frames_to_copy] = indata[:frames_to_copy]
        
        # Update counter
        counter[0] += frames_to_copy
        
        # Display progress
        progress = counter[0] / buffer_size * 100
        if counter[0] % (SAMPLE_RATE * 5) < frames:  # Every 5 seconds
            print(f"Recording progress: {progress:.1f}%")
    
    # Start recording
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, callback=callback):
        print("Recording started. Please speak...")
        
        # Wait until recording is complete
        while counter[0] < buffer_size:
            sd.sleep(100)  # Sleep for 100ms
    
    print("Recording complete!")
    
    # Save all data to the queue for processing
    audio_queue.put(buffer[:counter[0]])

def split_and_save():
    """Split recorded audio into clips and save them"""
    # Get recorded audio from queue
    audio_data = audio_queue.get()
    
    # Calculate number of complete clips
    num_samples = len(audio_data)
    samples_per_clip = int(SAMPLE_RATE * CLIP_DURATION)
    num_clips = num_samples // samples_per_clip
    
    print(f"Splitting audio into {num_clips} clips of {CLIP_DURATION} seconds each...")
    
    # Split and save each clip
    for i in range(num_clips):
        # Generate random filename
        filename = f"env_audio_{uuid.uuid4().hex[:8]}.{FORMAT}"
        filepath = os.path.join(OUTPUT_DIR, filename)
        
        # Get clip data
        start_idx = i * samples_per_clip
        end_idx = start_idx + samples_per_clip
        clip_data = audio_data[start_idx:end_idx]
        
        # Save clip
        sf.write(filepath, clip_data, SAMPLE_RATE)
        print(f"Saved clip {i+1}/{num_clips}: {filename}")
    
    print(f"All {num_clips} audio clips saved to {OUTPUT_DIR}/ directory")

if __name__ == "__main__":
    print("Environmental Audio Recording Tool")
    print("=================================")
    print(f"This script will record {TOTAL_DURATION} seconds of audio")
    print(f"and split it into {CLIP_DURATION}-second clips.")
    print(f"Output will be saved to the '{OUTPUT_DIR}/' directory.")
    print()
    
    input("Press Enter to begin recording...")
    
    # Start recording in a separate thread
    recording_thread = threading.Thread(target=record_audio)
    recording_thread.start()
    
    # Wait for recording to complete
    recording_thread.join()
    
    # Process and save the clips
    split_and_save()
    
    print("Done!")