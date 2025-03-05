import os
import json
import time
import threading
import numpy as np
import librosa
import sounddevice as sd
import tensorflow as tf
from tensorflow.keras.models import load_model
import queue
import atexit
import signal
from flask import Flask, jsonify, render_template, send_from_directory
from flask_socketio import SocketIO

# Configuration
MODEL_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_DIR, 'audio_event_detector.h5')
CATEGORIES_PATH = os.path.join(MODEL_DIR, 'categories.json')

# Audio settings
SAMPLE_RATE = 44100
WINDOW_SIZE = 1024  # ~23ms at 44.1kHz
HOP_LENGTH = 512   # ~12ms at 44.1kHz
BUFFER_DURATION = 2  # seconds for each analysis window
DETECTION_THRESHOLD = 0.85  # Increased from 0.7 to reduce false positives

# Initialize Flask app
app = Flask(__name__, static_folder='.', static_url_path='')
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables
audio_buffer = queue.Queue()
stop_recording = threading.Event()
detection_thread = None
model = None
categories = {}
audio_stream = None  # Keep a reference to the audio stream

# Add these global variables after your existing ones
recent_predictions = []
PREDICTION_HISTORY_SIZE = 3  # Number of recent predictions to consider
MIN_CONSISTENT_PREDICTIONS = 2  # Minimum number of consistent predictions required

def load_trained_model():
    """Load the trained model and categories"""
    global model, categories
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Please train a model first.")
    
    if not os.path.exists(CATEGORIES_PATH):
        raise FileNotFoundError(f"Categories file not found at {CATEGORIES_PATH}. Please train a model first.")
    
    # Load model
    model = load_model(MODEL_PATH)
    print(f"Model loaded from {MODEL_PATH}")
    
    # Load categories
    with open(CATEGORIES_PATH, 'r') as f:
        categories = json.load(f)
    
    print(f"Loaded {len(categories)} categories: {list(categories.values())}")
    
    return model, categories

def audio_callback(indata, frames, time_info, status):
    """Callback function for audio stream"""
    if status:
        print(f"Audio callback status: {status}")
        
    # Convert to mono if needed
    if indata.shape[1] > 1:
        data = np.mean(indata, axis=1)
    else:
        data = indata.flatten()
    
    # Put the audio data in the queue
    audio_buffer.put(data.copy())

def extract_features(audio_data, sr=SAMPLE_RATE):
    """Extract MFCC features from audio data"""
    # Normalize audio
    audio_data = librosa.util.normalize(audio_data)
    
    # Extract MFCCs
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=20, n_fft=WINDOW_SIZE, hop_length=HOP_LENGTH)
    
    # Transpose to have time as the first dimension
    mfccs = mfccs.T
    
    # Ensure consistent length (trim or pad)
    target_length = 128  # This should match what was used during training
    if mfccs.shape[0] > target_length:
        mfccs = mfccs[:target_length, :]
    else:
        # Pad with zeros
        pad_width = target_length - mfccs.shape[0]
        mfccs = np.pad(mfccs, ((0, pad_width), (0, 0)), mode='constant')
    
    return mfccs

def detect_events():
    """Process audio buffer and detect events"""
    global model, categories, stop_recording, recent_predictions
    
    buffer_samples = int(SAMPLE_RATE * BUFFER_DURATION)
    audio_data = np.zeros(buffer_samples)
    samples_collected = 0
    
    last_detection_time = time.time()
    cooldown_period = 1.0  # seconds between detections to avoid rapid repeats
    
    print("Starting audio event detection...")
    
    while not stop_recording.is_set():
        try:
            # Get data from the buffer queue
            if not audio_buffer.empty():
                new_data = audio_buffer.get()
                
                # Add to our running buffer
                data_length = min(len(new_data), buffer_samples - samples_collected)
                audio_data[samples_collected:samples_collected + data_length] = new_data[:data_length]
                samples_collected += data_length
                
                # If we have enough samples, process them
                if samples_collected >= buffer_samples:
                    current_time = time.time()
                    
                    # Extract features
                    features = extract_features(audio_data)
                    
                    # Reshape for model input (add batch dimension)
                    features = np.expand_dims(features, axis=0)
                    
                    # Make prediction
                    predictions = model.predict(features, verbose=0)[0]
                    max_prob = np.max(predictions)
                    predicted_class = np.argmax(predictions)
                    
                    # Store prediction in history
                    recent_predictions.append((predicted_class, max_prob))
                    if len(recent_predictions) > PREDICTION_HISTORY_SIZE:
                        recent_predictions.pop(0)
                    
                    # Only process if we're not in cooldown period
                    if current_time - last_detection_time >= cooldown_period:
                        # Check for consistent predictions
                        class_counts = {}
                        for cls, prob in recent_predictions:
                            if prob >= DETECTION_THRESHOLD:
                                class_counts[cls] = class_counts.get(cls, 0) + 1
                        
                        # Find the most consistent class with high confidence
                        most_consistent_class = None
                        most_consistent_count = 0
                        for cls, count in class_counts.items():
                            if count > most_consistent_count and count >= MIN_CONSISTENT_PREDICTIONS:
                                most_consistent_class = cls
                                most_consistent_count = count
                        
                        # If we have a consistent prediction above threshold
                        if most_consistent_class is not None:
                            event_name = categories.get(str(most_consistent_class), f"Unknown-{most_consistent_class}")
                            
                            # Calculate average confidence for this class
                            class_probs = [prob for cls, prob in recent_predictions if cls == most_consistent_class]
                            avg_confidence = sum(class_probs) / len(class_probs)
                            
                            print(f"Detected: {event_name} (confidence: {avg_confidence:.2f}, consistency: {most_consistent_count}/{PREDICTION_HISTORY_SIZE})")
                            
                            # Emit to web clients
                            socketio.emit('detection', {
                                'event': event_name,
                                'confidence': float(avg_confidence),
                                'consistency': most_consistent_count,
                                'timestamp': current_time
                            })
                            
                            last_detection_time = current_time
                    
                    # Slide the buffer (keep the last half for overlap)
                    half_buffer = buffer_samples // 2
                    audio_data[:half_buffer] = audio_data[half_buffer:]
                    samples_collected = half_buffer
            
            else:
                # If queue is empty, wait a bit
                time.sleep(0.01)
                
        except Exception as e:
            print(f"Error in detection loop: {e}")
            time.sleep(0.1)  # Prevent tight error loops

def start_detection():
    """Start the audio stream and detection thread"""
    global detection_thread, stop_recording, audio_stream
    
    # Reset the stop flag
    stop_recording.clear()
    
    try:
        # Start audio stream
        audio_stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            callback=audio_callback
        )
        audio_stream.start()
        
        # Start detection thread
        detection_thread = threading.Thread(target=detect_events)
        detection_thread.daemon = True
        detection_thread.start()
        
        return True
    except Exception as e:
        print(f"Error starting detection: {e}")
        return False

def stop_detection():
    """Stop the audio stream and detection thread"""
    global stop_recording, audio_stream
    
    # Set the stop flag to stop the detection thread
    stop_recording.set()
    
    # Wait for thread to finish
    if detection_thread and detection_thread.is_alive():
        detection_thread.join(timeout=2.0)
    
    # Stop the audio stream
    if audio_stream:
        audio_stream.stop()
        audio_stream.close()
    
    return True

# Flask routes
@app.route('/')
def index():
    """Serve the main detection interface"""
    return app.send_static_file('detection.html')

@app.route('/status')
def status():
    """Return the status of the detection system"""
    model_loaded = model is not None
    categories_loaded = len(categories) > 0
    is_detecting = detection_thread is not None and detection_thread.is_alive()
    
    return jsonify({
        'model_loaded': model_loaded,
        'categories_loaded': categories_loaded,
        'categories': list(categories.values()) if categories else [],
        'is_detecting': is_detecting
    })

@app.route('/start', methods=['POST'])
def start():
    """Start the detection process"""
    success = start_detection()
    return jsonify({'success': success})

@app.route('/stop', methods=['POST'])
def stop():
    """Stop the detection process"""
    success = stop_detection()
    return jsonify({'success': success})

# SocketIO events
@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print('Client connected')
    socketio.emit('status', {
        'model_loaded': model is not None,
        'is_detecting': detection_thread is not None and detection_thread.is_alive()
    })

# Main function
if __name__ == '__main__':
    try:
        # Load model
        load_trained_model()
        
        # Start Flask server
        print("Starting server on http://127.0.0.1:2500")
        socketio.run(app, debug=True, host='0.0.0.0', port=2500)
    except Exception as e:
        print(f"Error: {e}")