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
from flask import Flask, jsonify, render_template, send_from_directory, request
from flask_socketio import SocketIO

# Add these imports to the top of your detection.py
import csv
import requests
import smtplib
import os.path
import uuid
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.audio import MIMEAudio
import soundfile as sf
from twilio.rest import Client
from pydub import AudioSegment
from datetime import datetime

# Configuration
MODEL_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_DIR, 'audio_event_detector.h5')
CATEGORIES_PATH = os.path.join(MODEL_DIR, 'categories.json')

# Audio settings
SAMPLE_RATE = 44100
WINDOW_SIZE = 1024  # ~23ms at 44.1kHz
HOP_LENGTH = 512   # ~12ms at 44.1kHz
BUFFER_DURATION = 5  # seconds for each analysis window
DETECTION_THRESHOLD = 0.70  # Temporary lower threshold for testing
AMBULANCE_THRESHOLD = 0.80  # Temporary lower threshold for testing
MIN_CONSISTENT_PREDICTIONS = 2  # Require fewer consistent detections for testing
PREDICTION_HISTORY_SIZE = 6  # Number of predictions to keep in history
COOLDOWN_PERIOD = 2.0  # Seconds between detections
BACKGROUND_NOISE_SAMPLES = []  # To store background noise profiles
CLASS_CONFUSION_MATRIX = {}  # Track which classes are confused with each other

# Add these configuration variables
# WhatsApp configuration (using Twilio)
TWILIO_ACCOUNT_SID = "YOUR_TWILIO_ACCOUNT_SID"  # Replace with your Twilio Account SID
TWILIO_AUTH_TOKEN = "YOUR_TWILIO_AUTH_TOKEN"  # Replace with your Twilio Auth Token
TWILIO_PHONE_NUMBER = "YOUR_TWILIO_PHONE_NUMBER"  # Replace with your Twilio phone number

# Email configuration
GMAIL_ADDRESS = "b22ds056@mace.ac.in"  # Replace with your Gmail address
GMAIL_APP_PASSWORD = "your-app-password"  # Replace with your Gmail App Password (not your regular Gmail password)

# SMS configuration (using Twilio)
SMS_FROM_NUMBER = "YOUR_TWILIO_PHONE_NUMBER"  # Same as TWILIO_PHONE_NUMBER usually

# Notification settings
NOTIFICATION_COOLDOWN = 300  # Seconds between notifications (5 minutes)
THREAT_COUNT_THRESHOLD = 1  # Send notification after this many detections of the same threat
EXCLUDED_EVENTS = ['env_audio']  # Events that don't trigger notifications
NOTIFICATION_DIR = "notifications"  # Directory to store notification audio files
os.makedirs(NOTIFICATION_DIR, exist_ok=True)

# Initialize Flask app
app = Flask(__name__, static_folder='.', static_url_path='')
socketio = SocketIO(app, cors_allowed_origins="*", logger=True, engineio_logger=True, async_mode='threading')

# Global variables
audio_buffer = queue.Queue()
stop_recording = threading.Event()
detection_thread = None
model = None
categories = {}
audio_stream = None  # Keep a reference to the audio stream
recent_predictions = []
class_thresholds = {}  # Will store custom thresholds for each class
DEFAULT_THRESHOLD = 0.90  # Base threshold for all classes

# Add this to your global variables
threat_detection_counts = {}  # Track number of detections per threat
last_notification_time = {}  # Track when last notification was sent for each threat
contacts = []  # Will store contact information

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
    last_detected_class = None
    
    # Initialize recent predictions as an empty list
    recent_predictions = []
    
    print("Starting audio event detection...")
    
    # Create a reverse mapping from event name to class index
    event_to_class = {v: int(k) for k, v in categories.items()}
    ambulance_class = event_to_class.get('ambulance', -1)
    
    while not stop_recording.is_set():
        try:
            # Get data from the buffer queue
            try:
                # Non-blocking get with timeout
                new_data = audio_buffer.get(timeout=0.5)
                
                # Add to our running buffer
                data_length = min(len(new_data), buffer_samples - samples_collected)
                audio_data[samples_collected:samples_collected + data_length] = new_data[:data_length]
                samples_collected += data_length
                
                # If we have enough samples, process them
                if samples_collected >= buffer_samples:
                    current_time = time.time()
                    
                    # DEBUG: Log audio signal statistics
                    rms = np.sqrt(np.mean(audio_data**2))
                    print(f"Audio buffer filled: RMS={rms:.4f}, Max={audio_data.max():.4f}, Min={audio_data.min():.4f}")
                    
                    # Extract features
                    features = extract_features(audio_data)
                    
                    # Reshape for model input (add batch dimension)
                    features = np.expand_dims(features, axis=0)
                    
                    # Make prediction
                    predictions = model.predict(features, verbose=0)[0]
                    max_prob = np.max(predictions)
                    predicted_class = np.argmax(predictions)
                    predicted_class_name = categories.get(str(predicted_class), f"Unknown-{predicted_class}")
                    
                    # DEBUG: Print all predictions above a minimal threshold
                    print(f"Top predictions:")
                    sorted_indices = np.argsort(predictions)[::-1]
                    for idx in sorted_indices[:3]:  # Show top 3 predictions
                        class_name = categories.get(str(idx), f"Unknown-{idx}")
                        print(f"  {class_name}: {predictions[idx]:.4f}")
                    
                    # Add to prediction history with timestamp
                    recent_predictions.append({
                        'class': predicted_class, 
                        'prob': max_prob,
                        'time': current_time,
                        'features': features.mean(axis=1).flatten()  # Store average feature vector
                    })
                    
                    # Keep only recent predictions
                    recent_predictions = [p for p in recent_predictions 
                                         if current_time - p['time'] < 10.0]  # Last 10 seconds
                    
                    # Only process if we're not in cooldown period
                    if current_time - last_detection_time >= COOLDOWN_PERIOD:
                        # Count predictions by class and apply filters
                        class_counts = {}
                        class_probs = {}
                        
                        # Get the most recent consistent predictions (last 2 seconds)
                        recent_consistent = [p for p in recent_predictions 
                                           if current_time - p['time'] < 2.0]
                        
                        for pred in recent_consistent:
                            cls = pred['class']
                            prob = pred['prob']
                            
                            # Use the correct threshold based on the class
                            threshold = AMBULANCE_THRESHOLD if cls == ambulance_class else DETECTION_THRESHOLD
                            
                            if prob >= threshold:
                                class_counts[cls] = class_counts.get(cls, 0) + 1
                                class_probs.setdefault(cls, []).append(prob)
                        
                        # Additional pattern analysis for ambulance class
                        if ambulance_class in class_counts and class_counts[ambulance_class] >= MIN_CONSISTENT_PREDICTIONS:
                            # Calculate coherence of predictions
                            feature_vectors = np.array([p['features'] for p in recent_consistent 
                                                     if p['class'] == ambulance_class])
                            
                            if len(feature_vectors) >= 2:
                                # Calculate average pairwise correlation
                                correlations = []
                                for i in range(len(feature_vectors)):
                                    for j in range(i+1, len(feature_vectors)):
                                        corr = np.corrcoef(feature_vectors[i], feature_vectors[j])[0, 1]
                                        correlations.append(corr)
                                
                                avg_correlation = np.mean(correlations) if correlations else 0
                                
                                # If coherence is low, it might be a false positive
                                if avg_correlation < 0.7:  # Ambulance sounds should be coherent
                                    print(f"Rejecting ambulance detection due to low coherence: {avg_correlation:.2f}")
                                    del class_counts[ambulance_class]
                                    del class_probs[ambulance_class]
                        
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
                            avg_confidence = np.mean(class_probs[most_consistent_class])
                            
                            # Skip consecutive detections of the same class
                            if most_consistent_class != last_detected_class or current_time - last_detection_time >= 5.0:
                                print(f"Detected: {event_name} (confidence: {avg_confidence:.2f}, " +
                                      f"consistency: {most_consistent_count}/{len(recent_consistent)})")
                                
                                # Emit to web clients
                                debug_emit('detection', {
                                    'event': event_name,
                                    'confidence': float(avg_confidence),
                                    'consistency': most_consistent_count,
                                    'timestamp': current_time
                                })
                                
                                # Call threat detection handler
                                handle_threat_detection(
                                    event_name=event_name,
                                    confidence=avg_confidence,
                                    audio_data=audio_data.copy()  # Send a copy of the audio data
                                )
                                
                                last_detection_time = current_time
                                last_detected_class = most_consistent_class
                    
                    # Slide the buffer (keep the last half for overlap)
                    half_buffer = buffer_samples // 2
                    audio_data[:half_buffer] = audio_data[half_buffer:]
                    samples_collected = half_buffer
            
            except queue.Empty:
                # Queue is empty, continue waiting
                continue
                
        except Exception as e:
            print(f"Error in detection loop: {e}")
            time.sleep(0.1)  # Prevent tight error loops
    
    print("Detection thread stopped")

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
        
        print("Audio detection started")
        return True
    except Exception as e:
        print(f"Error starting detection: {e}")
        return False

def stop_detection():
    """Stop the audio stream and detection thread"""
    global stop_recording, audio_stream
    
    try:
        # Set the stop flag to stop the detection thread
        stop_recording.set()
        
        # Stop the audio stream
        if audio_stream is not None:
            audio_stream.stop()
            audio_stream.close()
            audio_stream = None
        
        # Wait for thread to finish
        if detection_thread and detection_thread.is_alive():
            detection_thread.join(timeout=2.0)
        
        # Clear the audio buffer
        while not audio_buffer.empty():
            try:
                audio_buffer.get_nowait()
            except queue.Empty:
                break
        
        print("Audio detection stopped")
        return True
    except Exception as e:
        print(f"Error stopping detection: {e}")
        return False

def calibrate_background_noise(duration=10.0):
    """Record background noise and calibrate the system"""
    global audio_stream, audio_buffer, model, BACKGROUND_NOISE_SAMPLES, class_thresholds
    
    print(f"Calibrating background noise for {duration} seconds...")
    print("Please remain quiet during calibration...")
    
    # Clear any existing audio in the buffer
    while not audio_buffer.empty():
        audio_buffer.get()
    
    # Start audio stream if not already started
    temp_stream = None
    if audio_stream is None:
        temp_stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            callback=audio_callback
        )
        temp_stream.start()
    
    # Collect background noise samples
    buffer_samples = int(SAMPLE_RATE * duration)
    noise_samples = np.zeros(buffer_samples)
    samples_collected = 0
    
    start_time = time.time()
    while samples_collected < buffer_samples and (time.time() - start_time) < duration + 1:
        if not audio_buffer.empty():
            new_data = audio_buffer.get()
            data_length = min(len(new_data), buffer_samples - samples_collected)
            noise_samples[samples_collected:samples_collected + data_length] = new_data[:data_length]
            samples_collected += data_length
        else:
            time.sleep(0.01)
    
    # Stop the temporary stream if we created one
    if temp_stream is not None:
        temp_stream.stop()
        temp_stream.close()
    
    # Clear the buffer again
    while not audio_buffer.empty():
        audio_buffer.get()
    
    print(f"Collected {samples_collected/SAMPLE_RATE:.1f} seconds of background noise")
    
    # Process the background noise to understand its characteristics
    if samples_collected > 0:
        # Divide into 1-second segments for multiple samples
        segment_length = SAMPLE_RATE
        num_segments = samples_collected // segment_length
        
        class_activations = {}
        
        print(f"Analyzing {num_segments} segments of background noise...")
        
        for i in range(num_segments):
            segment = noise_samples[i*segment_length:(i+1)*segment_length]
            
            # Extract features from the noise segment
            features = extract_features(segment)
            
            # Store the noise profile
            BACKGROUND_NOISE_SAMPLES.append(features)
            
            # Reshape for model input (add batch dimension)
            features = np.expand_dims(features, axis=0)
            
            # Check what the model predicts for background noise
            if model is not None:
                predictions = model.predict(features, verbose=0)[0]
                
                # Track which classes are activated by background noise
                for j, prob in enumerate(predictions):
                    class_activations.setdefault(j, []).append(prob)
        
        # Set class-specific thresholds based on background noise
        for cls, activations in class_activations.items():
            mean_activation = np.mean(activations)
            std_activation = np.std(activations)
            
            event_name = categories.get(str(cls), f"Unknown-{cls}")
            
            # Set threshold higher for classes activated by background noise
            if mean_activation > 0.1:
                # Dynamic threshold: base + (mean + 2*std)
                class_thresholds[cls] = min(0.98, DETECTION_THRESHOLD + mean_activation + 2*std_activation)
                print(f"Class {event_name} activated by background ({mean_activation:.3f}±{std_activation:.3f})")
                print(f"Setting higher threshold for {event_name}: {class_thresholds[cls]:.3f}")
            
        # Count which environment classes are most frequently predicted
        predictions = [np.argmax(model.predict(np.expand_dims(features, axis=0), verbose=0)[0]) 
                     for features in BACKGROUND_NOISE_SAMPLES]
        
        classes, counts = np.unique(predictions, return_counts=True)
        for i, cls in enumerate(classes):
            event_name = categories.get(str(cls), f"Unknown-{cls}")
            print(f"Background noise classified as {event_name}: {counts[i]/len(predictions)*100:.1f}%")
            
            # If this class appears frequently in background, increase its threshold dramatically
            if counts[i]/len(predictions) > 0.3:  # If >30% of background classified as this
                class_thresholds[cls] = 0.98  # Very high threshold
                print(f"WARNING: {event_name} appears frequently in background noise")
                print(f"Setting very high threshold for {event_name}: {class_thresholds[cls]:.3f}")
        
        print("Background noise calibration complete.")
        print("Class-specific detection thresholds:")
        for cls, threshold in class_thresholds.items():
            event_name = categories.get(str(cls), f"Unknown-{cls}")
            print(f"  {event_name}: {threshold:.3f}")
    
    return BACKGROUND_NOISE_SAMPLES

def analyze_model_confusion():
    """Analyze potential model confusion between classes"""
    global model, categories, BACKGROUND_NOISE_SAMPLES
    
    if not BACKGROUND_NOISE_SAMPLES or model is None:
        print("No background samples available for analysis")
        return
    
    print("\nAnalyzing potential model confusion...")
    
    # Get predictions for all background samples
    all_preds = []
    for features in BACKGROUND_NOISE_SAMPLES:
        features = np.expand_dims(features, axis=0)
        pred = model.predict(features, verbose=0)[0]
        all_preds.append(pred)
    
    all_preds = np.array(all_preds)
    
    # For each class, see if it's consistently confused with background
    avg_activations = np.mean(all_preds, axis=0)
    
    print("\nClass activations on background noise:")
    for i, avg_act in enumerate(avg_activations):
        if avg_act > 0.1:  # Only show classes with significant activations
            event_name = categories.get(str(i), f"Unknown-{i}")
            print(f"  {event_name}: {avg_act:.4f}")
    
    # Check if there's an environment audio class
    env_audio_idx = -1
    for idx, name in categories.items():
        if name.lower() in ['env_audio', 'environment', 'background', 'ambient']:
            env_audio_idx = int(idx)
            break
    
    if env_audio_idx >= 0:
        print(f"\nFound environment audio class (index {env_audio_idx})")
        
        # Check if the environment audio class is properly classified
        env_audio_detections = 0
        for pred in all_preds:
            if np.argmax(pred) == env_audio_idx:
                env_audio_detections += 1
        
        print(f"Background correctly classified as environment: {env_audio_detections}/{len(all_preds)} " +
              f"({env_audio_detections/len(all_preds)*100:.1f}%)")
        
        if env_audio_detections / len(all_preds) < 0.5:
            print("WARNING: Model doesn't consistently recognize background as environment audio")
            print("Consider retraining the model with more environment samples")
    
    # Optional: Exclude specific problematic classes from detection
    print("\nAutomatically excluding problematic classes...")
    
    excluded_classes = []
    # Automatically exclude classes that are frequently activated by background
    for i, avg_act in enumerate(avg_activations):
        if avg_act > 0.2:  # Threshold for automatic exclusion
            excluded_classes.append(i)
            event_name = categories.get(str(i), f"Unknown-{i}")
            print(f"Excluding class {event_name} (activation: {avg_act:.4f})")
    
    # Set extremely high thresholds for excluded classes
    for cls in excluded_classes:
        if cls in range(len(categories)):
            class_thresholds[cls] = 0.999  # Effectively disable detection
            event_name = categories.get(str(cls), f"Unknown-{cls}")
            print(f"Excluded class {event_name} (index {cls}) from detection")

def cleanup():
    """Clean up resources when the application shuts down"""
    print("Cleaning up resources...")
    global audio_stream
    
    # Stop detection if it's running
    stop_detection()
    
    # Make sure audio stream is closed
    if (audio_stream):
        try:
            audio_stream.stop()
            audio_stream.close()
        except Exception as e:
            print(f"Error closing audio stream: {e}")
    
    # Clear any remaining items in the queue
    while not audio_buffer.empty():
        try:
            audio_buffer.get_nowait()
        except queue.Empty:
            break

# Add this debug function
def debug_emit(event, data):
    """Wrapper for socketio.emit that adds debugging"""
    print(f"EMITTING {event}: {data}")
    socketio.emit(event, data)

# Add this function
def send_heartbeat():
    """Send heartbeat to clients"""
    while True:
        socketio.sleep(5)
        socketio.emit('heartbeat', {'time': time.time()})

# Contact management
def load_contacts():
    """Load contacts from CSV file"""
    contacts = []
    try:
        with open('contact.csv', 'r') as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                contacts.append({
                    'name': row.get('NAME', ''),
                    'email': row.get('EMAIL', ''),
                    'phone': row.get('PHONE', '')
                })
        print(f"Loaded {len(contacts)} contacts")
        return contacts
    except Exception as e:
        print(f"Error loading contacts: {e}")
        return []

# Notification functions
def send_whatsapp_message(to_number, message):
    """Send WhatsApp message using Twilio"""
    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        message = client.messages.create(
            body=message,
            from_=f"whatsapp:{TWILIO_PHONE_NUMBER}",
            to=f"whatsapp:{to_number}"
        )
        print(f"WhatsApp message sent to {to_number}: {message.sid}")
        return True
    except Exception as e:
        print(f"Error sending WhatsApp message: {e}")
        return False

def send_email(to_email, subject, body, audio_file=None):
    """Send email via Gmail"""
    try:
        msg = MIMEMultipart()
        msg['From'] = GMAIL_ADDRESS
        msg['To'] = to_email
        msg['Subject'] = subject
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Attach audio file if provided
        if audio_file and os.path.exists(audio_file):
            with open(audio_file, 'rb') as f:
                audio_attachment = MIMEAudio(f.read(), _subtype="wav")
                audio_attachment.add_header('Content-Disposition', 'attachment', filename=os.path.basename(audio_file))
                msg.attach(audio_attachment)
        
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
        text = msg.as_string()
        server.sendmail(GMAIL_ADDRESS, to_email, text)
        server.quit()
        
        print(f"Email sent to {to_email}")
        return True
    except Exception as e:
        print(f"Error sending email: {e}")
        return False

def send_sms(to_number, message):
    """Send SMS using Twilio"""
    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        message = client.messages.create(
            body=message,
            from_=SMS_FROM_NUMBER,
            to=to_number
        )
        print(f"SMS sent to {to_number}: {message.sid}")
        return True
    except Exception as e:
        print(f"Error sending SMS: {e}")
        return False

def save_audio_sample(audio_data, event_name):
    """Save a portion of the audio data for notification attachments"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{NOTIFICATION_DIR}/threat_{event_name}_{timestamp}.wav"
    
    try:
        # Save the audio data as a WAV file
        sf.write(filename, audio_data, SAMPLE_RATE)
        print(f"Audio sample saved to {filename}")
        return filename
    except Exception as e:
        print(f"Error saving audio sample: {e}")
        return None

def handle_threat_detection(event_name, confidence, audio_data):
    """Handle detection of a potential threat"""
    global threat_detection_counts, last_notification_time, contacts
    
    # Skip if this is not a threat (in excluded events list)
    if event_name.lower() in [e.lower() for e in EXCLUDED_EVENTS]:
        return
    
    current_time = time.time()
    
    # Initialize counters if this is a new threat
    if event_name not in threat_detection_counts:
        threat_detection_counts[event_name] = 0
        last_notification_time[event_name] = 0
    
    # Increment detection count
    threat_detection_counts[event_name] += 1
    
    # Check if we should send notifications
    if (threat_detection_counts[event_name] >= THREAT_COUNT_THRESHOLD and 
            current_time - last_notification_time[event_name] > NOTIFICATION_COOLDOWN):
        
        print(f"ALERT: {event_name} detected {threat_detection_counts[event_name]} times! Sending notifications...")
        
        # Save audio sample
        audio_file = save_audio_sample(audio_data, event_name)
        
        # Prepare notification message
        notification_message = (
            f"⚠️ ALERT: {event_name} detected!\n\n"
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Confidence: {confidence:.2%}\n"
            f"Location: Your home monitoring system"
        )
        
        # Load contacts if not already loaded
        if not contacts:
            contacts = load_contacts()
        
        # Send notifications to all contacts
        for contact in contacts:
            # Send WhatsApp message
            if contact['phone']:
                send_whatsapp_message(contact['phone'], notification_message)
                
            # Send email
            if contact['email']:
                email_subject = f"⚠️ ALERT: {event_name} Detected"
                send_email(contact['email'], email_subject, notification_message, audio_file)
                
            # Send SMS
            if contact['phone']:
                # SMS version is shorter due to length limitations
                sms_message = f"ALERT: {event_name} detected at {datetime.now().strftime('%H:%M:%S')}. Check email for details."
                send_sms(contact['phone'], sms_message)
        
        # Update the last notification time
        last_notification_time[event_name] = current_time
        
        # Reset the counter
        threat_detection_counts[event_name] = 0

# Register cleanup handlers
atexit.register(cleanup)
signal.signal(signal.SIGINT, lambda sig, frame: cleanup())
signal.signal(signal.SIGTERM, lambda sig, frame: cleanup())

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
    print('Client connected with ID:', request.sid)
    socketio.emit('status', {
        'model_loaded': model is not None,
        'is_detecting': detection_thread is not None and detection_thread.is_alive(),
        'connection_id': request.sid
    }, to=request.sid)

@socketio.on('client_test')
def handle_client_test(data):
    print(f"Received client_test: {data}")
    socketio.emit('server_test', {'response': 'Server received your message', 'timestamp': time.time()})

# Add this after your SocketIO events
@app.errorhandler(Exception)
def handle_error(e):
    print(f"Flask error: {str(e)}")
    import traceback
    traceback.print_exc()
    return jsonify(error=str(e)), 500

# Add this route at the end of your Flask routes section
@app.route('/test_detection', methods=['GET'])
def test_detection():
    """Force a test detection event for debugging"""
    print("TEST DETECTION endpoint called")
    current_time = time.time()
    
    test_data = {
        'event': "test_event",
        'confidence': 0.95,
        'consistency': 5,
        'timestamp': current_time
    }
    
    print(f"Emitting test detection: {test_data}")
    socketio.emit('detection', test_data)
    
    # Try with namespace too as a backup
    socketio.emit('detection', test_data, namespace='/')
    
    return jsonify({
        'success': True, 
        'message': 'Test detection event emitted',
        'timestamp': current_time
    })

# Main function
if __name__ == '__main__':
    try:
        # Load model
        load_trained_model()
        
        # Initialize class-specific thresholds
        class_thresholds = {int(idx): DETECTION_THRESHOLD for idx in categories}
        
        # Calibrate with background noise
        bg_noise = calibrate_background_noise(10.0)
        
        # Analyze potential confusion in the model
        analyze_model_confusion()
        
        # Load contacts
        contacts = load_contacts()  # Remove the global declaration here
        print(f"Loaded {len(contacts)} contacts for notifications")
        
        # Start Flask server
        print("Starting server on http://127.0.0.1:2500")
        socketio.run(app, debug=True, host='0.0.0.0', port=2500, allow_unsafe_werkzeug=True)
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Make sure we clean up on exit
        cleanup()