import os
import json
import numpy as np
import librosa
import soundfile as sf
from flask import Flask, request, jsonify, render_template, send_file
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten, LSTM
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, Callback
from sklearn.model_selection import train_test_split
import time
import threading
import warnings
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('train_model')

# Update the Flask and SocketIO initialization
app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading', logger=True, engineio_logger=True)

AUDIO_DIR = 'audio'
MODEL_DIR = 'models'

# Initialize models directory
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# Global variables for tracking training status
training_job = None
stop_training = False

class WebSocketCallback(Callback):
    def __init__(self, total_epochs):
        super().__init__()
        self.total_epochs = total_epochs
        self.batch_logs = {}
        self.current_epoch = 0  # Track the current epoch
        
    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch  # Update the current epoch
        socketio.emit('training_update', {
            'type': 'progress',
            'epoch': epoch + 1,
            'totalEpochs': self.total_epochs,
            'batchProgress': 0
        }, namespace='/ws/training')
        
        socketio.emit('training_update', {
            'type': 'log',
            'message': f'Starting epoch {epoch + 1}/{self.total_epochs}',
            'level': 'info'
        }, namespace='/ws/training')
    
    def on_batch_end(self, batch, logs=None):
        global stop_training
        if stop_training:
            self.model.stop_training = True
            return
        
        self.batch_logs = logs or {}
        # Update batch progress less frequently to reduce WebSocket traffic
        if batch % 10 == 0:  # Only send updates every 10 batches
            socketio.emit('training_update', {
                'type': 'progress',
                'epoch': self.current_epoch + 1,  # Use our tracked current epoch
                'totalEpochs': self.total_epochs,
                'batchProgress': min(1.0, (batch + 1) / 100)  # Approximation
            }, namespace='/ws/training')
    
    def on_epoch_end(self, epoch, logs=None):
        global stop_training
        if stop_training:
            self.model.stop_training = True
            return
        
        # Send detailed metrics
        metrics = {
            'train_loss': float(logs.get('loss')),
            'train_acc': float(logs.get('accuracy')),
            'val_loss': float(logs.get('val_loss', 0)),
            'val_acc': float(logs.get('val_accuracy', 0))
        }
        
        socketio.emit('training_update', {
            'type': 'progress',
            'epoch': epoch + 1,  # epochs are 0-indexed, add 1 for display
            'totalEpochs': self.total_epochs,
            'batchProgress': 1.0,
            'metrics': metrics
        }, namespace='/ws/training')
        
        message = f'Epoch {epoch + 1}/{self.total_epochs} - ' + ', '.join([f'{k}: {v:.4f}' for k, v in metrics.items()])
        logger.info(message)
        
        socketio.emit('training_update', {
            'type': 'log',
            'message': message,
            'level': 'info'
        }, namespace='/ws/training')

@app.route('/')
def index():
    return app.send_static_file('train.html')

@app.route('/api/categories', methods=['GET'])
def get_categories():
    categories = []
    if os.path.exists(AUDIO_DIR):
        categories = [d for d in os.listdir(AUDIO_DIR) 
                     if os.path.isdir(os.path.join(AUDIO_DIR, d)) and 
                     any(f.endswith('.wav') for f in os.listdir(os.path.join(AUDIO_DIR, d)))]
    return jsonify(categories)

def load_audio(file_path, sr=None):
    """Custom audio loading function that attempts to use soundfile first,
    then falls back to librosa with warnings suppressed"""
    try:
        data, sample_rate = sf.read(file_path)
        # Convert to mono if stereo
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)
        if sr is not None and sr != sample_rate:
            # Resample if necessary
            data = librosa.resample(data, orig_sr=sample_rate, target_sr=sr)
        return data, sr or sample_rate
    except Exception:
        # If soundfile fails, fall back to librosa with warnings suppressed
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=FutureWarning)
            return librosa.load(file_path, sr=sr, mono=True)

def extract_features(file_path, feature_config):
    try:
        y, sr = load_audio(file_path, sr=None)
        y = librosa.util.normalize(y)
        
        window_size = feature_config['windowSize'] / 1000  # Convert ms to seconds
        hop_length = feature_config['hopLength'] / 1000  # Convert ms to seconds
        
        n_fft = int(window_size * sr)
        hop_length_samples = int(hop_length * sr)
        
        if feature_config['featureType'] == 'mfcc':
            features = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=feature_config['nMfcc'],
                                          n_fft=n_fft, hop_length=hop_length_samples)
        elif feature_config['featureType'] == 'mel':
            features = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, 
                                                   hop_length=hop_length_samples)
            features = librosa.power_to_db(features)
        else:  # stft
            features = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length_samples))
            features = librosa.amplitude_to_db(features)
        
        # Transpose to have time as the first dimension
        features = features.T
        
        # Ensure consistent length (trim or pad)
        target_length = 128  # Fixed length for all samples
        if features.shape[0] > target_length:
            features = features[:target_length, :]
        else:
            # Pad with zeros
            pad_width = target_length - features.shape[0]
            features = np.pad(features, ((0, pad_width), (0, 0)), mode='constant')
        
        return features
    except Exception as e:
        print(f"Error extracting features from {file_path}: {e}")
        return None

def prepare_dataset(categories, feature_config, train_split=0.8, use_augmentation=False):
    X = []
    y = []
    
    socketio.emit('training_update', {
        'type': 'log',
        'message': f"Preparing dataset for {len(categories)} categories...",
        'level': 'info'
    }, namespace='/ws/training')  # Add namespace here
    
    label_map = {category: i for i, category in enumerate(categories)}
    
    for category in categories:
        category_path = os.path.join(AUDIO_DIR, category)
        if not os.path.exists(category_path):
            continue
        
        socketio.emit('training_update', {
            'type': 'log',
            'message': f"Processing category: {category}",
            'level': 'info'
        }, namespace='/ws/training')  # Add namespace here
        
        files = [f for f in os.listdir(category_path) if f.endswith('.wav')]
        label = label_map[category]
        
        for file_name in files:
            file_path = os.path.join(category_path, file_name)
            features = extract_features(file_path, feature_config)
            
            if features is not None:
                X.append(features)
                y.append(label)
                
                # Simple augmentation: add small noise
                if use_augmentation:
                    noise_level = 0.005
                    noise = np.random.normal(0, noise_level, features.shape)
                    augmented_features = features + noise
                    X.append(augmented_features)
                    y.append(label)
    
    X = np.array(X)
    y = np.array(y)
    y_categorical = to_categorical(y, num_classes=len(categories))
    
    # Split dataset
    X_train, X_val, y_train, y_val = train_test_split(X, y_categorical, test_size=1-train_split, 
                                                     stratify=y, random_state=42)
    
    socketio.emit('training_update', {
        'type': 'log',
        'message': f"Dataset prepared: {X_train.shape[0]} training samples, {X_val.shape[0]} validation samples",
        'level': 'info'
    }, namespace='/ws/training')  # Add namespace here
    
    return X_train, X_val, y_train, y_val, len(categories)

def create_model(model_config, input_shape, num_classes):
    from tensorflow.keras.layers import Input
    
    # Define model based on selected type
    if model_config['modelType'] == 'cnn':
        inputs = Input(shape=input_shape)
        x = Conv1D(32, 3, activation='relu')(inputs)
        x = MaxPooling1D(pool_size=3)(x)
        
        for _ in range(model_config['numLayers'] - 1):
            x = Conv1D(64, 3, activation='relu')(x)
            x = MaxPooling1D(pool_size=3)(x)
        
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        outputs = Dense(num_classes, activation='softmax')(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    elif model_config['modelType'] == 'lstm':
        inputs = Input(shape=input_shape)
        x = LSTM(64, return_sequences=True)(inputs)
        
        for i in range(model_config['numLayers'] - 1):
            return_seq = (i < model_config['numLayers'] - 2)
            x = LSTM(64, return_sequences=return_seq)(x)
        
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        outputs = Dense(num_classes, activation='softmax')(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    elif model_config['modelType'] == 'cnn-lstm':
        inputs = Input(shape=input_shape)
        x = Conv1D(32, 3, activation='relu')(inputs)
        x = MaxPooling1D(pool_size=2)(x)
        x = Conv1D(64, 3, activation='relu')(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = LSTM(64)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        outputs = Dense(num_classes, activation='softmax')(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    else:  # Default to a simpler model (transformer-inspired)
        inputs = Input(shape=input_shape)
        x = Conv1D(32, 3, activation='relu')(inputs)
        x = MaxPooling1D(pool_size=2)(x)
        x = Conv1D(64, 3, activation='relu')(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Flatten()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        outputs = Dense(num_classes, activation='softmax')(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # Compile the model
    model.compile(
        loss='categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=model_config['learningRate']),
        metrics=['accuracy']
    )
    
    # Log model summary with namespace
    model.summary(print_fn=lambda x: socketio.emit('training_update', {
        'type': 'log',
        'message': x,
        'level': 'info'
    }, namespace='/ws/training'))
    
    return model

def train_model_task(config):
    global stop_training
    stop_training = False
    
    try:
        socketio.emit('training_update', {
            'type': 'log',
            'message': "Starting model training process...",
            'level': 'info'
        }, namespace='/ws/training')  # Add namespace here
        
        # Prepare dataset
        X_train, X_val, y_train, y_val, num_classes = prepare_dataset(
            config['categories'],
            config['featureConfig'],
            config['trainSplit'],
            config['modelConfig']['useAugmentation']
        )
        
        # Create model
        input_shape = (X_train.shape[1], X_train.shape[2])
        model = create_model(config['modelConfig'], input_shape, num_classes)
        
        # Setup callbacks
        callbacks = [WebSocketCallback(config['modelConfig']['epochs'])]
        
        if config['modelConfig']['useEarlyStopping']:
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )
            callbacks.append(early_stopping)
        
        # Train the model
        socketio.emit('training_update', {
            'type': 'log',
            'message': "Starting model training...",
            'level': 'info'
        }, namespace='/ws/training')  # Add namespace here
        
        history = model.fit(
            X_train, y_train,
            epochs=config['modelConfig']['epochs'],
            batch_size=config['modelConfig']['batchSize'],
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=0
        )
        
        if not stop_training:
            # Save the model
            if not os.path.exists(MODEL_DIR):
                os.makedirs(MODEL_DIR)
                
            model_path = os.path.join(MODEL_DIR, 'audio_event_detector.h5')
            model.save(model_path)
            
            # Save the category mapping
            category_mapping = {i: category for i, category in enumerate(config['categories'])}
            with open(os.path.join(MODEL_DIR, 'categories.json'), 'w') as f:
                json.dump(category_mapping, f)
            
            # Get final metrics
            final_metrics = {
                'train_loss': float(history.history['loss'][-1]),
                'train_acc': float(history.history['accuracy'][-1]),
                'val_loss': float(history.history['val_loss'][-1]),
                'val_acc': float(history.history['val_accuracy'][-1])
            }
            
            socketio.emit('training_update', {
                'type': 'complete',
                'results': final_metrics
            }, namespace='/ws/training')  # Add namespace here
            
            socketio.emit('training_update', {
                'type': 'log',
                'message': f"Training completed successfully! Model saved to {model_path}",
                'level': 'success'
            }, namespace='/ws/training')  # Add namespace here
        else:
            socketio.emit('training_update', {
                'type': 'log',
                'message': "Training was stopped by the user",
                'level': 'warning'
            }, namespace='/ws/training')  # Add namespace here
    
    except Exception as e:
        socketio.emit('training_update', {
            'type': 'error',
            'message': str(e)
        }, namespace='/ws/training')  # Add namespace here
        socketio.emit('training_update', {
            'type': 'log',
            'message': f"Error during training: {str(e)}",
            'level': 'error'
        }, namespace='/ws/training')  # Add namespace here

@app.route('/api/train', methods=['POST'])
def train_model():
    global training_job
    
    if training_job is not None and training_job.is_alive():
        return jsonify({'success': False, 'error': 'Training is already in progress'})
    
    config = request.json
    training_job = threading.Thread(target=train_model_task, args=(config,))
    training_job.start()
    
    return jsonify({'success': True, 'jobId': str(int(time.time()))})

@app.route('/api/train/stop', methods=['POST'])
def stop_model_training():
    global stop_training
    stop_training = True
    return jsonify({'success': True})

@app.route('/api/model/download')
def download_model():
    model_path = os.path.join(MODEL_DIR, 'audio_event_detector.h5')
    if os.path.exists(model_path):
        return send_file(model_path, as_attachment=True)
    else:
        return jsonify({'error': 'Model file not found'}), 404

# Update the handle_connect function
@socketio.on('connect', namespace='/ws/training')
def handle_connect():
    logger.info("Client connected")
    emit('response', {'data': 'Connected to server'})

# Update the main execution
if __name__ == '__main__':
    logger.info(f"Starting server on port 2300")
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True, host='0.0.0.0', port=2300)