import os
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__, static_folder='.', static_url_path='')

AUDIO_DIR = 'audio'

@app.route('/')
def index():
    return send_from_directory('.', 'admin.html')

@app.route('/api/folders', methods=['GET'])
def get_folders():
    folders = []
    if os.path.exists(AUDIO_DIR):
        folders = [d for d in os.listdir(AUDIO_DIR) if os.path.isdir(os.path.join(AUDIO_DIR, d))]
    return jsonify(folders)

@app.route('/api/folders', methods=['POST'])
def create_folder():
    data = request.json
    folder_name = data.get('name')
    
    if not folder_name:
        return jsonify({'error': 'Folder name is required'}), 400
    
    folder_path = os.path.join(AUDIO_DIR, folder_name)
    
    if not os.path.exists(AUDIO_DIR):
        os.makedirs(AUDIO_DIR)
    
    if os.path.exists(folder_path):
        return jsonify({'error': 'Folder already exists'}), 400
    
    os.makedirs(folder_path)
    return jsonify({'success': True, 'folder': folder_name})

@app.route('/api/folders/<folder>/files', methods=['GET'])
def get_files(folder):
    folder_path = os.path.join(AUDIO_DIR, folder)
    
    if not os.path.exists(folder_path):
        return jsonify([])
    
    files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
    return jsonify(files)

@app.route('/api/upload/recording', methods=['POST'])
def upload_recording():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    audio_file = request.files['audio']
    folder = request.form.get('folder', 'recordings')
    
    folder_path = os.path.join(AUDIO_DIR, folder)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    file_path = os.path.join(folder_path, audio_file.filename)
    audio_file.save(file_path)
    
    return jsonify({'success': True, 'file': audio_file.filename, 'folder': folder})

@app.route('/api/upload/files', methods=['POST'])
def upload_files():
    if 'files' not in request.files:
        return jsonify({'error': 'No files provided'}), 400
    
    files = request.files.getlist('files')
    folder = request.form.get('folder', 'uploads')
    
    folder_path = os.path.join(AUDIO_DIR, folder)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    uploaded_files = []
    
    for file in files:
        if file.filename.endswith('.wav'):
            file_path = os.path.join(folder_path, file.filename)
            file.save(file_path)
            uploaded_files.append(file.filename)
    
    return jsonify({'success': True, 'files': uploaded_files, 'folder': folder})

if __name__ == '__main__':
    app.run(debug=True)