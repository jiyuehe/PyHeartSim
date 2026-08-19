# Copyright 2026 Jiyue He
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#%%
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import threading
import webbrowser
import time
import subprocess
import numpy as np
from flask import Flask, render_template, send_from_directory, jsonify, request
import configuration

#%%
_tool_dir = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder=_tool_dir)

@app.route('/ui_select_nodes.css')
def serve_css():
    return send_from_directory(_tool_dir, 'ui_select_nodes.css')

# load geometry data
directory = configuration.directory_setup()
directory['result'] = directory['mesh_obj']

name_prefixs = configuration.mesh_name()
name_prefix = name_prefixs[0]

file_path = directory['data'] / f'{name_prefix}_mesh.npz'
data = np.load(file_path, allow_pickle=False)
geometry_data = {k: data[k] for k in data.files}

node = geometry_data['vertex'] # triangular mesh vertices xyz coordinates
face = geometry_data['face'] # triangular mesh faces vertex indices

flag_file = directory['result'] / f'{name_prefix}_node_flag.npy'
if os.path.exists(flag_file):
    node_flag = np.load(flag_file).copy()
else:
    node_flag = np.zeros(len(node), dtype=int)

@app.route('/')
def index():
    return render_template('ui_select_nodes.html')

@app.route('/api/nodes')
def get_nodes():
    return jsonify({
        'positions': node.flatten().tolist(),
        'faces': face.flatten().tolist(),
        'face_count': len(face),
        'flags': node_flag.tolist(),
        'count': len(node)
    })

@app.route('/api/save', methods=['POST'])
def save_flags():
    global node_flag
    data = request.get_json()
    node_flag = np.array(data['flags'], dtype=int)
    np.save(flag_file, node_flag)

    return jsonify({'status': 'saved'})

if __name__ == '__main__':
    server_port = 5001
    
    # stop any stale server that is already listening on Flask's port.
    stopped_port = subprocess.run(
        ['fuser', '-k', '-TERM', f'{server_port}/tcp'],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0
    if stopped_port:
        time.sleep(0.5)

    # open the patient data observer user interface
    threading.Timer(1.0, webbrowser.open, args=[f'http://127.0.0.1:{server_port}']).start() # runs webbrowser.open on a background thread after a 1-second delay, while the main thread proceeds to start Flask. The 1-second delay gives Flask time to start up before the browser tries to connect
    app.run(debug=False, port=server_port, host='0.0.0.0')
    