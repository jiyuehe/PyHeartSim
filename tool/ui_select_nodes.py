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

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import threading
import webbrowser
import numpy as np
from flask import Flask, render_template, jsonify, request
import configuration

app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__)))

# load geometry data
directory = configuration.directory_setup()
directory['result'] = directory['home'] / 'result'
directory['result'].mkdir(exist_ok=True)

name_prefix = configuration.mesh_name(0)

file_path = directory['data'] / f'{name_prefix}_clinical_data.npz'
data = np.load(file_path, allow_pickle=False)
geometry_data = {k: data[k] for k in data.files}
node = geometry_data['voxel3mm']

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
        'flags': node_flag.tolist(),
        'count': len(node)
    })

@app.route('/api/save', methods=['POST'])
def save_flags():
    global node_flag
    data = request.get_json()
    node_flag = np.array(data['flags'], dtype=int)
    np.save(flag_file, node_flag)

    s1 = np.where(node_flag == 1)[0]
    s2 = np.where(node_flag == 2)[0]
    print(f"Saved node flags to '{name_prefix}_node_flag.npy'")
    print(f's1_pacing_node_id ({len(s1)} nodes):')
    print(','.join(map(str, s1)))
    print(f's2_pacing_node_id ({len(s2)} nodes):')
    print(','.join(map(str, s2)))

    return jsonify({
        'status': 'saved',
        's1_count': len(s1),
        's2_count': len(s2)
    })

if __name__ == '__main__':
    port = 5000
    url = f'http://127.0.0.1:{port}'
    print(f'Starting 3D Node Selection Tool at {url}')
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    app.run(host='127.0.0.1', port=port, debug=False)
