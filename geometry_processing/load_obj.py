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

import numpy as np

def execute(file_directory, mesh_name):
    file_path = file_directory / f"{mesh_name}.obj"

    vertex = []
    face = []
    with open(file_path, 'r', encoding='utf-8') as fid:
        for line in fid:
            if line.startswith('v '):
                vertex.append(np.fromstring(line[2:], sep=' '))
            elif line.startswith('f '):
                # subtract 1 to convert 1-based OBJ indices to 0-based python indices
                face.append(np.fromstring(line[2:], sep=' ', dtype=int) - 1)


    vertex =  np.asarray(vertex)
    face = np.asarray(face, dtype=int)

    return vertex, face
