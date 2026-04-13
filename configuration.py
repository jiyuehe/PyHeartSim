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
from pathlib import Path
script_dir = os.path.dirname(os.path.abspath(__file__)) # get the path of the current script
os.chdir(script_dir) # change the working directory
script_dir = Path(script_dir)

def directory_setup():
    # directory folder
    directory = {}
    directory['home'] = script_dir
    directory['mesh_database'] = script_dir / 'mesh_database'
    directory['result'] = script_dir.parent / '0_result'
    directory['data'] = script_dir.parent / '0_data'

    # create the folder if it does not exist
    directory['result'].mkdir(exist_ok=True)
    directory['data'].mkdir(exist_ok=True)

    return directory

