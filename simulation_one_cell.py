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

# NOTE:
# This script is useful for comparing heart models.

#%%
import matplotlib.pyplot as plt # for plotting
import configuration
import utility

#%%
directory = configuration.directory_setup()

# simulation parameters
dt = 0.05 # ms, time step
t_final = 1000 # ms, compute simulation of this much ms

# pacing parameters
pacing_start_time = 100 # ms, time to start pacing
pacing_cycle_length = 300 # ms, time between consecutive pacings
pacing_duration = 2 # ms, duration of each pacing
J_stim_value = 1 # pacing strength

# compute simulation of the Mitchell-Schaeffer model
print('compute simulation of the Mitchell-Schaeffer model')
heart_model_flag = 1 # 1 for Mitchell-Schaeffer, 2 for Aliev-Panfilov
pacing_signal, time, action_potential, gate_variable = utility.one_cell.solve_differential_equation(heart_model_flag, dt, t_final, pacing_start_time, pacing_cycle_length, pacing_duration, J_stim_value)
Mitchell_Schaeffer = {}
Mitchell_Schaeffer['time'] = time
Mitchell_Schaeffer['action_potential'] = action_potential
Mitchell_Schaeffer['gate_variable'] = gate_variable
Mitchell_Schaeffer['pacing_signal'] = pacing_signal

# compute simulation of the Aliev-Panfilov model
print('compute simulation of the Aliev-Panfilov model')
heart_model_flag = 2 # 1 for Mitchell-Schaeffer, 2 for Aliev-Panfilov
pacing_signal, time, action_potential, gate_variable = utility.one_cell.solve_differential_equation(heart_model_flag, dt, t_final, pacing_start_time, pacing_cycle_length, pacing_duration, J_stim_value)
Aliev_Panfilov = {}
Aliev_Panfilov['time'] = time
Aliev_Panfilov['action_potential'] = action_potential
Aliev_Panfilov['gate_variable'] = gate_variable
Aliev_Panfilov['pacing_signal'] = pacing_signal

#%%
# plot a comparison of the simulation results of the two models
models = [
    ("Mitchell-Schaeffer", Mitchell_Schaeffer, 0),
    ("Aliev-Panfilov", Aliev_Panfilov, 1),
]

fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex=True)

for model_name, model_data, column_id in models:
    # action potential
    axes[0, column_id].plot(model_data['time'], model_data['action_potential'], 'b')
    axes[0, column_id].grid(True)
    axes[0, column_id].set_ylabel("u (action potential)")
    axes[0, column_id].set_title(f"{model_name} model")

    # gate variable
    axes[1, column_id].plot(model_data['time'], model_data['gate_variable'], 'b')
    axes[1, column_id].set_ylabel('h (gate variable)')
    axes[1, column_id].grid(True)

    # pacing signal
    axes[2, column_id].plot(model_data['time'], model_data['pacing_signal'], 'b')
    axes[2, column_id].set_ylabel("Pacing signal")
    axes[2, column_id].set_xlabel(f"Time (ms)")
    axes[2, column_id].grid(True)

plt.tight_layout()
plt.savefig(directory['result'] / 'single_cell_simulation_comparison.png', dpi=300)
plt.close()

print('done')
