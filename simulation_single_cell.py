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
import matplotlib.pyplot as plt
import numpy as np
import configuration

#%%
directory = configuration.directory_setup()

heart_model_flag = 2 # 1: Mitchell-Schaeffer. 2: Aliev-Panfilov

# simulation parameters
dt = 0.05 # ms
t_final = 1000 # ms

# pacing parameters
pacing_start_time = 100 # ms
pacing_cycle_length = 300 # ms
pacing_duration = 5 # ms
J_stim_value = 1 # pacing strength

# heart model parameters
if heart_model_flag == 1: # Mitchell-Schaeffer
    model_name = "Mitchell-Schaeffer"

    tau_in = 0.08 # 0.3. 
    tau_out = 6 # 6. 
    tau_open = 80 # 120. 
    tau_close = 30 # 80. 
    u_gate = 0.13

    # initialize variables
    u = 0 # action potential
    h = 1 # gate variable

    time_scale = 1 # 1 model time unit = 1 ms
elif heart_model_flag == 2: # Aliev-Panfilov
    model_name = "Aliev-Panfilov"

    k = 12.0
    a = 0.15
    epsilon_0 = 0.002
    mu_1 = 0.2
    mu_2 = 0.3

    # initialize variables
    u = 0 # action potential
    h = 0 # gate variable

    time_scale = 6 # 1 model time unit = 6 ms

dt = dt / time_scale
t_final = t_final / time_scale
pacing_start_time = pacing_start_time / time_scale
pacing_cycle_length = pacing_cycle_length / time_scale
pacing_duration = pacing_duration / time_scale

# define the differential equations as functions
def f_u(u,h,J_stim):
    if heart_model_flag == 1: # Mitchell–Schaeffer
        return h * u**2 * (1 - u) / tau_in - u / tau_out + J_stim
    elif heart_model_flag == 2: # Aliev-Panfilov
        return -k*u*(u - a)*(u - 1) - u*h + J_stim

def f_h(u,h):
    if heart_model_flag == 1: # Mitchell–Schaeffer
        if u < u_gate:
            return (1 - h) / tau_open
        elif u >= u_gate:
            return -h / tau_close
    elif heart_model_flag == 2: # Aliev-Panfilov
        return (epsilon_0 + mu_1*h/(u + mu_2)) * (-h - k*u*(u - a - 1))

# time loop
time = np.array([])
action_potential = []
gate_variable = []
pacing_signal = []
nsteps = int(t_final / dt)
t0 = pacing_start_time
for n in range(nsteps):
    t = n * dt

    # pacing
    J_stim = 0
    if (t-pacing_start_time) % pacing_cycle_length == 0: 
        t0 = t # beginning of each pacing cycle
    if t >= t0 and t <= t0 + pacing_duration:
        J_stim = J_stim_value
    pacing_signal.append(J_stim)

    # Runge–Kutta 4 (RK4) solver
    k1_u = f_u(u, h, J_stim)
    k1_h = f_h(u, h)

    k2_u = f_u(u + 0.5*dt*k1_u, h + 0.5*dt*k1_h, J_stim)
    k2_h = f_h(u + 0.5*dt*k1_u, h + 0.5*dt*k1_h)

    k3_u = f_u(u + 0.5*dt*k2_u, h + 0.5*dt*k2_h, J_stim)
    k3_h = f_h(u + 0.5*dt*k2_u, h + 0.5*dt*k2_h)

    k4_u = f_u(u + dt*k3_u, h + dt*k3_h, J_stim)
    k4_h = f_h(u + dt*k3_u, h + dt*k3_h)

    u = u + (dt/6.0)*(k1_u + 2*k2_u + 2*k3_u + k4_u)
    h = h + (dt/6.0)*(k1_h + 2*k2_h + 2*k3_h + k4_h)

    # record
    time = np.append(time, t*time_scale) # convert back to ms for plotting
    action_potential.append(u)
    gate_variable.append(h)

#%%
# plot the simulation
fig, axes = plt.subplots(3, 1, figsize=(8,8), sharex=True)

# action potential
axes[0].plot(time, action_potential, 'b')
axes[0].grid(True)
axes[0].set_ylabel("u (action potential)")
axes[0].set_title(f"{model_name} model, single cell simulation")

# gate variable
axes[1].plot(time, gate_variable, 'b')
axes[1].set_ylabel('h (gate variable)')
axes[1].grid(True)

# pacing signal
axes[2].plot(time, pacing_signal, 'b')
axes[2].set_ylabel("Pacing signal")
axes[2].set_xlabel(f"Time (ms), # steps: {time.shape[0]}")
axes[2].grid(True)

plt.tight_layout()
plt.savefig(directory['result'] / f'single_cell_simulation_{model_name.replace("-", "_")}.png', dpi=300)
plt.close()

print('done')
