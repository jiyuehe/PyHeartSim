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

def equation_u(heart_model_flag, u, h, J_stim, tau_in, tau_out, k, a):
    if heart_model_flag == 1: # Mitchell–Schaeffer
        return h * u**2 * (1 - u) / tau_in - u / tau_out + J_stim
    elif heart_model_flag == 2: # Aliev-Panfilov
        return -k*u*(u - a)*(u - 1) - u*h + J_stim

def equation_h(heart_model_flag, u, h, tau_open, tau_close, u_gate, k, a, epsilon_0, mu_1, mu_2):
    if heart_model_flag == 1: # Mitchell–Schaeffer
        if u < u_gate:
            return (1 - h) / tau_open
        elif u >= u_gate:
            return -h / tau_close
    elif heart_model_flag == 2: # Aliev-Panfilov
        return (epsilon_0 + mu_1*h/(u + mu_2)) * (-h - k*u*(u - a - 1))

def solve_differential_equation(heart_model_flag, dt, t_final, pacing_start_time, pacing_cycle_length, pacing_duration, J_stim_value):
    # Mitchell-Schaeffer model parameters
    tau_in = 0.08 # 0.3. 
    tau_out = 6 # 6. 
    tau_open = 80 # 120. 
    tau_close = 30 # 80. 
    u_gate = 0.13

    # Aliev-Panfilov model parameters
    k = 12.0
    a = 0.15
    epsilon_0 = 0.002
    mu_1 = 0.2
    mu_2 = 0.3

    if heart_model_flag == 1: # Mitchell-Schaeffer
        u = 0 # initialize action potential
        h = 1 # initialize gate variable
        time_scale = 1 # 1 model time unit = 1 ms
    elif heart_model_flag == 2: # Aliev-Panfilov
        u = 0 # initialize action potential
        h = 0 # initialize gate variable
        time_scale = 6 # 1 model time unit = 6 ms

    dt = dt / time_scale
    t_final = t_final / time_scale
    pacing_start_time = pacing_start_time / time_scale
    pacing_cycle_length = pacing_cycle_length / time_scale
    pacing_duration = pacing_duration / time_scale

    time = np.array([])
    action_potential = []
    gate_variable = []
    pacing_signal = []
    nsteps = int(t_final / dt)
    for n in range(nsteps):
        if n % (nsteps//10) == 0:
            print(f'compute time step {n}/{nsteps}')

        t = n * dt

        # pacing
        J_stim = 0
        if (t-pacing_start_time) % pacing_cycle_length == 0: 
            pacing_start_time = t # beginning of each pacing cycle
        if t >= pacing_start_time and t <= pacing_start_time + pacing_duration:
            J_stim = J_stim_value
        pacing_signal.append(J_stim)

        # Runge–Kutta 4 (RK4) solver
        k1_u = equation_u(heart_model_flag, u, h, J_stim, tau_in, tau_out, k, a)
        k1_h = equation_h(heart_model_flag, u, h, tau_open, tau_close, u_gate, k, a, epsilon_0, mu_1, mu_2)

        k2_u = equation_u(heart_model_flag, u + 0.5*dt*k1_u, h + 0.5*dt*k1_h, J_stim, tau_in, tau_out, k, a)
        k2_h = equation_h(heart_model_flag, u + 0.5*dt*k1_u, h + 0.5*dt*k1_h, tau_open, tau_close, u_gate, k, a, epsilon_0, mu_1, mu_2)

        k3_u = equation_u(heart_model_flag, u + 0.5*dt*k2_u, h + 0.5*dt*k2_h, J_stim, tau_in, tau_out, k, a)
        k3_h = equation_h(heart_model_flag, u + 0.5*dt*k2_u, h + 0.5*dt*k2_h, tau_open, tau_close, u_gate, k, a, epsilon_0, mu_1, mu_2)

        k4_u = equation_u(heart_model_flag, u + dt*k3_u, h + dt*k3_h, J_stim, tau_in, tau_out, k, a)
        k4_h = equation_h(heart_model_flag, u + dt*k3_u, h + dt*k3_h, tau_open, tau_close, u_gate, k, a, epsilon_0, mu_1, mu_2)

        u = u + (dt/6.0)*(k1_u + 2*k2_u + 2*k3_u + k4_u)
        h = h + (dt/6.0)*(k1_h + 2*k2_h + 2*k3_h + k4_h)

        # save result
        time = np.append(time, t*time_scale) # convert back to ms for plotting
        action_potential.append(u)
        gate_variable.append(h)

    return pacing_signal, time, action_potential, gate_variable
