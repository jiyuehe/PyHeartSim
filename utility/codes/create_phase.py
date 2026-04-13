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
import matplotlib.pyplot as plt
from scipy.signal import find_peaks # pip install scipy

# NOTE:
# This only works well if the action potential shapes are consistant:
# focal arrhythmia, simple rotor arrhythmia.
# It does not work well for fibrillation, because the action potential shapes are not consistant.

def execute(ap, v_gate):
    # action poential phase. phase within the action potential shape
    # --------------------------------------------------
    a = np.where(ap > v_gate)[0]  # time index where ap > v_gate
    
    debug_plot = 0
    if debug_plot == 1:
        plt.figure()
        plt.plot(ap, 'b')
        plt.plot(a, ap[a], 'r.')
        plt.show()

    if len(a) > 0:
        # find discontinuities in the indices
        diff_a = np.diff(a)
        b = np.where(np.abs(diff_a) > 1)[0]
        
        # create p array with start and end points of continuous segments
        if len(b) > 0:
            p = np.concatenate([[a[0]], a[b], a[b+1], [a[-1]]])
        else:
            p = np.array([a[0], a[-1]])
        
        p = np.sort(p)

        if debug_plot == 1:
            plt.figure()
            plt.plot(ap, 'b')
            plt.scatter(p, ap[p], s=100, c='r', marker='.')
            plt.show()
        
        # phase interval
        phase_interval = np.zeros((len(p)//2, 2))
        for n in range(len(p)//2):
            phase_interval[n, :] = p[n*2:n*2+2]

        # phase
        L_median = int(np.ceil(np.median(phase_interval[:, 1] - phase_interval[:, 0])))
        ap_phase = np.zeros_like(ap)
        
        for n in range(phase_interval.shape[0]):
            m = phase_interval[n, :].astype(int)
            L = len(range(m[0], m[1]))

            if m[0] == 0: 
                ap_phase[m[0]:m[1]] = np.linspace(m[0]/L_median, 1, L)
            elif m[0] != 0 and m[1] != len(ap_phase)-1:
                ap_phase[m[0]:m[1]] = np.linspace(0, 1, L)
            elif m[1] == len(ap_phase)-1:
                ap_phase[m[0]:m[1]] = np.linspace(0, min(1, L/L_median), L)
                
    else:  # if a is empty
        ap_phase = np.zeros_like(ap, dtype=float)

    if debug_plot == 1:
        plt.figure()
        plt.plot(ap, 'b')
        plt.plot(ap_phase, 'g')
        plt.tight_layout()
        plt.show()

    # activation phase. phase in between 2 activation peaks
    # --------------------------------------------------
    peaks, _ = find_peaks(ap, height=np.mean(ap)) # time index of peaks

    if debug_plot == 1:
        plt.figure()
        plt.plot(ap, 'b')
        plt.scatter(peaks, ap[peaks], c='r')
        plt.show()

    if len(peaks) > 0:
        # include t start and t final
        if peaks[0] != 0:
            peaks = np.concatenate([[0], peaks])
        if peaks[-1] != len(ap) - 1:
            peaks = np.concatenate([peaks, [len(ap) - 1]])
        
        # phase interval
        phase_interval = np.zeros((len(peaks)-1, 2), dtype=int)
        for n in range(len(peaks)-1):
            phase_interval[n, :] = [peaks[n], peaks[n+1]]

        # phase
        L_median = int(np.ceil(np.median(phase_interval[:, 1] - phase_interval[:, 0])))
        activation_phase = np.zeros_like(ap)
        
        for n in range(phase_interval.shape[0]):
            m = phase_interval[n, :]
            L = len(range(m[0], m[1]))

            if m[0] == 0:
                activation_phase[m[0]:m[1]] = np.linspace(1-L/L_median, 1, L)
            elif m[0] != 0 and m[1] != len(activation_phase)-1:
                activation_phase[m[0]:m[1]] = np.linspace(0, 1, L)
            elif m[1] == len(activation_phase)-1:
                activation_phase[m[0]:m[1]+1] = np.linspace(0, min(1, L/L_median), L+1)
                
    else:  # if peaks is empty
        activation_phase = np.zeros_like(ap, dtype=float)

    if debug_plot == 1:
        plt.figure()
        plt.plot(ap, 'b')
        plt.plot(activation_phase, 'g')
        plt.tight_layout()
        plt.show()

    return ap_phase, activation_phase