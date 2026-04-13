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
from PIL import Image # pip install pillow
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

def set_axes_equal(ax): 
    # make 3D plot axes have equal scale so spheres look like spheres
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    max_range = max([abs(x_limits[1] - x_limits[0]), abs(y_limits[1] - y_limits[0]), abs(z_limits[1] - z_limits[0])]) / 2.0
    ax.set_xlim3d([np.mean(x_limits) - max_range, np.mean(x_limits) + max_range])
    ax.set_ylim3d([np.mean(y_limits) - max_range, np.mean(y_limits) + max_range])
    ax.set_zlim3d([np.mean(z_limits) - max_range, np.mean(z_limits) + max_range])

def crop_image(image_file_name): # crop result images to remove excess whitespace
    img = Image.open(image_file_name).convert("RGBA")
    data = np.array(img)

    # true where pixel is not white
    threshold = 245  # define near-white as background
    mask = np.any(data[:, :, :3] < threshold, axis=2)
    coords = np.argwhere(mask)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1

    img.crop((x0, y0, x1, y1)).save(image_file_name)

def convert_value_to_red_blue(data, data_min, data_max, data_threshold):
    # identify non-active regions (below threshold or NaN) before processing
    non_active_id = (data <= data_threshold) | np.isnan(data) | (data == -1)
    
    # clip values (NaN will remain NaN after clipping)
    data_clipped = np.clip(data, data_min, data_max)
    
    # calculate hue (NaN values will produce NaN hue)
    hue = (data_clipped - data_min) / (data_max - data_min) * (240.0 / 360.0)

    # assign color using HSV colormap
    hsv = np.zeros((hue.size, 3))
    hsv[:, 0] = np.nan_to_num(hue, nan=0.0)  # Replace NaN with 0 for color calculation
    hsv[:, 1] = 1.0
    hsv[:, 2] = 1.0
    map_color = mcolors.hsv_to_rgb(hsv)

    # assign non-active regions (including NaN) to gray
    map_color[non_active_id, :] = 0.5

    return map_color

def convert_value_to_purple_yellow(data, data_min, data_max, data_threshold):
    # identify non-active regions (below threshold or NaN) before processing
    non_active_id = (data <= data_threshold) | np.isnan(data)
    
    # clip values (NaN will remain NaN after clipping)
    data_clipped = np.clip(data, data_min, data_max)
    
    # normalize data to range [0, 1]
    normalized = (data_clipped - data_min) / (data_max - data_min)
    
    # use Viridis colormap (purple/dark blue -> cyan -> green -> yellow)
    viridis = plt.cm.viridis
    
    # apply colormap to normalized data
    map_color = viridis(np.nan_to_num(normalized, nan=0.0))[:, :3]  # take only RGB, drop alpha
    
    # assign non-active regions (including NaN) to gray
    map_color[non_active_id, :] = 0.5
    
    return map_color
