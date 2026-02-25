#%%
import os
from pathlib import Path
script_dir = os.path.dirname(os.path.abspath(__file__)) # get the path of the current script
os.chdir(script_dir) # change the working directory
script_dir = Path(script_dir)

import codes
import numpy as np # pip install numpy
import matplotlib # pip install matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox

#%%
data_flag = 1 # 1: from PyHeartSim. 2: from SPH-HeartSim

# load data
if data_flag == 1:
    data_dir = script_dir.parent / 'data'
    node_flag_dir = script_dir.parent / 'data'

    # loaded = np.load(data_dir / 'sheet.npz')
    # loaded = np.load(data_dir / 'slab.npz')
    loaded = np.load(data_dir / '49_2-LA_edited.npz')

    node = loaded['voxel']
elif data_flag == 2:
    data_dir = script_dir.parent / 'SPH-HeartSim' / 'build' / 'sim' / 'bin' / 'output'
    node_flag_dir = script_dir.parent / 'SPH-HeartSim' / 'result'

    t, voltage, gate_variable, stress, xyz = codes.load_SPH_simulation_result.execute(data_dir)
    node = xyz[:,0,:]

if os.path.exists(node_flag_dir / 'node_flag.npy'):
    node_flag = np.load(node_flag_dir / 'node_flag.npy')
else:
    node_flag = np.zeros(len(node), dtype=int)

debug_flag = 0
if debug_flag == 1:
    np.set_printoptions(threshold=np.inf)
    print('s1 pacing nodes: ')
    print(", ".join(map(str, np.where(node_flag==1)[0])))
    print('s2 pacing nodes: ')
    print(", ".join(map(str, np.where(node_flag==2)[0])))

#%%
# user interface
# ------------------------------
# initialize variables
selection_polygon = []
is_selecting = [False]  # is selection mode or not
selection_mode = [False]  # rotate/select mode

# plot the nodes
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(node[:, 0], node[:, 1], node[:, 2], c='grey', s=5, depthshade=True, marker='.')
ax.set_axis_off()
ax.set_title('3D Node Selection Tool')

# set axes equal
x_min, x_max = node[:, 0].min(), node[:, 0].max()
y_min, y_max = node[:, 1].min(), node[:, 1].max()
z_min, z_max = node[:, 2].min(), node[:, 2].max()
padding = 0.01
x_range = x_max - x_min
y_range = y_max - y_min
z_range = z_max - z_min
ax.set_xlim3d(x_min - padding*x_range, x_max + padding*x_range)
ax.set_ylim3d(y_min - padding*y_range, y_max + padding*y_range)
ax.set_zlim3d(z_min - padding*z_range, z_max + padding*z_range)

# text for showing current mode (rotation or selection mode)
mode_text = fig.text(0.5, 0.95, 
    'Mode: Rotate (press "a" change to Select)',
    ha='center', fontsize=12, weight='bold', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

# instruction text
fig.text(0.5, 0.02, 
    'Toggle Rotate/Select mode: Press "a" | Select nodes: Left click mouse and encircle | Clear all: Right click mouse',
    ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# textbox for flag value
ax_textbox = plt.axes([0.15, 0.90, 0.1, 0.04]) # [left, bottom, width, height]
textbox = TextBox(ax_textbox, 'Flag Value:', initial='1')
fig.text(0.15, 0.88, '1: s1. 2: s2', fontsize=10,)

# a button to save the node flags
ax_save_button = plt.axes([0.81, 0.90, 0.08, 0.04])
btn_save = Button(ax_save_button, 'Save')

# update the node colors according to node_flag
def update_node_color():
    colors = ['grey'] * node.shape[0] # initial color of all nodes 

    color_map = ['blue', 'red', 'orange', 'magenta']
    flagged_node_id = np.where(node_flag > 0)[0]
    for n in flagged_node_id:
        colors[n] = color_map[(node_flag[n]-1) % len(color_map)] 
        # '% len(color_map)' is to wrap around if the flag value is larger then the amount of color:
        # for example, if there are 4 colors in color_map, 
        # then len(color_map) = 4,
        # for node_flag[n] = 6, will have (node_flag[n]-1) % len(color_map) = 5%4 = 1, will give the 2nd element in color_map
    scatter.set_color(colors)

    fig.canvas.draw_idle()

update_node_color()

#%%
# key press to change mode: selection or rotation
def on_key(event):
    if event.key.lower() == 'a':
        selection_mode[0] = not selection_mode[0] # toggle true and false

        if selection_mode[0]:
            mode_text.set_text('Mode: Select (press "a" change to Rotate)')
            mode_text.set_bbox(dict(facecolor='yellow'))
            ax.disable_mouse_rotation()
        else:
            mode_text.set_text('Mode: Rotate (press "a" change to Select)')
            mode_text.set_bbox(dict(facecolor='lightgreen'))
            ax.mouse_init()

        fig.canvas.draw_idle()

fig.canvas.mpl_connect('key_press_event', on_key) # fig.canvas.mpl_connect('event_name', callback_function)

# at the time instance of a mouse click (either left or right click)
def on_press(event):
    if event.inaxes != ax: # if mouse outside of the user interface
        return
    
    if event.button == 1 and selection_mode[0]: # mouse left click and in selection mode
        is_selecting[0] = True
        selection_polygon.clear()
        selection_polygon.append((event.xdata, event.ydata))
    elif event.button == 3: # mouse right click
        node_flag[:] = 0
        selection_polygon.clear()
        update_node_color()

fig.canvas.mpl_connect('button_press_event', on_press) # fig.canvas.mpl_connect('event_name', callback_function)

# while the mouse is moving
def on_motion(event):
    if not is_selecting[0] or event.inaxes != ax: # if not pressing down mouse left key and in selection mode, or mouse outside of the user interface
        return
    
    if event.xdata is not None and event.ydata is not None: 
        selection_polygon.append((event.xdata, event.ydata)) # record the mouse x and y data of the screen

    debug_flag = 0
    if debug_flag == 1:
        polygon_array = np.array(selection_polygon, dtype=float)
        np.set_printoptions(precision=6, suppress=True)
        print('selection_polygon: ')
        print(polygon_array)

fig.canvas.mpl_connect('motion_notify_event', on_motion) # fig.canvas.mpl_connect('event_name', callback_function)

# at the time instance of mouse click release
def on_release(event):
    if not is_selecting[0] or event.button != 1:
        return
    
    is_selecting[0] = False

    if len(selection_polygon) < 3:
        print("Selection too small, need at least 3 points")
        selection_polygon.clear()
        return
    
    selection_polygon.append(selection_polygon[0]) # add the first point at the end to close the polygon

    debug_flag = 0
    if debug_flag == 1:
        print('selection_polygon: ')
        print(selection_polygon)

        polygon_array = np.array(selection_polygon, dtype=float)
        np.set_printoptions(precision=6, suppress=True)
        print('polygon_array: ')
        print(polygon_array)

    flag_value = int(textbox.text)

    # find out the nodes inside the selection region
    proj_matrix = ax.get_proj()
    node_2d = []
    depths = []
    for each_node in node:
        vec = np.array([each_node[0], each_node[1], each_node[2], 1.0])
        proj = proj_matrix @ vec
        if proj[3] != 0:
            x2d, y2d, z_depth = proj[0]/proj[3], proj[1]/proj[3], proj[2]/proj[3]
        else:
            x2d, y2d, z_depth = proj[0], proj[1], proj[2]
        node_2d.append([x2d, y2d])
        depths.append(z_depth)

    node_2d = np.array(node_2d)
    depths = np.array(depths)

    path = matplotlib.path.Path(selection_polygon)
    inside = path.contains_points(node_2d)
    inside_indices = np.where(inside)[0]

    min_depth = np.min(depths[inside_indices])
    depth_threshold = np.abs(min_depth) * 1/100 # a percentage of the depth
    front_facing = np.abs(depths - min_depth) < depth_threshold
    inside = inside & front_facing # grab only the nodes on front side, remove nodes in the back side

    node_flag[inside] = flag_value # assign flag value

    selection_polygon.clear()
    
    update_node_color()

fig.canvas.mpl_connect('button_release_event', on_release) # fig.canvas.mpl_connect('event_name', callback_function)

# the 'save' button
def save_selection(event):
    filename = 'node_flag.npy'
    np.save(node_flag_dir / filename, node_flag)
    print(f"Saved node flags to '{filename}'")

    s1_pacing_node_id = np.where(node_flag == 1)[0]
    s2_pacing_node_id = np.where(node_flag == 2)[0]
    print('s1_pacing_node_id:')
    print(','.join(map(str, s1_pacing_node_id)))
    print('s2_pacing_node_id:')
    print(','.join(map(str, s2_pacing_node_id)))

btn_save.on_clicked(save_selection)

plt.show() # show the user interface. this needs to be the last line so that the above functions can run
