# find the pacing locations
activation_uni = input_arguments['geometry_data']['activation_uni']
activation_uni = activation_uni.astype(float)
activation_uni[activation_uni==0] = np.nan
electrode_node_id = input_arguments['geometry_data']['electrode_node_id']
node = input_arguments['geometry_data']['voxel']
electrode_nodes = node[electrode_node_id, :]

# find pacing electrodes
pacing_electrodes_id = np.where(activation_uni <= np.nanmin(activation_uni)+5)
pacing_node_id = electrode_node_id[pacing_electrodes_id]
pacing_nodes = node[pacing_node_id, :]

# cluster the pacing_nodes into 2 clusters based on distance
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2, random_state=0)
cluster_labels = kmeans.fit_predict(pacing_nodes)
cluster_centers = kmeans.cluster_centers_

# find the node id that is nearest to each cluster center
from scipy.spatial.distance import cdist
distances = cdist(cluster_centers, node)
nearest_node_ids = np.argmin(distances, axis=1)

debug_plot = 0
if debug_plot == 1:
    data = activation_uni
    data_min = np.nanmin(data)
    data_max = np.nanmax(data)
    data_threshold = data_min - 0.01
    map_color = common.convert_data_to_color.execute(data, data_min, data_max, data_threshold)

    # use plotly to display the electrode nodes and assign them map_color
    import plotly.graph_objects as go
    fig = go.Figure(data=[go.Scatter3d(
        x=electrode_nodes[:, 0],
        y=electrode_nodes[:, 1],
        z=electrode_nodes[:, 2],
        mode='markers',
        marker=dict(
            size=5,
            color=map_color
        ),
        name='Electrodes'
    )])
    
    # Add red cross markers for pacing electrodes
    fig.add_trace(go.Scatter3d(
        x=pacing_nodes[:, 0],
        y=pacing_nodes[:, 1],
        z=pacing_nodes[:, 2],
        mode='markers',
        marker=dict(
            size=4,
            color='red',
            symbol='x'
        ),
        name='Pacing Electrodes'
    ))
    
    # Plot cluster centers with black cross
    fig.add_trace(go.Scatter3d(
        x=node[nearest_node_ids, 0],
        y=node[nearest_node_ids, 1],
        z=node[nearest_node_ids, 2],
        mode='markers',
        marker=dict(
            size=4,
            color='black',
            symbol='x'
        ),
        name='Cluster Centers'
    ))
    
    fig.update_layout(
        title='Electrode Nodes with Activation Color',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        )
    )
    fig.show()
# ==============================

s1 = nearest_node_ids[0] # node id for s1 pacing
s2 = nearest_node_ids[1] # node id for s2 pacing