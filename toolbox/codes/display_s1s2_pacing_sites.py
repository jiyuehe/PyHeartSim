# import codes
import numpy as np
import matplotlib.pyplot as plt

def execute(node, node_s1, node_s2):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("pacing sites")

    # plot all nodes
    ax.scatter(node[:, 0], node[:, 1], node[:, 2], c='gray', s=0.01, label='geometry', alpha=0.5)

    # plot s1 pacing nodes
    if np.ndim(node_s1) > 1:
        ax.scatter(node_s1[:, 0], node_s1[:, 1], node_s1[:, 2], c='blue', s=20, label='S1 pacing', alpha=1)
    elif np.ndim(node_s1) == 1:
        ax.scatter(node_s1[0], node_s1[1], node_s1[2], c='blue', s=20, label='S1 pacing', alpha=1)

    # plot s2 pacing nodes
    ax.scatter(node_s2[:, 0], node_s2[:, 1], node_s2[:, 2], c='red', s=20, label='S2 pacing', alpha=1)

    # labels
    ax.legend()
    plt.axis('off')

    # codes.set_axes_equal.execute(ax)
