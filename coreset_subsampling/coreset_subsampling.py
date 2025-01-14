import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection

# Generate synthetic dataset
np.random.seed(0)
num_points = 200
data = np.random.rand(num_points, 2)


def euclidean_distance(a, b):
    return np.linalg.norm(a - b, axis=1)


def coreset_subsampling(data, coreset_size):
    """Coreset subsampling function"""
    indices = [np.random.randint(len(data))]
    coreset = data[indices]
    distances = euclidean_distance(data, coreset)

    for _ in range(coreset_size - 1):
        new_index = np.argmax(distances)
        indices.append(new_index)
        new_point = data[new_index]
        coreset = np.vstack([coreset, new_point])
        new_distances = euclidean_distance(data, new_point.reshape(1, -1))
        distances = np.minimum(distances, new_distances)

    return indices


# Parameters
coreset_size = 10
selected_indices = coreset_subsampling(data, coreset_size)

# Set up the figure and axis
fig, ax = plt.subplots()
fig.set_size_inches(8, 6)
ax.scatter(data[:, 0], data[:, 1], c='blue', label='Data Points')
selected_scat = ax.scatter([], [], c='red', label='Coreset Points')
highlight_scat = ax.scatter(
    [], [], c='yellow', s=100, edgecolor='black', label='New Point')
line_collection = LineCollection(
    [], color='green', linewidth=1, linestyle='--')
ax.add_collection(line_collection)
ax.legend()
ax.set_title(f'Coreset Subsampling (Size: {coreset_size})')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)



def init():
    """Initialization function for the animation"""
    selected_scat.set_offsets(np.empty((0, 2)))
    highlight_scat.set_offsets(np.empty((0, 2)))
    line_collection.set_segments([])
    return selected_scat, highlight_scat, line_collection



def update(frame):
    """Update function for the animation"""
    current_indices = selected_indices[:frame]
    new_index = selected_indices[frame]

    # Update selected and highlighted points
    selected_scat.set_offsets(data[current_indices])
    highlight_scat.set_offsets(data[new_index].reshape(1, -1))

    # Update lines showing distance comparisons
    segments = [np.array([data[i], data[new_index]]) for i in current_indices]
    line_collection.set_segments(segments)

    return selected_scat, highlight_scat, line_collection


# Create the animation
ani = animation.FuncAnimation(
    fig, update, frames=coreset_size, init_func=init,
    # Slower animation (1.5 seconds per frame)
    blit=True, repeat=False, interval=1500
)

# Display the animation
plt.show()
