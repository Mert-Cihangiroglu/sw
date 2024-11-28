import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torchvision.datasets import CIFAR10
from torch.utils.data import Subset

def dirichlet_partition(dataset, num_clients, alpha, seed=42):
    if seed is not None:
        np.random.seed(seed)
    targets = np.array(dataset.targets)
    num_classes = len(np.unique(targets))
    indices = np.arange(len(targets))
    class_indices = [indices[targets == i] for i in range(num_classes)]
    proportions = np.random.dirichlet([alpha] * num_clients, num_classes)
    client_indices = [[] for _ in range(num_clients)]
    class_fractions = np.zeros((num_clients, num_classes))  # To store the fraction of each class in each client

    for c, class_idx in enumerate(class_indices):
        np.random.shuffle(class_idx)
        class_split = np.split(class_idx, (proportions[c] * len(class_idx)).cumsum()[:-1].astype(int))
        for client_id, split in enumerate(class_split):
            client_indices[client_id].extend(split)
            class_fractions[client_id, c] += len(split) / len(dataset)

    client_datasets = [Subset(dataset, idxs) for idxs in client_indices]
    return client_datasets, class_fractions

def plot_3d_data_distribution(class_fractions_list, alphas, num_clients, num_classes):
    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(111, projection='3d')
    x_labels = [f"Client {i+1}" for i in range(num_clients)]
    y_labels = [f"Class {i}" for i in range(num_classes)]
    xpos, ypos = np.meshgrid(np.arange(len(x_labels)), np.arange(len(alphas)), indexing="ij")
    xpos = xpos.flatten()
    ypos = ypos.flatten()
    zpos = np.zeros_like(xpos)

    dx = dy = 0.6
    for alpha_idx, alpha in enumerate(alphas):
        data = class_fractions_list[alpha_idx]
        for client_idx in range(num_clients):
            heights = data[client_idx]
            for class_idx, height in enumerate(heights):
                ax.bar3d(xpos[client_idx] + alpha_idx * len(x_labels), ypos[class_idx], zpos[class_idx],
                         dx, dy, height, label=f"Alpha = {alpha}")

    ax.set_xlabel("Clients")
    ax.set_ylabel("Classes")
    ax.set_zlabel("Fraction of Samples")
    plt.show()

# Step 1: Load Dataset
dataset = CIFAR10(root="./data", train=True, download=True)

# Step 2: Generate Data for Different Alphas
alphas = [4, 2, 1, 0.5, 0.25, 0.125]
num_clients = 5
class_fractions_list = []

for alpha in alphas:
    _, class_fractions = dirichlet_partition(dataset, num_clients, alpha)
    class_fractions_list.append(class_fractions)

# Step 3: Plot the 3D Data Distribution
plot_3d_data_distribution(class_fractions_list, alphas, num_clients, len(np.unique(dataset.targets)))