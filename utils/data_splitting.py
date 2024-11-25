import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import random
from collections import defaultdict
from rich.console import Console
from rich.table import Table
import numpy as np
from torch.utils.data import DataLoader, Subset
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from utils.client_class_assignments import ClientClassAssignments
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision.datasets import CIFAR10


def load_cifar10_data(train_size=0.8):
    """
    Load CIFAR-10 dataset and split into training and validation sets with appropriate transformations.
    
    Args:
        train_size (float): Proportion of data to use for training.
        
    Returns:
        train_data (Subset): Training data subset with augmentation and normalization.
        val_data (Subset): Validation data subset with normalization.
        test_data (Dataset): Official CIFAR-10 test dataset with normalization.
    """
    # Define transformations
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Load CIFAR-10 dataset
    full_train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    # Split the full training set into training and validation sets
    num_train = int(len(full_train_set) * train_size)
    num_val = len(full_train_set) - num_train
    train_data, val_data = torch.utils.data.random_split(full_train_set, [num_train, num_val])

    # Change validation data transformation to normalization only
    val_data.dataset.transform = transform_test  # Apply test transform to validation subset

    return train_data, val_data, test_set

def partition_data_iid(dataset, num_clients, batch_size):
    """
    Partition dataset into IID splits for each client.
    
    Args:
        dataset (Subset): The training dataset to partition.
        num_clients (int): Number of clients.
        batch_size (int): Batch size for each client's data loader.
        
    Returns:
        client_loaders (list): List of DataLoaders, one for each client.
    """
    num_samples_per_client = len(dataset) // num_clients
    indices = np.random.permutation(len(dataset))
    
    client_indices = [indices[i * num_samples_per_client : (i + 1) * num_samples_per_client] for i in range(num_clients)]
    client_loaders = [DataLoader(Subset(dataset, idx), batch_size=batch_size, shuffle=True) for idx in client_indices]
    
    return client_loaders

def partition_data_class_specific(dataset, num_clients, num_classes_per_client, batch_size):
    """
    Partition dataset so each client receives only a few classes, collectively covering all classes.
    
    Args:
        dataset (Subset): The training dataset to partition.
        num_clients (int): Number of clients.
        num_classes_per_client (int): Number of classes per client.
        batch_size (int): Batch size for each client's data loader.
        
    Returns:
        client_loaders (list): List of DataLoaders, one for each client.
        class_assignments (dict): Dictionary with client IDs as keys and lists of assigned classes as values.
    """
    targets = np.array(dataset.dataset.targets)
    subset_indices = np.array(dataset.indices)
    labels = targets[subset_indices]

    # Group indices by class
    class_indices = {class_id: subset_indices[labels == class_id].tolist() for class_id in range(10)}
    
    client_indices = [[] for _ in range(num_clients)]
    class_assignments = defaultdict(list)

    # Assign unique classes to each client while covering all classes collectively
    available_classes = list(class_indices.keys())
    class_groups = [available_classes[i:i+num_classes_per_client] for i in range(0, len(available_classes), num_classes_per_client)]
    for client_id in range(num_clients):
        selected_classes = class_groups[client_id % len(class_groups)]
        class_assignments[client_id] = selected_classes

        for class_id in selected_classes:
            # Allocate samples for the selected classes to this client
            num_samples = len(class_indices[class_id]) // (num_clients // len(class_groups))
            client_indices[client_id].extend(class_indices[class_id][:num_samples])
            class_indices[class_id] = class_indices[class_id][num_samples:]

    # Create DataLoaders for each client
    client_loaders = [
        DataLoader(Subset(dataset.dataset, indices), batch_size=batch_size, shuffle=True)
        for indices in client_indices
    ]
    return client_loaders, class_assignments

def partition_data_dominant_classes(dataset, num_clients, num_dominant_classes, dominant_fraction, batch_size):
    """
    Partition dataset into non-IID splits with dominant classes for each client.
    
    Args:
        dataset (Subset): The training dataset to partition.
        num_clients (int): Number of clients.
        num_dominant_classes (int): Number of dominant classes per client.
        dominant_fraction (float): Fraction of data to allocate to dominant classes.
        batch_size (int): Batch size for each client's data loader.
        
    Returns:
        client_loaders (list): List of DataLoaders, one for each client.
        dominant_class_assignments (dict): Dictionary of dominant classes assigned to each client.
    """
    targets = np.array(dataset.dataset.targets)
    subset_indices = np.array(dataset.indices)
    labels = targets[subset_indices]

    # Group indices by class
    class_indices = {class_id: subset_indices[labels == class_id].tolist() for class_id in range(10)}
    
    client_indices = [[] for _ in range(num_clients)]
    dominant_class_assignments = defaultdict(list)

    for client_id in range(num_clients):
        # Select dominant classes for this client
        dominant_classes = random.sample(list(class_indices.keys()), num_dominant_classes)
        dominant_class_assignments[client_id] = dominant_classes

        for class_id in range(10):
            if class_id in dominant_classes:
                # Allocate dominant fraction for dominant classes
                num_samples = int(len(class_indices[class_id]) * dominant_fraction / num_clients)
            else:
                # Allocate remaining fraction for non-dominant classes
                num_samples = int(len(class_indices[class_id]) * (1 - dominant_fraction) / (10 - num_dominant_classes) / num_clients)
            
            client_indices[client_id].extend(class_indices[class_id][:num_samples])
            class_indices[class_id] = class_indices[class_id][num_samples:]

    # Distribute remaining samples in round-robin fashion
    for class_id, indices in class_indices.items():
        for i, idx in enumerate(indices):
            client_indices[i % num_clients].append(idx)

    client_loaders = [
        DataLoader(Subset(dataset.dataset, indices), batch_size=batch_size, shuffle=True)
        for indices in client_indices
    ]
    return client_loaders, dominant_class_assignments

def partition_data_specific_classes(dataset, client_class_assignments, dominant_fraction, batch_size):
    """
    Partition dataset so that each client gets specific classes assigned.
    
    Args:
        dataset (Subset): The training dataset to partition.
        client_class_assignments (dict): Dictionary specifying class assignments for each client.
                                         Example: {0: [1, 2], 1: [3, 4]} for client 0 getting classes 1 and 2.
        dominant_fraction (float): Fraction of data to allocate to specified classes for each client.
        batch_size (int): Batch size for each client's data loader.
        
    Returns:
        client_loaders (list): List of DataLoaders, one for each client.
        assigned_classes (dict): Dictionary showing assigned classes for each client.
    """
    # Access targets from the original dataset within the Subset object
    if isinstance(dataset, Subset):
        targets = np.array(dataset.dataset.targets)[dataset.indices]
    else:
        targets = np.array(dataset.targets)
    num_clients = len(client_class_assignments)
    
    # Group indices by class
    class_indices = {class_id: np.where(targets == class_id)[0].tolist() for class_id in range(10)}
    
    client_indices = [[] for _ in range(num_clients)]
    assigned_classes = defaultdict(list)
    
    for client_id, assigned_classes_list in client_class_assignments.items():
        # Assign specified classes to each client
        assigned_classes[client_id] = assigned_classes_list
        
        for class_id in range(10):
            if class_id in assigned_classes_list:
                # Allocate dominant fraction of samples for assigned classes
                num_samples = int(len(class_indices[class_id]) * dominant_fraction)
                selected_indices = class_indices[class_id][:num_samples]
                client_indices[client_id].extend(selected_indices)
                class_indices[class_id] = class_indices[class_id][num_samples:]
            else:
                # Skip non-assigned classes for this client
                continue

    # Create DataLoaders for each client with specified batch sizes
    client_loaders = [
        DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=True)
        for indices in client_indices
    ]
    
    return client_loaders, assigned_classes

def dirichlet_partition(dataset, num_clients, alpha, seed=42):
    """
    Partition a dataset into non-IID splits using Dirichlet distribution.

    Args:
        dataset (Dataset): PyTorch dataset to partition.
        num_clients (int): Number of clients.
        alpha (float): Dirichlet concentration parameter (smaller = more non-IID).
        seed (int, optional): Random seed to ensure reproducibility.

    Returns:
        List[Subset]: List of PyTorch Subsets, one for each client.
    """
    if seed is not None:
        np.random.seed(seed)  # Set the random seed for reproducibility

    targets = np.array(dataset.targets)  # Assuming dataset has a 'targets' attribute
    num_classes = len(np.unique(targets))
    indices = np.arange(len(targets))

    # Group indices by class
    class_indices = [indices[targets == i] for i in range(num_classes)]

    # Generate Dirichlet distribution for each class
    proportions = np.random.dirichlet([alpha] * num_clients, num_classes)

    # Allocate indices to clients
    client_indices = [[] for _ in range(num_clients)]
    for c, class_idx in enumerate(class_indices):
        np.random.shuffle(class_idx)  # Shuffle class indices (controlled by seed above)
        class_split = np.split(class_idx, (proportions[c] * len(class_idx)).cumsum()[:-1].astype(int))
        for client_id, split in enumerate(class_split):
            client_indices[client_id].extend(split)

    # Create Subsets for each client
    client_datasets = [Subset(dataset, idxs) for idxs in client_indices]
    return client_datasets

def prepare_data(batch_size, num_clients, alpha, setting='iid', num_classes_per_client=None):
    """
    Prepare CIFAR-10 dataset for federated learning with various distribution settings.

    Args:
        batch_size (int): Batch size for data loaders.
        num_clients (int): Number of clients.
        setting (str): Distribution setting ('iid', 'class_specific', 'dominant', 'dirichlet').
        alpha (float, optional): Dirichlet concentration parameter for 'dirichlet' setting.
        num_classes_per_client (int, optional): Number of classes per client for 'class_specific' setting.

    Returns:
        client_loaders, validator_loaders (dict or single DataLoader), and test_loader.
    """
    # Load CIFAR-10 dataset
    transform = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_data = CIFAR10(root="./data", train=True, download=True, transform=transform)
    val_data = CIFAR10(root="./data", train=False, download=True, transform=transform)
    test_data = CIFAR10(root="./data", train=False, download=True, transform=transform)

    if setting == 'iid':
        # IID: Split the dataset into equal parts
        total_samples = len(train_data)
        samples_per_client = total_samples // num_clients
        indices = np.arange(total_samples)
        np.random.shuffle(indices)

        client_loaders = [
            DataLoader(
                Subset(train_data, indices[i * samples_per_client:(i + 1) * samples_per_client]),
                batch_size=batch_size,
                shuffle=True,
            )
            for i in range(num_clients)
        ]
        validator_loaders = create_class_specific_validators(val_data, batch_size)
        description = "IID Setting"

    elif setting == 'dirichlet':
        # Dirichlet-based non-IID partitioning
        client_datasets = dirichlet_partition(train_data, num_clients=num_clients, alpha=alpha)
        client_loaders = [
            DataLoader(client_ds, batch_size=batch_size, shuffle=True) for client_ds in client_datasets
        ]
        validator_loaders = create_class_specific_validators(val_data, batch_size)
        description = f"Dirichlet Non-IID Setting (alpha={alpha})"

    elif setting == 'class_specific':
        # Class-specific non-IID partitioning
        client_loaders, class_assignments = partition_data_class_specific(
            train_data, num_clients, num_classes_per_client, batch_size
        )
        validator_loaders = create_class_specific_validators(val_data, batch_size)
        description = "Class-Specific Non-IID Setting"
        console.print(f"[bold green]Class assignments per client: {class_assignments}[/bold green]")

    else:
        raise ValueError("Unsupported setting. Choose from 'iid', 'dirichlet', or 'class_specific'.")

    # Print data statistics
    print_data_statistics(client_loaders, validator_loaders, description=description)

    # Return loaders
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return client_loaders, validator_loaders, test_loader

# Utility Functions
def create_class_specific_validators(val_data, batch_size):
    """
    Create DataLoaders for validators, one for each class in the validation set.

    Args:
        val_data (Dataset): Validation dataset.
        batch_size (int): Batch size for DataLoader.

    Returns:
        dict[int, DataLoader]: Dictionary of DataLoaders keyed by class ID.
    """
    validator_loaders = {
        class_id: DataLoader(
            Subset(val_data, [idx for idx, label in enumerate(val_data.targets) if label == class_id]),
            batch_size=batch_size,
            shuffle=False,
        )
        for class_id in range(10)  # CIFAR-10 has 10 classes
    }
    return validator_loaders

def print_data_statistics(client_loaders, validator_loaders=None, description=""):
    """
    Print statistics table for client and validator datasets.
    
    Args:
        client_loaders (list): List of DataLoaders, one for each client.
        validator_loaders (dict or DataLoader): Dict of class-specific validators (non-IID) or single DataLoader (IID).
        description (str): Description of the distribution setting.
    """
    console = Console()
    console.print(f"\n[bold blue]Dataset Statistics for {description}[/bold blue]")
    table = Table(title=f"{description} - Client and Validator Dataset Statistics")
    table.add_column("Class ID", justify="right", style="cyan", no_wrap=True)

    # Add columns for each client
    for i in range(len(client_loaders)):
        table.add_column(f"Client {i}", justify="right", style="magenta")

    # Add columns for validators if provided
    if validator_loaders:
        if isinstance(validator_loaders, dict):  # Class-specific validators for non-IID settings
            for class_id in sorted(validator_loaders.keys()):
                table.add_column(f"Validator Class {class_id}", justify="right", style="green")
        else:  # General validation set for IID setting
            table.add_column("Validator Set", justify="right", style="green")

    # Collect statistics for each client
    client_stats = []
    for loader in client_loaders:
        all_labels = []
        for _, labels in loader:
            all_labels.extend(labels.tolist())
        client_stats.append(np.bincount(all_labels, minlength=10))

    # Collect statistics for each validator (if present)
    validator_stats = []
    if validator_loaders:
        if isinstance(validator_loaders, dict):  # Class-specific validators
            for class_id, loader in validator_loaders.items():
                all_labels = []
                for _, labels in loader:
                    all_labels.extend(labels.tolist())
                validator_stats.append(np.bincount(all_labels, minlength=10))
        else:  # Single validator set (IID setting)
            all_labels = []
            for _, labels in validator_loaders:
                all_labels.extend(labels.tolist())
            validator_stats.append(np.bincount(all_labels, minlength=10))

    # Fill in the table for each class
    for class_id in range(10):
        row = [str(class_id)]
        # Add client stats
        for stats in client_stats:
            row.append(str(stats[class_id]) if class_id < len(stats) else "0")
        # Add validator stats
        if validator_loaders:
            for stats in validator_stats:
                row.append(str(stats[class_id]) if class_id < len(stats) else "0")
        table.add_row(*row)

    console.print(table)
    
def test_all_settings():
    batch_size = 64
    num_clients = 10

    # Test IID Setting
    print("\n[bold green]Testing IID Setting[/bold green]")
    prepare_federated_data(batch_size=batch_size, num_clients=num_clients, setting='iid')

    # Test Class-Specific Non-IID Setting
    num_classes_per_client = 3
    print("\n[bold green]Testing Class-Specific Non-IID Setting[/bold green]")
    prepare_federated_data(batch_size=batch_size, num_clients=num_clients, setting='class_specific', num_classes_per_client=num_classes_per_client)

    # Test Dominant Class Non-IID Setting
    num_dominant_classes = 3
    dominant_fraction = 1
    print("\n[bold green]Testing Dominant Class Non-IID Setting[/bold green]")
    prepare_federated_data(batch_size=batch_size, num_clients=num_clients, setting='dominant', num_dominant_classes=num_dominant_classes, dominant_fraction=dominant_fraction)

if __name__ == "__main__":
    test_all_settings()