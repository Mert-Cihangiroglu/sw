import torch
import copy
from torchvision import models
from utils.data_splitting import prepare_data
from utils.client import Client
from utils.validator import Validator
from utils.federated_utils import fed_avg_aggregation
from utils.device import get_device
from utils.evaluation import *
from utils.models import ResNet18
from rich.console import Console
import warnings
from datetime import datetime
warnings.filterwarnings("ignore")
from torch.utils.data import Subset
import json



def dirichlet_partition(dataset, num_clients, alpha=0.5):
    """
    Partition a dataset into non-IID splits using Dirichlet distribution.

    Args:
        dataset (Dataset): PyTorch dataset to partition.
        num_clients (int): Number of clients.
        alpha (float): Dirichlet concentration parameter (lower values = more non-IID).

    Returns:
        List[Subset]: List of PyTorch Subsets, one for each client.
    """
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
        np.random.shuffle(class_idx)
        class_split = np.split(class_idx, (proportions[c] * len(class_idx)).cumsum()[:-1].astype(int))
        for client_id, split in enumerate(class_split):
            client_indices[client_id].extend(split)

    # Create Subsets for each client
    client_datasets = [Subset(dataset, idxs) for idxs in client_indices]

    return client_datasets


def main(num_rounds=10, num_clients=10, num_classes_per_client=2, batch_size=64, iid_setting=True, special_distribution =False, dirichlet=False, class_spesific=False,  aggregation_method="fedavg", top_k=3, scale_down_factor = 0.5):
    # Get the current time
    begin_time = datetime.now().time()
    print("Begin Time:", begin_time)
    console = Console()
    print("Settings:")
    print(f"  Number of Federated Rounds: {num_rounds}")
    print(f"  Total Number of Clients: {num_clients}")
    print(f"  Classes per Client (Non-IID Setting): {num_classes_per_client}")
    print(f"  Batch Size per Client: {batch_size}")
    print(f"  IID Setting: {'Enabled' if iid_setting else 'Disabled'}")
    print(f"  Special Distribution: {'Enabled' if special_distribution else 'Disabled'}")
    print(f"  Class-Specific Partitioning: {'Enabled' if class_spesific else 'Disabled'}")
    print(f"  Dirichlet Partitioning: {'Enabled' if dirichlet else 'Disabled'}")
    print(f"  Aggregation Method: {aggregation_method}")
    print(f"  Top-K for Model Selection: {top_k}")
    
    # Step 1: Decide if running in IID or Non-IID mode and prepare data accordingly
    console.print("[bold blue]Preparing Dataset[/bold blue]")
    if iid_setting:
        client_loaders, validator_loader, test_loader = prepare_data(
            batch_size=batch_size, 
            num_clients=num_clients, 
            setting='iid'
        )
        description = "IID Setting"
    
    if special_distribution:
        client_loaders, validator_loaders, test_loader = prepare_data(
            batch_size=batch_size, 
            num_clients=num_clients, 
            setting='non_iid_specific_classes', 
            num_classes_per_client=num_classes_per_client
        )
        description = "Non-IID Setting - Specific Classes" 
         
    if class_spesific:
        client_loaders, validator_loaders, test_loader = prepare_data(
            batch_size=batch_size, 
            num_clients=num_clients, 
            setting='class_specific', 
            num_classes_per_client=num_classes_per_client
        )
        description = "Non-IID Setting"
    
    if dirichlet:
        client_loaders, validator_loaders, test_loader = prepare_data(
            batch_size=32,
            num_clients=10,
            setting='dirichlet',
            alpha=0.5  # Adjust alpha for varying levels of non-IID
        )
        description = "Non-IID Dirichlet"
        
    console.print(f"[bold green]Using {description}[/bold green]\n")

    # Step 2: Initialize Clients
    device = get_device()
    console.print("[bold blue]Initializing Clients[/bold blue]")
    clients = []
    for client_id, data_loader in enumerate(client_loaders):
        clients.append(Client(client_id=client_id, data_loader=data_loader, device=device))
    console.print(f"Initialized {len(clients)} clients.\n")
    
    # Step 3: Initialize Global Model
    console.print("[bold blue]Initializing Global Model (ResNet-18)[/bold blue]")
    global_model = models.resnet18(pretrained=False, num_classes=10).to(device)
    #global_model = ResNet18().to(device)
    console.print("Global model initialized.\n")
    
    # Step 4: Initialize Validators
    console.print("[bold blue]Initializing Validators[/bold blue]")
    validators = []
    if iid_setting:
        # For IID setting, use a single general validator 
        validators.append(Validator(validator_id=0, data_loader=validator_loader, device=device, global_model=global_model))
    else:
        # For Non-IID, initialize class-specific validators
        for class_id, data_loader in validator_loaders.items():
            validators.append(Validator(validator_id=class_id, data_loader=data_loader, device=device, global_model=global_model))
    console.print(f"Initialized {len(validators)} validators.\n")
    
    # Training loop for federated rounds
    console.print("[bold blue]Starting Federated Training Rounds[/bold blue]")
    for round_num in range(num_rounds):
        console.print(f"\n[bold magenta]Round {round_num + 1}/{num_rounds}[/bold magenta]")
        
        # Step 5: Train each client's local model using the latest global model
        for client in clients:
            # Each client starts with a copy of the global model
            client_model = copy.deepcopy(global_model).to(device)
            console.print(f"  Training Client {client.client_id}")
            client.train(client_model)
        
        # Step 6: Aggregation
        console.print("\n[bold blue]Aggregating Models[/bold blue]")
        
        # Retrieve the models from each client
        local_models = [client.model for client in clients]
        
        if aggregation_method == "fedavg":
            global_model = fed_avg_aggregation(local_models)
            console.print("Aggregation complete.\n")
        
        elif aggregation_method == "v1":
            # Step 1: Validators evaluate each model and select top-performing ones
            class_specific_models = []
            for validator in validators:
                top_models = validator.get_top_models(local_models, top_k=top_k)
                
                # Step 2: Prune and average top models
                class_model = validator.prune_and_average_models(top_models)
                class_specific_models.append(class_model)
            
            # Step 3: FedAvg across all class-specific models to create the global model
            global_model = fed_avg_aggregation(class_specific_models)
            console.print("Aggregation complete.\n")
            
            console.print("[bold blue]Model Usage Statistics by Validators[/bold blue]")
            for validator in validators:
                usage_stats = validator.get_model_usage_stats()
                console.print(f"[Validator {validator.validator_id}] Model Usage Stats: {usage_stats}")
            
        elif aggregation_method == "v2":
            
            console.print("[bold blue]Starting Useful Weight Aggregation[/bold blue]")
            # List to hold class-specific models after pruning
            class_specific_models = []
            # Select only the validators for classes 0 and 2
            #selected_validators = [validator for validator in validators if validator.validator_id in [0, 2]]

            for validator in validators: #selected_validators:
                console.print(f"[Validator {validator.validator_id}] Evaluating top model from local models.")
                
                # update the global model
                validator.global_model = copy.deepcopy(global_model).to(device)
               
                # Step 1: Select the top-performing model for each validator
                top_model, top_model_id = validator.get_top_model(local_models)
                validator.log_model_usage(top_model_id)  # Log model usage
                console.print(f"[Validator {validator.validator_id}] Top model selected from Client {top_model_id}")

                # Step 2: Check if the mask exists
                console.print(f"[Validator {validator.validator_id}] is generating the mask.")
                
                '''
                if round_num < 4:
                    validator.train_class_specific_model()                  # Train and create the mask if it doesn’t exist
                    mask = validator.generate_and_save_weighted_mask()      # Generate and save the mask
                    
                else:
                '''
                #mask  = validator.generate_mask_from_gm()
                mask  = validator.generate_mask_from_lm(top_model)
                
                # Step 3: Apply the mask to the top-performing model
                pruned_model = validator.apply_mask_to_model(top_model,mask)  # Mask application
                class_specific_models.append(pruned_model)
                console.print(f"[Validator {validator.validator_id}] Mask applied to the model from Client {top_model_id}.")

            # Step 4: Perform federated averaging on the pruned models to create the global model
            console.print("[bold blue]Performing Federated Averaging on Class-Specific Pruned Models[/bold blue]")
            global_model = fed_avg_aggregation(class_specific_models)
            console.print("Aggregation complete.\n")
            
            console.print("[bold blue]Model Usage Statistics by Validators[/bold blue]")
            for validator in validators:
                usage_stats = validator.get_model_usage_stats()
                console.print(f"[Validator {validator.validator_id}] Model Usage Stats: {usage_stats}")
            
        elif aggregation_method == "v3":
            console.print(f"[bold blue]Starting {aggregation_method}[/bold blue]")
            class_specific_models = []

            for validator in validators:
                # Get top-performing model
                top_model, top_model_id = validator.get_top_model(local_models)
                validator.log_model_usage(top_model_id)

                # Generate and apply class-specific mask
                mask = validator.generate_class_specific_mask(top_model, freeze_early_layers=True)
                pruned_model = validator.apply_class_specific_mask(top_model, mask)

                # Append pruned model to class-specific models for federated averaging
                class_specific_models.append(pruned_model)

            # Perform federated averaging on the class-specific pruned models
            global_model = fed_avg_aggregation(class_specific_models)
            # global model =  sum(masks) weighted averaging 
            console.print("Aggregation complete.\n")
            
            console.print("[bold blue]Model Usage Statistics by Validators[/bold blue]")
            for validator in validators:
                usage_stats = validator.get_model_usage_stats()
                console.print(f"[Validator {validator.validator_id}] Model Usage Stats: {usage_stats}")

        elif aggregation_method == "gradcam":
            console.print(f"[bold blue]Starting {aggregation_method}[/bold blue]")
            class_specific_models = []

            for validator in validators:
                # Get top-performing model
                top_model, top_model_id = validator.get_top_model(local_models)
                validator.log_model_usage(top_model_id)
                
                # Generate and apply class-specific mask
                mask = validator.generate_mask_with_gradcam(top_model, scale_down_factor, target_layer="layer4" )
                pruned_model = validator.apply_class_specific_mask(top_model, mask)

                # Append pruned model to class-specific models for federated averaging
                class_specific_models.append(pruned_model)

            # Perform federated averaging on the class-specific pruned models
            global_model = fed_avg_aggregation(class_specific_models)
            # global model =  sum(masks) weighted averaging 
            console.print("Aggregation complete.\n")
            
            console.print("[bold blue]Model Usage Statistics by Validators[/bold blue]")
            for validator in validators:
                usage_stats = validator.get_model_usage_stats()
                console.print(f"[Validator {validator.validator_id}] Model Usage Stats: {usage_stats}")
            
        else:
            raise ValueError("Unsupported aggregation method.")
        
        # Step 7: Evaluate the global model on the test set after aggregation
        console.print("[bold blue]Evaluating Global Model on Validation/Test Set[/bold blue]")
        round_loss, round_accuracy, _ = evaluate_model_with_class_metrics(global_model, test_loader, device)
        console.print(f"[bold green]End of Round {round_num + 1} - Global Model Loss: {round_loss:.4f}, Accuracy: {round_accuracy * 100:.2f}%[/bold green]\n")
    
    # Final evaluation after all rounds
    console.print("[bold blue]Final Evaluation of Global Model on Test Set[/bold blue]")
    final_loss, final_accuracy = evaluate_model(global_model, test_loader, device)
    console.print(f"[bold green]Final Global Model Loss: {final_loss:.4f}, Accuracy: {final_accuracy * 100:.2f}%[/bold green]\n")
    
    # Get the current time
    end_time = datetime.now().time()
    print("End Time:", end_time)
    
if __name__ == "__main__":
    # Load configurations from the JSON file
    with open("configurations.json", "r") as config_file:
        configurations = json.load(config_file)

    # Iterate over each configuration and call main
    for i, config in enumerate(configurations):
        print(f"Running configuration {i + 1}/{len(configurations)}")
        main(**config)