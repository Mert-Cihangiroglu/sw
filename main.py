
from torchvision import models
from utils.data_splitting import prepare_data
from utils.client import Client
from utils.validator import Validator
from utils.federated_utils import *
from torch.utils.data import Subset
from utils.device import get_device
from utils.evaluation import *
from utils.models import ResNet18
from rich.console import Console
#from utils.attacks import *
from utils.visualize import *
from datetime import datetime
import torch
import copy
import json
import warnings
warnings.filterwarnings("ignore")



def main(num_rounds=10, num_clients=10, num_classes_per_client=2, batch_size=64, iid_setting=True, 
         special_distribution=False, dirichlet=False, class_spesific=False, aggregation_method="fedavg", 
         top_k=3, scale_down_factor=None, attack_type=None, attack_params=None, alpha=None, zip_percent=None):
    
    round_accuracies = []
    # Print the settings
    print_settings(
        num_rounds=num_rounds,
        num_clients=num_clients,
        num_classes_per_client=num_classes_per_client,
        batch_size=batch_size,
        iid_setting=iid_setting,
        special_distribution=special_distribution,
        dirichlet=dirichlet,
        class_spesific=class_spesific,
        aggregation_method=aggregation_method,
        top_k=top_k,
        scale_down_factor=scale_down_factor,
        attack_type=attack_type,
        attack_params=attack_params,
        alpha=alpha,
        zip_percent=zip_percent
    )

    # Step 1: Decide if running in IID or Non-IID mode and prepare data accordingly
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
        description = "Non-IID Setting Class Specific"
    
    if dirichlet:
        client_loaders, validator_loaders, test_loader = prepare_data(
            batch_size=batch_size,
            num_clients=num_clients,
            setting='dirichlet',
            alpha=alpha  # Adjust alpha for varying levels of non-IID
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
    console.print("Global model initialized.\n")
    
    #Apply Attack (if specified)
    if attack_type:
        console.print(f"[bold red]Applying {attack_type} Attack[/bold red]")
        if attack_type == "label_flipping":
            attack = LabelFlippingAttack(attack_params.get("flip_mapping", {0: 1, 1: 0}))
            client_loaders = [attack.apply(loader) for loader in client_loaders]
        elif attack_type == "dba":
            attack = DBAttack(trigger_function=attack_params["trigger_function"], 
                              target_label=attack_params["target_label"])
            client_loaders = [attack.apply(loader) for loader in client_loaders]
        else:
            raise ValueError(f"Unsupported attack type: {attack_type}")
    
    # Step 4: Initialize Validators
    console.print("[bold blue]Initializing Validators[/bold blue]")
    validators = []
    if iid_setting:
        # For IID setting, use a single general validator 
        validators.append(Validator(validator_id=0, data_loader=validator_loader, device=device, scale_down_factor =scale_down_factor, zip_percent= zip_percent))
    else:
        # For Non-IID, initialize class-specific validators
        for class_id, data_loader in validator_loaders.items():
            validators.append(Validator(validator_id=class_id, data_loader=data_loader, device=device, scale_down_factor =scale_down_factor, zip_percent= zip_percent))
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
    
        elif aggregation_method == "lm_mask":
            
            console.print("[bold blue]Starting Useful Weight Aggregation[/bold blue]")
            # List to hold class-specific models after pruning
            class_specific_models = []
            local_model_masks = []
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

                mask  = validator.generate_mask_from_lm(top_model)
                local_model_masks.append(mask)
                
                # Step 3: Apply the mask to the top-performing model
                pruned_model = validator.apply_mask_to_model(top_model,mask)  # Mask application
                class_specific_models.append(pruned_model)
                
                console.print(f"[Validator {validator.validator_id}] Mask applied to the model from Client {top_model_id}.")

            # Step 4: Perform federated averaging on the pruned models to create the global model
            console.print("[bold blue]Performing Federated Averaging on Class-Specific Pruned Models[/bold blue]")
            #global_model = fed_avg_aggregation(class_specific_models)
            global_model = weighted_masked_aggregation(class_specific_models, local_model_masks)
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
        round_accuracies.append(round_accuracy)
        console.print(f"[bold green]End of Round {round_num + 1} - Global Model Loss: {round_loss:.4f}, Accuracy: {round_accuracy * 100:.2f}%[/bold green]\n")
    
    # Final evaluation after all rounds
    console.print("[bold blue]Final Evaluation of Global Model on Test Set[/bold blue]")
    final_loss, final_accuracy = evaluate_model(global_model, test_loader, device)
    console.print(f"[bold green]Final Global Model Loss: {final_loss:.4f}, Accuracy: {final_accuracy * 100:.2f}%[/bold green]\n")
    
    # Get the current time
    end_time = datetime.now().time()
    print("End Time:", end_time)
    
    #plot_global_accuracies(round_accuracies, num_rounds)
    return round_accuracies


def run_experiment(config_folder):
    """
    Runs experiments for all configurations in the given folder.
    """
    config_files = [f for f in os.listdir(config_folder) if f.endswith(".json")]

    for config_file in config_files:
        config_path = os.path.join(config_folder, config_file)
        print(f"Running configurations from {config_path}")

        with open(config_path, "r") as file:
            configurations = json.load(file)

        for i, config in enumerate(configurations):
            # Create experiment name based on method and config file
            experiment_name = f"{config['aggregation_method']}_{os.path.splitext(config_file)[0]}"
            print(f"Running experiment {i + 1}/{len(configurations)}: {experiment_name}")

            # Run the main experiment and get global accuracies
            round_accuracies = main(**config)

            # Save results for the experiment
            save_experiment_results(config, round_accuracies, experiment_name)
            
def save_experiment_results(config, round_accuracies, experiment_name):
    """
    Save experiment results to a unique directory based on the configuration.
    """
    # Create a unique directory for the experiment
    results_dir = os.path.join("final_results", experiment_name)
    os.makedirs(results_dir, exist_ok=True)

    # Save global accuracies
    accuracy_file = os.path.join(results_dir, f"{experiment_name}_global_accuracies.json")
    with open(accuracy_file, "w") as file:
        json.dump(round_accuracies, file)

    print(f"[Experiment {experiment_name}] Results saved to {results_dir}")
    
def generate_final_plot(results_dir):
    """
    Generate a final PNG showing the global accuracy development over time for all experiments.
    """
    all_accuracies = {}
    # Load all global accuracy results
    for experiment_folder in os.listdir(results_dir):
        folder_path = os.path.join(results_dir, experiment_folder)
        if os.path.isdir(folder_path):  # Check if it's a directory
            accuracy_file = os.path.join(folder_path, f"{experiment_folder}_global_accuracies.json")
            if os.path.exists(accuracy_file):
                with open(accuracy_file, "r") as file:
                    all_accuracies[experiment_folder] = json.load(file)

    # Plot all accuracies
    plt.figure(figsize=(12, 8))
    for experiment_name, accuracies in all_accuracies.items():
        plt.plot(range(1, len(accuracies) + 1), accuracies, label=experiment_name)

    plt.title("Global Model Accuracy Over Time for All Experiments")
    plt.xlabel("Federated Round")
    plt.ylabel("Global Model Accuracy")
    plt.legend(loc="lower right")
    plt.grid(True)

    # Save the plot
    plot_file = os.path.join(results_dir, "global_accuracies_summary.png")
    plt.savefig(plot_file)
    plt.close()

    print(f"Final accuracy plot saved to {plot_file}")

if __name__ == "__main__":
    # Define the base configuration directory
    base_config_dir = "configurations"
    results_dir = "final_results"

    # Ensure the results directory exists
    os.makedirs(results_dir, exist_ok=True)

    # Run experiments for each aggregation method
    for method in ["fedavg", "lm_mask", "gradcam"]:
        method_config_dir = os.path.join(base_config_dir, method)
        if os.path.exists(method_config_dir):
            print(f"Running experiments for {method}")
            run_experiment(method_config_dir)
        else:
            print(f"No configurations found for {method}")
    
    # Generate final plot after all experiments
    generate_final_plot(results_dir)