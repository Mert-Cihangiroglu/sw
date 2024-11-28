from rich.console import Console
import matplotlib.pyplot as plt
from datetime import datetime

def print_settings(num_rounds, num_clients, num_classes_per_client, batch_size, iid_setting, 
                   special_distribution, dirichlet, class_spesific, aggregation_method, top_k, 
                   scale_down_factor, attack_type, attack_params, alpha, zip_percent):
    """
    Prints all settings for the current experiment in a structured format.
    """
    console = Console()
    console.print("[bold blue]Experiment Settings[/bold blue]")
    print(f"  Begin Time: {datetime.now().time()}")
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
    print(f"  Scale-Down Factor (if applicable): {scale_down_factor if scale_down_factor else 'Not Applicable'}")
    print(f"  Zip Percent (if applicable): {zip_percent if zip_percent else 'Not Applicable'}")
    print(f"  Dirichlet Alpha (if applicable): {alpha if alpha else 'Not Applicable'}")
    print(f"  Attack Type: {attack_type if attack_type else 'None'}")
    if attack_type and attack_params:
        print(f"  Attack Parameters: {attack_params}")
    print(f"  Experiment Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 80)
    
def plot_global_accuracies(round_accuracies, num_rounds):
   
    plt.figure(figsize=(10, 6))
    # Plot global model accuracy
    plt.plot(range(1, num_rounds + 1), round_accuracies, label="Global Model Accuracy", linewidth=2, marker='o')
    plt.title("Global Model Accuracy Over Federated Rounds")
    plt.xlabel("Rounds")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()
    plt.savefig("global_model_accuracies.png")
    plt.show()