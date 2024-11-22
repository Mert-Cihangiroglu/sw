import torch
from rich.console import Console

# Initialize Rich Console
console = Console()

# Device setup function
def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        console.print("🚀 Using [bold green]GPU (CUDA)[/bold green]")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        console.print("🍏 Using [bold blue]MPS (Apple Silicon)[/bold blue]")
    else:
        device = torch.device("cpu")
        console.print("🖥️ Using [bold red]CPU[/bold red]")
    return device
