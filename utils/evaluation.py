import torch
import os
import json
import matplotlib.pyplot as plt
from rich.console import Console
from sklearn.metrics import classification_report


# Initialize Rich Console
console = Console()


def evaluate_model(model, data_loader, device):
    """
    Evaluate the model on the provided data loader and return the average loss and accuracy.
    
    Args:
        model (torch.nn.Module): The model to evaluate.
        data_loader (torch.utils.data.DataLoader): DataLoader for the validation or test set.
        device (torch.device): Device to perform evaluation on (CPU, GPU, or MPS).
        
    Returns:
        tuple: (average_loss, accuracy) of the model on the dataset.
    """
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item() * data.size(0)  # Sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # Get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    average_loss = total_loss / total
    accuracy = correct / total
    return average_loss, accuracy


def evaluate_model_with_class_metrics(model, data_loader, device):
    """
    Evaluate the model and calculate class-specific precision, recall, F1-score, and accuracy.
    """
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data, labels in data_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * data.size(0)
            
            # Predictions
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(data_loader.dataset)
    # Generate classification report
    class_report = classification_report(
        all_labels, 
        all_preds, 
        output_dict=True, 
        zero_division=0
    )
    overall_accuracy = class_report["accuracy"]
    
    # Print class-specific metrics
    for class_id, metrics in class_report.items():
        if isinstance(metrics, dict):  # Exclude "accuracy" key
            precision, recall, f1, support = metrics["precision"], metrics["recall"], metrics["f1-score"], metrics["support"]
            console.print(f"[blue]Class {class_id} - Precision: {precision:.2f}, Recall: {recall:.2f}, F1-score: {f1:.2f}, Support: {support}[/blue]")


    return avg_loss, overall_accuracy, class_report