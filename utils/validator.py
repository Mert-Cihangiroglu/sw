# utils/validator.py
from utils.tensor_pruner import TensorPruner
from torchvision.models import resnet18
from rich.console import Console
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import models
from copy import deepcopy
import torch.nn as nn
import torch
import copy
import os


console = Console()

class Validator:
    def __init__(self, validator_id, data_loader, device, global_model, zip_percent=0.5):
        self.validator_id = validator_id
        self.data_loader = data_loader
        self.device = device
        self.zip_percent = zip_percent
        self.criterion = nn.CrossEntropyLoss()
        self.class_specific_model = None
        self.mask_path = f"saved_masks/validator_{validator_id}_mask.pth"
        self.pruner = TensorPruner(zip_percent=self.zip_percent)
        self.model_usage_log = {}
        self.global_model = global_model
        self.scale_down_factor=0.2
        
        # Load mask if it exists
        if os.path.exists(self.mask_path):
            self.mask = torch.load(self.mask_path)
            console.print(f"[Validator {self.validator_id}] Loaded saved mask from {self.mask_path}")
        else:
            self.mask = None
    
    def log_model_usage(self, model_id):
        """Log usage of a specific client model."""
        if model_id not in self.model_usage_log:
            self.model_usage_log[model_id] = 0
        self.model_usage_log[model_id] += 1
    
    def get_model_usage_stats(self):
        """Retrieve usage stats for logging or saving."""
        return {k: v for k, v in self.model_usage_log.items()}

    def validate(self, model):
        """Evaluate the model on the validator's dataset and return accuracy."""
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in self.data_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        accuracy = correct / total
        return accuracy

    def get_top_models(self, models, top_k=3):
        """Evaluate all models and select the top K models based on accuracy."""
        model_scores = {}
        for model_id, model in enumerate(models):
            accuracy = self.validate(model)
            model_scores[model_id] = accuracy

        # Sort models by accuracy and select the top K models
        sorted_models = sorted(model_scores, key=model_scores.get, reverse=True)
        top_models = [models[i] for i in sorted_models[:top_k]]
        return top_models

    def get_top_model(self, models):
        """Evaluate all models and select the top-performing one based on accuracy."""
        model_scores = {}
        for model_id, model in enumerate(models):
            accuracy = self.validate(model)
            model_scores[model_id] = accuracy

        # Find the model with the highest accuracy
        top_model_id = max(model_scores, key=model_scores.get)
        top_model = models[top_model_id]
        top_accuracy = model_scores[top_model_id]

        # Print which model was selected
        console.print(f"[Validator {self.validator_id}] Selected top model from Client {top_model_id} with accuracy: {top_accuracy:.2f}")

        return top_model, top_model_id

    def prune_and_average_models(self, top_models):
        """Apply TensorPruner on top models and average the pruned weights."""
        pruner = TensorPruner(zip_percent=self.zip_percent)
        averaged_state_dict = {}
        
        #print(f"[Validator {self.validator_id}] Starting prune_and_average_models")

        # Apply pruning on each parameter layer
        for name, param in top_models[0].state_dict().items():
            pruned_tensors = []
            
            
            
            #print(f"Processing layer '{name}'")
            #print(f"Original parameter shape: {param.shape}")

            for idx, model in enumerate(top_models):
                #print(f"  [Model {idx}] Processing model parameter layer '{name}'")
                
                param_tensor = model.state_dict()[name].clone().detach()
                
                # Debugging information for param_tensor
                #print(f"  Parameter tensor shape: {param_tensor.shape}")
                #print(f"  Parameter tensor device: {param_tensor.device}")
                #print(f"  Parameter tensor data type: {param_tensor.dtype}")

                # Skip non-floating point tensors (e.g., num_batches_tracked)
                if param_tensor.dtype not in (torch.float32, torch.float64):
                    #print(f"Skipping non-floating point tensor for layer {name}: {param_tensor}")
                    averaged_state_dict[name] = param_tensor  # Directly set from one model
                    continue

                # Check if the tensor is a scalar (for biases)
                if param_tensor.dim() == 0:
                    pruned_tensors.append(param_tensor)  # Directly add scalar bias without pruning
                    continue
                #self.visualize_weights(param, f"Original Weights for layer '{name}'")
                # Update threshold and prune tensor
                pruner.update_thresh_hold(param_tensor)
                pruned_tensor = pruner.prune_tensor(param_tensor)
                pruned_tensors.append(pruned_tensor)
                #print(f"  Appended pruned tensor for model {idx} with shape {pruned_tensor.shape}")
                #self.visualize_weights(pruned_tensor, f"Pruned Weights for model {idx} layer '{name}'")


            # Check if pruned_tensors list is empty
            if not pruned_tensors:
                print(f"[ERROR] No pruned tensors to stack for layer '{name}'")
                continue

            # Stack and average pruned tensors
            try:
                averaged_state_dict[name] = torch.stack(pruned_tensors).mean(dim=0)
                #print(f"  Averaged pruned tensor for layer '{name}' with shape {averaged_state_dict[name].shape}")
                #self.visualize_weights(averaged_state_dict[name], f"Averaged Pruned Weights for layer '{name}'")
            except Exception as e:
                print(f"[ERROR] Could not stack or average pruned tensors for layer '{name}': {e}")

        # Create a new model with averaged pruned weights
        averaged_model = copy.deepcopy(top_models[0])
        averaged_model.load_state_dict(averaged_state_dict)
        return averaged_model
      
    def initialize_resnet(self):
        """Initialize a ResNet model with 10 classes (full architecture)."""
        model = resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, 10)
        return model.to(self.device)
    
    def train_class_specific_model(self, epochs=5):
        """Train a model specific to the validator's class using only relevant data."""
        self.class_specific_model = self.initialize_resnet()
        self.class_specific_model.train()
        
        # Initialize optimizer here
        optimizer = torch.optim.SGD(self.class_specific_model.parameters(), lr=0.01)

        for epoch in range(epochs):
            running_loss = 0.0
            for data, target in self.data_loader:  # Validator's class-specific data
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()  # Reset gradients
                output = self.class_specific_model(data)
                loss = self.criterion(output, target)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f"[Validator {self.validator_id}] Epoch [{epoch+1}/{epochs}], Loss: {running_loss / len(self.data_loader)}")

    def create_weighted_mask(self):
        """Generate a mask highlighting important weights in each layer."""
        pruner = TensorPruner(zip_percent=self.zip_percent)
        mask = {}

        for name, param in self.class_specific_model.state_dict().items():
            if param.requires_grad:  # Only consider layers with gradients
                pruner.update_thresh_hold(param)
                mask[name] = (pruner.prune_tensor(param) != 0).float()
        
        return mask
    
    def apply_mask_to_model(self, model, mask):
        """Apply the generated mask to prune weights in the model."""
        pruned_model = copy.deepcopy(model)
        
        with torch.no_grad():
            for name, param in pruned_model.state_dict().items():
                if name in mask:
                    param.mul_(mask[name])  # Element-wise multiplication to keep only important weights

        return pruned_model
        
    def generate_and_save_mask(self):
        """Generate a mask for the class-specific model and save it."""
        self.mask = {}
        for name, param in self.class_specific_model.state_dict().items():
            if param.dtype in (torch.float32, torch.float64):
                self.pruner.update_thresh_hold(param)
                pruned_tensor = self.pruner.prune_tensor(param)
                self.mask[name] = (pruned_tensor != 0).float()  # Create binary mask

        # Ensure the directory exists
        os.makedirs(os.path.dirname(self.mask_path), exist_ok=True)
        torch.save(self.mask, self.mask_path)
        console.print(f"[Validator {self.validator_id}] Mask generated and saved at {self.mask_path}") 
        return self.mask
    
    def generate_and_save_weighted_mask(self, scale_down_factor=0.5):
        """
        Generate a weighted mask for the class-specific model that scales down 
        less important weights instead of zeroing them out.
        
        Args:
            scale_down_factor (float): Factor by which to scale down less important weights.
        """
        self.mask = {}
        for name, param in self.class_specific_model.state_dict().items():
            if param.dtype in (torch.float32, torch.float64):
                # Identify threshold for important weights
                self.pruner.update_thresh_hold(param)
                threshold = self.pruner.thresh_hold
                
                # Create mask that retains full value for important weights
                # and scales down less important ones
                importance_mask = (param.abs() >= threshold).float()
                scaled_down_weights = (param.abs() < threshold).float() * scale_down_factor
                
                # Combine the masks: important weights retain their original value,
                # less important ones are scaled down
                self.mask[name] = importance_mask + scaled_down_weights
            
        # Ensure the directory exists
        os.makedirs(os.path.dirname(self.mask_path), exist_ok=True)
        torch.save(self.mask, self.mask_path)
        console.print(f"[Validator {self.validator_id}] Scaled mask generated and saved at {self.mask_path}")
        
        return self.mask
        # (1) See if you can finetune the pretrained model.

    def generate_mask_from_gm(self):
        self.mask = {}
        pruner = TensorPruner(zip_percent=self.zip_percent)
        outputs = torch.tensor([], device=self.device)
        targets = torch.tensor([], device=self.device)
        
        for data, target in self.data_loader:
            data, target = data.to(self.device), target.to(self.device)
            output = self.global_model(data)
            outputs = torch.cat((outputs,output),0)
            targets = torch.cat((targets,target),0)
        loss = self.criterion(outputs,targets)
        
        gradients = torch.autograd.grad(
                loss, self.global_model.parameters(), create_graph=True)
        
        for (name, param), grad in zip(self.global_model.named_parameters(), gradients):
            #print(name, grad)
            pruner.update_thresh_hold(grad)
            self.mask[name] = (pruner.prune_tensor(grad) != 0).float()
            self.mask[name][self.mask[name] == 0] = self.scale_down_factor
            #print(self.mask)
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(self.mask_path), exist_ok=True)
        torch.save(self.mask, self.mask_path)
        console.print(f"[Validator {self.validator_id}] Scaled mask generated and saved at {self.mask_path}")

        return self.mask
    
    def generate_mask_from_lm(self, lm_model):
        self.mask = {}
        pruner = TensorPruner(zip_percent=self.zip_percent)
        outputs = torch.tensor([], device=self.device)
        targets = torch.tensor([], device=self.device)
        
        for data, target in self.data_loader:
            data, target = data.to(self.device), target.to(self.device)
            output = lm_model(data)
            outputs = torch.cat((outputs,output),0)
            targets = torch.cat((targets,target),0)
        loss = self.criterion(outputs,targets)
        
        gradients = torch.autograd.grad(
                loss, lm_model.parameters(), create_graph=True)
        
        for (name, param), grad in zip(lm_model.named_parameters(), gradients):
            #print(name, grad)
            pruner.update_thresh_hold(grad)
            self.mask[name] = (pruner.prune_tensor(grad) != 0).float()
            self.mask[name][self.mask[name] == 0] = self.scale_down_factor
            #print(self.mask)
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(self.mask_path), exist_ok=True)
        torch.save(self.mask, self.mask_path)
        console.print(f"[Validator {self.validator_id}] Scaled mask generated and saved at {self.mask_path}")

        return self.mask
        
    def generate_class_specific_mask(self, model, freeze_early_layers=True, scale_down_factor=0.5):
        """
        Generates a mask that retains only the neurons activated by the target class.
        
        Args:
            model (nn.Module): The top model to forward and backpropagate.
            freeze_early_layers (bool): If True, only the last few layers will be unfrozen during backpropagation.
            scale_down_factor (float): Factor by which to scale down less important weights instead of zeroing them out.
        
        Returns:
            mask (dict): A dictionary containing the mask for each layer.
        """
        console.print(f"[Validator {self.validator_id}] Generating class-specific mask for class {self.validator_id}")

        # Clone the model to avoid modifying the original
        class_specific_model = deepcopy(model).to(self.device)
        class_specific_model.eval()

        # Freeze early layers if specified (only allowing `layer4` and `fc` layers to update)
        if freeze_early_layers:
            for name, param in class_specific_model.named_parameters():
                if not name.startswith("layer4") and not name.startswith("fc"):
                    param.requires_grad = False

        # Capture gradients and activations for layers of interest
        activations, gradients = {}, {}

        def forward_hook(module, input, output):
            activations[module] = output.detach()

        def backward_hook(module, grad_in, grad_out):
            gradients[module] = grad_out[0].detach()

        # Register hooks for `layer4` and `fc`
        for name, layer in class_specific_model.named_modules():
            if "layer4" in name or "fc" in name:
                layer.register_forward_hook(forward_hook)
                layer.register_full_backward_hook(backward_hook)

        # Run through all target class data in the validator’s data loader
        for inputs, labels in self.data_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            # Forward pass
            outputs = class_specific_model(inputs)
            loss = F.cross_entropy(outputs, labels)

            # Backward pass
            class_specific_model.zero_grad()
            loss.backward()

        console.print(f"[Validator {self.validator_id}] Forward and backward pass completed for class {self.validator_id}")

        # Generate mask based on gradients collected
        mask = {}
        for layer, grad in gradients.items():
            # Threshold to determine important neurons (non-zero gradient regions)
            mask[layer] = (grad != 0).float() + (grad == 0).float() * scale_down_factor

        return mask

    def apply_class_specific_mask(self, model, mask):
        """
        Applies a generated class-specific mask to a model.
        
        Args:
            model (nn.Module): The model to which the mask will be applied.
            mask (dict): A dictionary containing the mask for each layer.
        
        Returns:
            pruned_model (nn.Module): A copy of the input model with the mask applied.
        """
        console.print(f"[Validator {self.validator_id}] Applying class-specific mask to model.")
        
        # Copy model to apply mask without modifying original
        pruned_model = deepcopy(model).to(self.device)
        
        with torch.no_grad():
            for name, param in pruned_model.named_parameters():
                # Apply mask only if it's in the mask dictionary
                if name in mask:
                    param.data *= mask[name].to(self.device)
        
        return pruned_model
    
    def generate_mask_with_gradcam(self, model,scale_down_factor, target_layer, ):
        """
        Generate a weighted mask using Grad-CAM relevance scores for the validator's specific class.

        Args:
            model (nn.Module): The model to analyze.
            target_layer (str): The layer to focus on for Grad-CAM.
            scale_down_factor (float): Factor to scale down less relevant weights.

        Returns:
            dict: Mask dictionary for each layer.
        """
        console.print(f"[Validator {self.validator_id}] Generating mask using Grad-CAM on layer {target_layer}.")

        # Step 1: Clone model and set it to evaluation mode
        gradcam_model = deepcopy(model).to(self.device)
        gradcam_model.eval()

        # Grad-CAM data storage
        activations, gradients = None, None

        # Forward hook to capture activations
        def forward_hook(module, input, output):
            nonlocal activations
            activations = output.detach()

        # Backward hook to capture gradients
        def backward_hook(module, grad_input, grad_output):
            nonlocal gradients
            gradients = grad_output[0].detach()

        # Register hooks on the target layer
        target_module = dict(gradcam_model.named_modules())[target_layer]
        target_module.register_forward_hook(forward_hook)
        target_module.register_full_backward_hook(backward_hook)

        # Step 2: Compute Grad-CAM relevance for the validator's specific class
        relevance_map = {name: torch.zeros_like(param) for name, param in model.named_parameters()}

        for inputs, _ in self.data_loader:  # Only need inputs since all labels are of the same class
            inputs = inputs.to(self.device)

            # Forward pass
            outputs = gradcam_model(inputs)

            # Compute loss for the validator's specific class
            loss = outputs[:, self.validator_id].sum()  # Only focus on validator's class

            # Backward pass
            gradcam_model.zero_grad()
            loss.backward()

            # Compute Grad-CAM relevance
            weights = gradients.mean(dim=(2, 3), keepdim=True)  # Global average pooling of gradients
            cam = F.relu((weights * activations).sum(dim=1))  # Weighted sum of activations

            # Normalize CAM to [0, 1]
            cam = (cam - cam.min()) / (cam.max() - cam.min())

            # Update relevance for model parameters
            for name, param in gradcam_model.named_parameters():
                relevance_map[name] += cam.mean().item() * torch.abs(param)  # Weighted by CAM score

        # Step 3: Normalize relevance map across all samples
        max_relevance = {name: relevance.max() for name, relevance in relevance_map.items()}
        for name, relevance in relevance_map.items():
            relevance_map[name] /= max_relevance[name]

        # Step 4: Create mask based on relevance
        self.mask = {}
        for name, relevance in relevance_map.items():
            # Retain weights with relevance > 0.5, scale down others
            self.mask[name] = (relevance > 0.5).float() + (relevance <= 0.5).float() * scale_down_factor

        # Step 5: Save the mask
        os.makedirs(os.path.dirname(self.mask_path), exist_ok=True)
        torch.save(self.mask, self.mask_path)
        console.print(f"[Validator {self.validator_id}] Mask using Grad-CAM saved at {self.mask_path}")

        return self.mask