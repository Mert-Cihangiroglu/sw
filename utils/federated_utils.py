# utils/federated_utils.py
import torch
import copy

def fed_avg_aggregation(local_models):
    """Performs FedAvg aggregation on a list of local models."""
    global_state_dict = local_models[0].state_dict()
    for key in global_state_dict.keys():
        global_state_dict[key] = torch.mean(torch.stack([model.state_dict()[key].float() for model in local_models]), dim=0)
    global_model = copy.deepcopy(local_models[0])
    global_model.load_state_dict(global_state_dict)
    return global_model

def weighted_masked_aggregation(local_models, masks):
    """
    Perform weighted aggregation of masked local_models.
    
    Args:
        local_models (list of nn.Module): List of pruned local_models to aggregate.
        masks (list of dict): Corresponding masks for each model.
    
    Returns:
        nn.Module: Aggregated global model.
    """
    # Initialize the aggregated model as a clone of the first model
    aggregated_model = copy.deepcopy(local_models[0])
    
    # Compute the importance (weight) of each mask
    mask_weights = []
    for mask in masks:
        total_importance = sum(torch.sum(layer_mask).item() for layer_mask in mask.values())
        mask_weights.append(total_importance)
    
    # Normalize the mask weights
    total_weight = sum(mask_weights)
    normalized_weights = [w / total_weight for w in mask_weights]
    
    # Aggregate parameters using normalized weights
    with torch.no_grad():
        for name, param in aggregated_model.named_parameters():
            aggregated_param = torch.zeros_like(param)
            for model, mask, weight in zip(local_models, masks, normalized_weights):
                if name in mask:
                    model_param = model.state_dict()[name]
                    aggregated_param += weight * model_param * mask[name]
                else:
                    # If no mask, use regular FedAvg for this layer
                    aggregated_param += weight * model.state_dict()[name]
            param.data.copy_(aggregated_param)

    return aggregated_model