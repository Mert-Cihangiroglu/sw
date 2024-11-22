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