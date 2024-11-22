import torch 
from utils.device import get_device  

class TensorPruner:
    def __init__(self, zip_percent):
        self.thresh_hold = 0.
        self.zip_percent = zip_percent
        self.device = get_device() 

    def update_thresh_hold(self, tensor):
        tensor_copy = tensor.clone().detach()
        tensor_copy = torch.abs(tensor_copy)
        survivial_values = torch.topk(tensor_copy.reshape(1, -1),
                                      int(tensor_copy.reshape(1, -1).shape[1] * self.zip_percent))
        self.thresh_hold = survivial_values[0][0][-1]

    def prune_tensor(self, tensor):
        # Create background_tensor directly on the correct device
        background_tensor = torch.zeros(tensor.shape, device=self.device).to(torch.float)
        #print("background_tensor", background_tensor)

        # Apply the threshold to prune the tensor
        pruned_tensor = torch.where(torch.abs(tensor) > self.thresh_hold, tensor, background_tensor)
        #print("tensor:", pruned_tensor)

        return pruned_tensor