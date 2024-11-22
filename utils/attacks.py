import torch
import cv2
import numpy as np


class BackdoorStrategy(object):
    def __init__(self, trigger_type, triggerX, triggerY):
        self.trigger_type = trigger_type
        self.triggerX = triggerX
        self.triggerY = triggerY

    def add_square_trigger(self, image):
        pixel_max = torch.max(image) if torch.max(image) > 1 else 1
        image[:, self.triggerY : self.triggerY + 5, self.triggerX : self.triggerX + 5] = pixel_max
        return image

    def add_pattern_trigger(self, image):
        pixel_max = torch.max(image) if torch.max(image) > 1 else 1
        image[:, self.triggerY + 0, self.triggerX + 0] = pixel_max
        image[:, self.triggerY + 1, self.triggerX + 1] = pixel_max
        image[:, self.triggerY - 1, self.triggerX + 1] = pixel_max
        image[:, self.triggerY + 1, self.triggerX - 1] = pixel_max
        return image

    def add_watermark_trigger(self, image, watermark_path="./watermarks/watermark.png"):
        watermark = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE)
        watermark = cv2.bitwise_not(watermark)
        watermark = cv2.resize(watermark, dsize=image[0].shape, interpolation=cv2.INTER_CUBIC)
        pixel_max = np.max(watermark)
        watermark = watermark.astype(np.float64) / pixel_max
        pixel_max_dataset = torch.max(image).item() if torch.max(image).item() > 1 else 1
        watermark *= pixel_max_dataset
        max_pixel = max(np.max(watermark), torch.max(image))
        image += watermark
        image[image > max_pixel] = max_pixel
        return image.float()

    def add_backdoor(self, image):
        if self.trigger_type == "pattern":
            image = self.add_pattern_trigger(image)
        elif self.trigger_type == "square":
            image = self.add_square_trigger(image)
        elif self.trigger_type == "watermark":
            image = self.add_watermark_trigger(image)
        return image
    
class LabelFlippingAttack:
    def __init__(self, flip_mapping):
        """
        Initialize the label flipping attack.

        Args:
            flip_mapping (dict): A dictionary specifying the labels to flip.
                                 Example: {0: 1, 1: 0} flips 0 to 1 and 1 to 0.
        """
        self.flip_mapping = flip_mapping

    def apply(self, data_loader):
        """
        Apply label flipping to the dataset.

        Args:
            data_loader (DataLoader): PyTorch DataLoader containing the dataset.

        Returns:
            DataLoader: DataLoader with flipped labels.
        """
        for _, (data, labels) in enumerate(data_loader):
            for i in range(len(labels)):
                if labels[i].item() in self.flip_mapping:
                    labels[i] = self.flip_mapping[labels[i].item()]
        return data_loader
    
    
import torch

class LIEAttack:
    def __init__(self, epsilon=0.1):
        """
        Initialize the LIE attack.

        Args:
            epsilon (float): Magnitude of the perturbation added to the gradients.
        """
        self.epsilon = epsilon

    def apply(self, gradients):
        """
        Apply LIE attack by perturbing the gradients.

        Args:
            gradients (list of torch.Tensor): List of gradients.

        Returns:
            list of torch.Tensor: Perturbed gradients.
        """
        perturbed_gradients = []
        for grad in gradients:
            noise = torch.normal(mean=0, std=self.epsilon, size=grad.shape).to(grad.device)
            perturbed_gradients.append(grad + noise)
        return perturbed_gradients
    
class DBAttack:
    def __init__(self, trigger_function, target_label):
        """
        Initialize the Distributed Backdoor Attack.

        Args:
            trigger_function (callable): Function to add the backdoor trigger to the input data.
            target_label (int): Label to assign for inputs with the trigger.
        """
        self.trigger_function = trigger_function
        self.target_label = target_label

    def apply(self, data_loader):
        """
        Apply the backdoor trigger and target label to the dataset.

        Args:
            data_loader (DataLoader): PyTorch DataLoader containing the dataset.

        Returns:
            DataLoader: DataLoader with backdoor triggers and labels applied.
        """
        for _, (data, labels) in enumerate(data_loader):
            for i in range(len(data)):
                data[i] = self.trigger_function(data[i])
                labels[i] = self.target_label
        return data_loader

class A3FLAttack:
    def __init__(self, aggregation_function):
        """
        Initialize the A3FL attack.

        Args:
            aggregation_function (callable): Function to manipulate the aggregation process.
        """
        self.aggregation_function = aggregation_function

    def apply(self, client_updates):
        """
        Manipulate the aggregation process.

        Args:
            client_updates (list of dict): List of model updates from clients.

        Returns:
            dict: Manipulated global update.
        """
        return self.aggregation_function(client_updates)