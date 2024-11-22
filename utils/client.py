# utils/client.py
import torch

class Client:
    def __init__(self, client_id, data_loader, device):
        self.client_id = client_id
        self.data_loader = data_loader
        self.device = device
        self.model = None  # Local model will be assigned and trained each round
    
    def train(self, model):
        self.model = model.to(self.device)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()
        
        self.model.train()
        for epoch in range(10):  # Adjust the number of local epochs as needed
            for data, target in self.data_loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()