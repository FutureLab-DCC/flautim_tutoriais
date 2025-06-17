from flautim.pytorch.federated.Experiment import Experiment
import flautim as fl
import flautim.metrics as flm
import numpy as np
import torch
import time

class MNISTExperiment(Experiment):
    def __init__(self, model, dataset, context, **kwargs):
        super(MNISTExperiment, self).__init__(model, dataset, context, **kwargs)
	
        self.criterion = torch.nn.CrossEntropyLoss()
        self.lr = kwargs.get('lr', 0.01)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.epochs = kwargs.get('epochs', 30)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    def training_loop(self, data_loader):
    
        self.model.to(self.device)
        self.model.train()
        
        correct, running_loss = 0.0, 0.0
        
        for batch in data_loader:
            images = batch["image"]
            labels = batch["label"]
            self.optimizer.zero_grad()
            
            outputs = self.model(images.to(self.device))
            loss = self.criterion(outputs, labels.to(self.device))
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            correct += (torch.max(outputs, 1)[1].cpu() == labels.cpu()).sum().item()

        accuracy = correct / len(data_loader.dataset)    
        avg_trainloss = running_loss / len(data_loader)           
        return float(avg_trainloss), {'ACCURACY': accuracy}

    def validation_loop(self, data_loader):
        
        self.model.to(self.device)
        self.model.eval()
        
        correct, loss = 0, 0.0
        with torch.no_grad():
            for batch in data_loader:
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)
                outputs = self.model(images)
                loss += self.criterion(outputs, labels).item()
                correct += (torch.max(outputs, 1)[1].cpu() == labels.cpu()).sum().item()
        
        accuracy = correct / len(data_loader.dataset)
        loss = loss / len(data_loader)

        return float(loss), {'ACCURACY': accuracy}
