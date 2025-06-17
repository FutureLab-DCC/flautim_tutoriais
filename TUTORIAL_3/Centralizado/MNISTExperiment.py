from flautim.pytorch.centralized.Experiment import Experiment
import flautim as fl
import flautim.metrics as flm
import numpy as np
import torch
import time

class MNISTExperiment(Experiment):
    def __init__(self, model, dataset, context, **kwargs):
        super(MNISTExperiment, self).__init__(model, dataset, context, **kwargs)
	
        self.criterion = torch.nn.CrossEntropyLoss()
        self.lr = kwargs.get('lr', 0.001)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.epochs = kwargs.get('epochs', 30)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    def training_loop(self, data_loader):
    
        self.model.to(self.device)
        self.model.train()
        
        running_loss = 0.0
        
        yhat, y_real = [], []
        
        for batch in data_loader:
            images = batch["image"]
            labels = batch["label"]
            self.optimizer.zero_grad()
            
            outputs = self.model(images.to(self.device))
            loss = self.criterion(outputs, labels.to(self.device))
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            yhat.append(predicted.detach().cpu())
            y_real.append(labels.detach().cpu())

        accuracy = self.metrics.ACCURACY(torch.cat(yhat).numpy(), torch.cat(y_real).numpy())
        accuracy_2 = self.metrics.ACCURACY_2(torch.cat(yhat).numpy(), torch.cat(y_real).numpy())
        avg_trainloss = running_loss / len(data_loader)           
        return float(avg_trainloss), {'ACCURACY': accuracy, 'ACCURACY_2': accuracy_2}

    def validation_loop(self, data_loader):
        
        self.model.to(self.device)
        self.model.eval()
        
        yhat, y_real = [], []
        
        loss = 0.0
        with torch.no_grad():
            for batch in data_loader:
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)
                outputs = self.model(images)
                loss += self.criterion(outputs, labels).item()                
                _, predicted = torch.max(outputs.data, 1)
                yhat.append(predicted.detach().cpu())
                y_real.append(labels.detach().cpu())
        
        accuracy = self.metrics.ACCURACY(torch.cat(yhat).numpy(), torch.cat(y_real).numpy())
        accuracy_2 = self.metrics.ACCURACY_2(torch.cat(yhat).numpy(), torch.cat(y_real).numpy())
        loss = loss / len(data_loader)

        return float(loss), {'ACCURACY': accuracy, 'ACCURACY_2': accuracy_2}
