from flautim.pytorch.centralized.Experiment import Experiment
import flautim as fl
import numpy as np
import torch
import time

class IRISExperiment(Experiment):
    def __init__(self, model, dataset, context, **kwargs):
        super(IRISExperiment, self).__init__(model, dataset, context, **kwargs)

        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.epochs = kwargs.get('epochs', 30)


    def training_loop(self, data_loader):
        self.model.train()
        error_loss = 0.0
        yhat, y_real = [], []
        
        for X, y in data_loader:
            self.optimizer.zero_grad()
            outputs = self.model(X)
            loss = self.criterion(outputs, y)
            loss.backward()
            self.optimizer.step()

            error_loss += loss.cpu().item()
            _, predicted = torch.max(outputs.data, 1)
            yhat.append(predicted.detach().cpu())
            y_real.append(y.detach().cpu())
            
        accuracy = self.metrics.ACCURACY(torch.cat(yhat).numpy(), torch.cat(y_real).numpy())
        accuracy_2 = self.metrics.ACCURACY_2(torch.cat(yhat).numpy(), torch.cat(y_real).numpy())
        error_loss = error_loss / len(data_loader)
                   
        return float(error_loss), {'ACCURACY': accuracy, 'ACCURACY_2': accuracy_2}

    def validation_loop(self, data_loader):
        error_loss = 0.0
        yhat, y_real = [], []
        self.model.eval()
        
        with torch.no_grad():
            for X, y in data_loader:
                outputs = self.model(X)
                error_loss += self.criterion(outputs, y).item()
                _, predicted = torch.max(outputs.data, 1)
                yhat.append(predicted.detach().cpu())
                y_real.append(y.detach().cpu())
                
        accuracy = self.metrics.ACCURACY(torch.cat(yhat).numpy(), torch.cat(y_real).numpy())
        accuracy_2 = self.metrics.ACCURACY_2(torch.cat(yhat).numpy(), torch.cat(y_real).numpy())
        error_loss = error_loss / len(data_loader)

        return float(error_loss), {'ACCURACY': accuracy, 'ACCURACY_2': accuracy_2}
        
    
