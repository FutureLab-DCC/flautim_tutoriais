from flautim.pytorch.federated.Experiment import Experiment
import flautim.metrics as flm
import numpy as np
import time
import torch

class IRISExperiment(Experiment):
    def __init__(self, model, dataset, context, **kwargs):
        super(IRISExperiment, self).__init__(model, dataset, context, **kwargs)

        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.epochs = kwargs.get('epochs', 20)
    
    # Exemplo de métrica implementada pelo usuário
    def accuracy_2(y, y_hat):
        y = np.asarray(y)
        y_hat = np.asarray(y_hat)
        return np.mean(y == y_hat)
    
    # Adiciona a métrica ao módulo de métricas
    flm.Metrics.accuracy_2 = accuracy_2
    
    
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

            error_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            yhat.append(predicted.detach().cpu())
            y_real.append(y.detach().cpu())
            
        accuracy = flm.Metrics.accuracy(torch.cat(yhat).numpy(), torch.cat(y_real).numpy())
        accuracy_2 = flm.Metrics.accuracy_2(torch.cat(yhat).numpy(), torch.cat(y_real).numpy())
        error_loss = error_loss / len(data_loader)
        return error_loss, {"ACCURACY": accuracy, "ACCURACY_2": accuracy_2}

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
                
        accuracy = flm.Metrics.accuracy(torch.cat(yhat).numpy(), torch.cat(y_real).numpy())
        accuracy_2 = flm.Metrics.accuracy_2(torch.cat(yhat).numpy(), torch.cat(y_real).numpy())
        error_loss = error_loss / len(data_loader)
        return error_loss, {"ACCURACY": accuracy, "ACCURACY_2": accuracy_2}
        
