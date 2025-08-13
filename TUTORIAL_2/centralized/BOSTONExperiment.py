from flautim.pytorch.centralized.Experiment import Experiment
import flautim as fl
import numpy as np
import torch
import time
import torchmetrics

class BostonExperiment(Experiment):
    def __init__(self, model, dataset, context, **kwargs):
        super(BostonExperiment, self).__init__(model, dataset, context, **kwargs)

        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.epochs = kwargs.get('epochs', 30)
        self.mape = torchmetrics.SymmetricMeanAbsolutePercentageError().to(self.context.device)

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
            yhat.append(outputs.detach().cpu())
            y_real.append(y.detach().cpu())

        yhat_tensor = torch.cat(yhat)
        y_real_tensor = torch.cat(y_real)
        mape_value = self.mape(yhat_tensor, y_real_tensor).item()

        error_loss = error_loss / len(data_loader)

        return float(error_loss), {'MAPE': mape_value}

    def validation_loop(self, data_loader):
        self.model.eval()
        error_loss = 0.0
        yhat, y_real = [], []

        with torch.no_grad():
            for X, y in data_loader:
                outputs = self.model(X)
                error_loss += self.criterion(outputs, y).item()
                yhat.append(outputs.detach().cpu())
                y_real.append(y.detach().cpu())

        yhat_tensor = torch.cat(yhat)
        y_real_tensor = torch.cat(y_real)
        mape_value = self.mape(yhat_tensor, y_real_tensor).item()

        error_loss = error_loss / len(data_loader)

        return float(error_loss), {'MAPE': mape_value}
