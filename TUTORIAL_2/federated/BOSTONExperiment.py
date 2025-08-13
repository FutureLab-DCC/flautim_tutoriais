from flautim.pytorch.federated.Experiment import Experiment
import flautim.metrics as flm
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

    # Definindo métrica
    def mape(y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        mask = y_true != 0
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if np.any(mask) else float('inf')

    # Register the new metric
    flm.Metrics.mape = mape

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


        mape_score = flm.Metrics.mape(torch.cat(y_real).numpy(), torch.cat(yhat).numpy())

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

        mape_score = flm.Metrics.mape(torch.cat(y_real).numpy(), torch.cat(yhat).numpy())

        error_loss = error_loss / len(data_loader)

        return float(error_loss), {'MAPE': mape_value}
