from flautim.pytorch.federated.Experiment import Experiment
import flautim as fl
import flautim.metrics as flm
import numpy as np
import torch
import time

from collections import OrderedDict
from sklearn.metrics import f1_score

from math import inf

from random import random

# Two auxhiliary functions to set and extract parameters of a model
def set_params(model, parameters):
    """Replace model parameters with those passed as `parameters`."""

    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


def get_params(model):
    """Extract model parameters as a list of NumPy arrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

class PoCExperimentFMNIST(Experiment):
    """
    Experimento baseline para o MNIST
    Baseado no tutorial do flautim e definições prévias.
    Utiliza o modelo BaselineModelMNIST e o dataset MNISTDataset (do tutorial).
    """

    def __init__(self, model, dataset, context, **kwargs):
        super(PoCExperimentFMNIST, self).__init__(model, dataset, context, **kwargs)
        
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        self.epochs = kwargs.get('epochs', 1)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.last_loss = inf
        self.last_participation_round = 0
        self.model = model
        self.data_size = len(dataset.train_partition) # Store the size of the local training data

    # --------------------------------
    # Ref: https://discuss.flower.ai/t/custom-client-selection-strategy/63
    def fit(self, parameters, config):
        set_params(self.model, parameters)
        epochs = config.get("epochs", self.epochs) # Get the number of epochs from config
        if epochs == -1:
            return parameters, 0, {"data_size": self.data_size}  # Just return client's data size

        if epochs == 0:
            # Estimate local loss without training the model
            local_loss, _ = self.validation_loop(self.dataset.dataloader(validation=True))
            local_loss += np.random.uniform(low=1e-10, high=1e-9) # Make sure that potential ties are broken at random
            return parameters, 0, {"local_loss": local_loss}  # Return the estimated local loss without updating the parameters

        # Train the model and return the updated parameters
        # self.fit(self.x_train, self.y_train, validation_data=(self.x_test, self.y_test), epochs = self.epochs, verbose=0)
        loss, metrics = self.training_loop(self.dataset.dataloader())
        return get_params(self.model), self.data_size, metrics
    # --------------------------------

    def training_loop(self, data_loader):
        """This method trains the model using the parameters sent by the
        server on the dataset of this client. At then end, the parameters
        of the locally trained model are communicated back to the server"""
        
        self.model.to(self.device)
        self.model.train()
        
        correct, running_loss = 0.0, 0.0

        criterion = torch.nn.CrossEntropyLoss()
        self.model.to(self.device)
        self.model.train()
        for batch in data_loader:
            images, labels = batch["image"].to(self.device), batch["label"].to(self.device)
            self.optimizer.zero_grad()
        
            outputs = self.model(images.to(self.device))
            loss = criterion(self.model(images), labels)
            
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            correct += (torch.max(outputs, 1)[1].cpu() == labels.cpu()).sum().item()

        accuracy = correct / len(data_loader.dataset)    
        avg_trainloss = running_loss / len(data_loader)
        
        self.last_loss = avg_trainloss           
        
        return float(avg_trainloss), {'ACCURACY': accuracy}


    def validation_loop(self, data_loader):
        """Evaluate the model sent by the server on this client's
        local validation set. Then return performance metrics."""

        all_predictions = []
        all_labels = []

        criterion = torch.nn.CrossEntropyLoss()
        correct, loss = 0, 0.0

        self.model.to(self.device)

        self.model.eval()
        with torch.no_grad():
            for batch in data_loader:
                images, labels = batch["image"].to(self.device), batch["label"].to(self.device)
                outputs = self.model(images)
                loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()

                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = correct / len(data_loader.dataset)
        f1 = f1_score(all_labels, all_predictions, average='macro')

        return float(loss), {"accuracy": accuracy, "f1": f1}
    