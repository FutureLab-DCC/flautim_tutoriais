from flautim.pytorch.Model import Model
import torch
import torch.nn as nn
import torch.nn.functional as F

class PoCModel_2HiddenLayers(Model):
    """
    MLP Profundo com DUAS CAMADAS OCULTAS para FMNIST (784 -> 256 -> 128 -> 10)
    Modelo como descrito no paper de referência de Power of Choice
    """
    def __init__(self, context, num_classes: int, **kwargs) -> None:
        super(PoCModel_2HiddenLayers, self).__init__(context, name = "2HiddenLayersNN", version = 1, id = 1, **kwargs)

        self.hidden1 = nn.Linear(784, 256)
        self.hidden2 = nn.Linear(256, 128)
        
        self.output_layer = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1) # Flatten da imagem (28x28 -> 784)
        
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        
        x = self.output_layer(x)
        return x