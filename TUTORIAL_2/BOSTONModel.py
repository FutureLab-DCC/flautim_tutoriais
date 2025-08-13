from flautim.pytorch.Model import Model
import torch

class BostonModel(Model):
    def __init__(self, context, num_classes = 13, **kwargs):
        super(BostonModel, self).__init__(context, name = "BOSTON-NN", **kwargs)

        # Rede neural com 13 entradas e 1 saída
        self.c1 = torch.nn.Linear(13, 10)
        self.c2 = torch.nn.Linear(10, 5)
        self.c3 = torch.nn.Linear(5, 1)


    def forward(self, x):
        x = torch.relu(self.c1(x))
        x = torch.relu(self.c2(x))
        x = torch.relu(self.c3(x))
        return x