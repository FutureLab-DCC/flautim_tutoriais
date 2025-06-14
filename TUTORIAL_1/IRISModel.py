from flautim.pytorch.Model import Model
import torch

class IRISModel(Model):
    def __init__(self, context, num_classes = 3, **kwargs):
        super(IRISModel, self).__init__(context, name = "IRIS-NN", **kwargs)
        
        # Rede neural com 4 entradas e 3 saídas
        self.c1 = torch.nn.Linear(4, 10)
        self.c2 = torch.nn.Linear(10, num_classes)
        

    def forward(self, x):
        x = torch.relu(self.c1(x))
        x = torch.relu(self.c2(x))
        return x


