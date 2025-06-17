import flautim as fl
import flautim.metrics as flm
import MNISTDataset, MNISTModel, MNISTExperiment
import torch
import numpy as np
import pandas as pd
from datasets import load_dataset

from torchvision.transforms import (
    Compose,
    Normalize,
    RandomCrop,
    RandomHorizontalFlip,
    ToTensor,
)


FM_NORMALIZATION = ((0.1307,), (0.3081,))
EVAL_TRANSFORMS = Compose([ToTensor(), Normalize(*FM_NORMALIZATION)])
TRAIN_TRANSFORMS = Compose(
    [
        RandomCrop(28, padding=4),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(*FM_NORMALIZATION),
    ]
)

DATASET="ylecun/mnist"

if __name__ == '__main__':

    context = fl.init()

    fl.log(f"Flautim inicializado!!!")
    
    partition = load_dataset("zalando-datasets/fashion_mnist")

    dataset = MNISTDataset.MNISTDataset(FM_NORMALIZATION, EVAL_TRANSFORMS, TRAIN_TRANSFORMS, partition, split_data=False, batch_size = 32, shuffle = False, num_workers = 0)

    model = MNISTModel.MNISTModel(context, num_classes = 10)

    fl.log(f"Modelo criado!!!")

    experiment = MNISTExperiment.MNISTExperiment(model, dataset,  context)

    fl.log(f"Experimento criado!!!")

    # Exemplo de métrica implementada pelo usuário
    def accuracy_2(y, y_hat):
        y = np.asarray(y)
        y_hat = np.asarray(y_hat)
        return np.mean(y == y_hat)

    # Adiciona a métrica ao módulo de métricas
    flm.Metrics.accuracy_2 = accuracy_2

    fl.log(f"Métrica adicionada!!!")

    experiment.run(metrics = {'ACCURACY': flm.Metrics.accuracy, 'ACCURACY_2': flm.Metrics.accuracy_2})
    
