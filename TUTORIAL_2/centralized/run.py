import flautim as fl
import BOSTONDataset, BOSTONModel, BOSTONExperiment
import pandas as pd
import numpy as np
import flautim.metrics as flm
import torch
import torchmetrics

if __name__ == '__main__':
    # Inicializa o contexto do experimento
    context = fl.init()

    fl.log("Flautim2 inicializado!!!")

    # Carrega e embaralha os dados
    df = pd.read_csv("https://raw.githubusercontent.com/selva86/datasets/refs/heads/master/BostonHousing.csv")
    file = df.astype('float32').sample(frac=1, random_state=42).reset_index(drop=True)

    dataset = BOSTONDataset.BostonDataset(file, batch_size=10, shuffle=False, num_workers=0)
    model = BOSTONModel.BostonModel(context)
    experiment = BOSTONExperiment.BostonExperiment(model, dataset, context)

    # Métrica de acurácia customizada 
    def accuracy_2(y, y_hat):
        y = np.asarray(y)
        y_hat = np.asarray(y_hat)
        return np.mean(np.abs(y - y_hat) < 2.0)

    # Métrica MAPE usando torchmetrics (Simétrica)
    def mape(y, y_hat):
        y = torch.tensor(y)
        y_hat = torch.tensor(y_hat)
        metric = torchmetrics.SymmetricMeanAbsolutePercentageError()
        return metric(y_hat, y).item()

    # Registra as métricas no módulo flautim
    flm.Metrics.accuracy_2 = accuracy_2
    flm.Metrics.mape = mape

    experiment.run(metrics={
        'ACCURACY': flm.Metrics.accuracy,
        'ACCURACY_2': flm.Metrics.accuracy_2,
        'MAPE': flm.Metrics.mape
    })
