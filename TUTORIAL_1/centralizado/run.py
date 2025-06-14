import flautim as fl
import IRISDataset, IRISModel, IRISExperiment
import numpy as np
import pandas as pd
import flautim.metrics as flm


if __name__ == '__main__':

    context = fl.init()

    fl.log(f"Flautim2 inicializado!!!")
    
    # Carregue os dados usando dataset próprio
    iris = pd.read_csv("./data/iris.csv", header=None)

    # Carregue os dados usando uma URL
    #iris = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header=None)
    iris.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
    iris['class'] = pd.factorize(iris['class'])[0]

    # Embaralhe os dados
    file = iris.sample(frac=1, random_state=42).reset_index(drop=True)
    
    dataset = IRISDataset.IRISDataset(file, batch_size = 10, shuffle = False, num_workers = 0)

    model = IRISModel.IRISModel(context)

    experiment = IRISExperiment.IRISExperiment(model, dataset, context)

    # Exemplo de métrica implementada pelo usuário
    def accuracy_2(y, y_hat):
        y = np.asarray(y)
        y_hat = np.asarray(y_hat)
        return np.mean(y == y_hat)
    
    # Adiciona a métrica ao módulo de métricas
    flm.Metrics.accuracy_2 = accuracy_2

    experiment.run(metrics = {'ACCURACY': flm.Metrics.accuracy, 'ACCURACY_2': flm.Metrics.accuracy_2})

