from flautim.pytorch.common import run_federated, weighted_average
from flautim.pytorch import Model, Dataset
from flautim.pytorch.federated import Experiment
import BOSTONDataset, BOSTONModel, BOSTONExperiment
import flautim as fl
import flwr
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerConfig, ServerAppComponents
import pandas as pd
import numpy as np


def fit_config(server_round: int):
    """Return training configuration dict for each round.

    Perform two rounds of training with one local epoch, increase to two local
    epochs afterwards.
    """
    config = {
        "server_round": server_round,  # The current round of federated learning
    }
    return config



def generate_server_fn(context, eval_fn, **kwargs):

    def create_server_fn(context_flwr:  Context):

        net = BOSTONModel.BostonModel(context, suffix = 0)
        params = ndarrays_to_parameters(net.get_parameters())

        strategy = flwr.server.strategy.FedAvg(
                          initial_parameters=params,
                          evaluate_metrics_aggregation_fn=weighted_average,
                          fraction_fit=0.2,  # 10% clients sampled each round to do fit()
                          fraction_evaluate=0.5,  # 50% clients sample each round to do evaluate()
                          evaluate_fn=eval_fn,
                          on_fit_config_fn = fit_config,
                          on_evaluate_config_fn = fit_config
                          )
        num_rounds = 20
        config = ServerConfig(num_rounds=num_rounds)

        return ServerAppComponents(config=config, strategy=strategy)
    return create_server_fn

def generate_client_fn(context, files):

    def create_client_fn(context_flwr:  Context):

        cid = int(context_flwr.node_config["partition-id"])
        file = int(cid)
        model = BOSTONModel.BostonModel(context, suffix = cid)
        dataset = BOSTONDataset.BostonDataset(files[file], batch_size = 10, shuffle = False, num_workers = 0)

        return BOSTONExperiment.BostonExperiment(model, dataset, context).to_client()

    return create_client_fn


def evaluate_fn(context, files):
    def fn(server_round, parameters, config):
        """This function is executed by the strategy it will instantiate
        a model and replace its parameters with those from the global model.
        The, the model will be evaluate on the test set (recall this is the
        whole MNIST test set)."""

        model = BOSTONModel.BostonModel(context, suffix = "FL-Global")
        model.set_parameters(parameters)

        dataset = BOSTONDataset.BostonDataset(files[0], batch_size = 10, shuffle = False, num_workers = 0)

        experiment = BOSTONExperiment.BostonExperiment(model, dataset, context)

        config["server_round"] = server_round

        loss, _, return_dic = experiment.evaluate(parameters, config)

        return loss, return_dic

    return fn

if __name__ == '__main__':

    context = fl.init()
    fl.log(f"Flautim2 inicializado!!!")

    num_clientes = 2

    boston = pd.read_csv("https://raw.githubusercontent.com/selva86/datasets/refs/heads/master/BostonHousing.csv")
    files = np.random.permutation(boston.astype('float32'))

    client_fn_callback = generate_client_fn(context, files)
    evaluate_fn_callback = evaluate_fn(context, files)
    server_fn_callback = generate_server_fn(context, eval_fn = evaluate_fn_callback)

    run_federated(client_fn_callback, server_fn_callback)