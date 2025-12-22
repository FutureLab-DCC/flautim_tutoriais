from flautim.pytorch.common import run_federated, weighted_average
from flautim.pytorch import Model, Dataset
from flautim.pytorch.federated import Experiment
import FMNISTDataset, PoCModel_2HiddenLayers, PoCExperimentFMNIST
from PoCExperimentFMNIST import get_params
import flautim as fl
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
from flwr.common import Context, ndarrays_to_parameters
import flwr
import pandas as pd
import numpy as np
from flwr.server import ServerConfig, ServerAppComponents
from datasets import load_dataset

from random import random
from flwr.server.client_manager import ClientManager
from flwr.common import Parameters, FitIns
from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy
from time import sleep

NUM_PARTITIONS = 100 # Numero de clientes totais e numero de partições de dados
DATASET = "zalando-datasets/fashion_mnist" 

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

# -------------------------------
# Ref:https://discuss.flower.ai/t/custom-client-selection-strategy/63
class MyStrategy(FedAvg):
    """Behaves just like FedAvg but with a modified sampling.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p = [] # probabilities for each client based on data size?
        self.d = 30 # number of candidate clients to consider each round
        self.m = self.min_fit_clients # number of clients to select each round
        # self.prev_clients = [] # store client proxies from previous round
    
    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager = ClientManager # client_manager alterado!
        ) -> list[tuple[ClientProxy, FitIns]]: 
        """Configure the next round of training."""
        # Get list with all the available clients (K clients)
        available_clients = list(client_manager.clients.values())

        print(f"Round {server_round} - Available clients: {[client.cid for client in available_clients]}")
        
        if self.p == []:
            data_sizes = {
                client.cid: client.fit(
                    FitIns(parameters, {"epochs": -1}), 
                    timeout=60, 
                    group_id=str(client.cid)
                ).metrics.get("data_size", 0)
                for client in available_clients
            }
            
            for key, value in data_sizes.items():
                print(f"Cliente {key} tem {value} data_size.")

            # Compute selection probabilities based on data size
            total_size = sum(data_sizes.values())
            self.p = [size / total_size for size in data_sizes.values()]

            for i in self.p:
                print(f"p = {i}")

            p_np = np.array(self.p)
            soma_real = p_np.sum()
            
            p_np /= soma_real # Normaliza a maior parte do array
            last_index = len(p_np) - 1
            desvio = 1.0 - p_np[:-1].sum()
            p_np[last_index] = desvio # Força o último elemento a compensar o desvio
            self.p = p_np.tolist()

        # Sample d clients with a probability proportional to their data size
        candidate_clients = np.random.choice(
            available_clients,
            size=min(self.d, len(available_clients)),
            p=self.p,
            replace=False
        )

        # Request the candidate clients to compute their local losses and return them to the server
        local_losses = {
            client.cid: client.fit(
                FitIns(parameters, {"epochs": 0}), 
                timeout=4,
                group_id=str(client.cid) #? Added to fix the error
            ).metrics.get('local_loss', float('inf'))
            for client in candidate_clients
        }

        # Select the top m clients with the highest local losses
        selected_clients_cids = sorted(local_losses, key=local_losses.get, reverse=True)[:self.m]
        selected_clients = []
        selected_clients.append([key for key in selected_clients_cids])

        # Return the selected clients with the FitIns objects
        return [(client_manager.clients.get(cid), FitIns(parameters, {})) for cid in selected_clients_cids]
        

# -------------------------------


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

        net = PoCModel_2HiddenLayers.PoCModel_2HiddenLayers(context, num_classes = 10, suffix = 0)
        ndarrays = get_params(net)
        global_model_init = ndarrays_to_parameters(ndarrays)

        strategy = MyStrategy(
                          evaluate_fn=eval_fn,
                          on_fit_config_fn = fit_config,
                          on_evaluate_config_fn = fit_config,
                          evaluate_metrics_aggregation_fn=weighted_average,  # callback defined earlier
                          initial_parameters=global_model_init,  # initialised global model,
                          fraction_fit=0.03,
                          min_fit_clients=3,
                        )
        num_rounds = 100
        config = ServerConfig(num_rounds=num_rounds)

        return ServerAppComponents(config=config, strategy=strategy)
    return create_server_fn

def generate_client_fn(context):

    def create_client_fn(context_flwr:  Context):

        global fds

        cid = int(context_flwr.node_config["partition-id"])

        partition = fds.load_partition(cid)

        model = PoCModel_2HiddenLayers.PoCModel_2HiddenLayers(context, num_classes = 10, suffix = cid)

        dataset = FMNISTDataset.FMNISTDataset(FM_NORMALIZATION, EVAL_TRANSFORMS, TRAIN_TRANSFORMS, partition, batch_size = 32, shuffle = False, num_workers = 0)

        return PoCExperimentFMNIST.PoCExperimentFMNIST(model, dataset,  context).to_client()

    return create_client_fn


def evaluate_fn(context):
    def fn(server_round, parameters, config):
        """This function is executed by the strategy it will instantiate
        a model and replace its parameters with those from the global model.
        The, the model will be evaluate on the test set (recall this is the
        whole MNIST test set)."""

        global FM_NORMALIZATION, EVAL_TRANSFORMS, TRAIN_TRANSFORMS, DATASET
        global fds

        model = PoCModel_2HiddenLayers.PoCModel_2HiddenLayers(context, num_classes = 10, suffix = "FL-Global")
        model.set_parameters(parameters)

        partition = fds.load_partition(0)

        dataset = FMNISTDataset.FMNISTDataset(FM_NORMALIZATION, EVAL_TRANSFORMS, TRAIN_TRANSFORMS, partition, batch_size = 32, shuffle = False, num_workers = 0)
        dataset.test_partition = load_dataset(DATASET)["test"]

        experiment = PoCExperimentFMNIST.PoCExperimentFMNIST(model, dataset, context)

        config["server_round"] = server_round

        loss, _, return_dic = experiment.evaluate(parameters, config)

        return loss, return_dic

    return fn

partitioner = DirichletPartitioner(
            num_partitions=NUM_PARTITIONS,
            partition_by="label",
            alpha=0.3,
            seed=42,
        )
fds = FederatedDataset(
            dataset=DATASET,
            partitioners={"train": partitioner},
        )


if __name__ == '__main__':

    context = fl.init()

    fl.log(f"Flautim inicializado!!!")


    client_fn_callback = generate_client_fn(context)
    evaluate_fn_callback = evaluate_fn(context)
    server_fn_callback = generate_server_fn(context, eval_fn = evaluate_fn_callback)

    fl.log(f"Experimento criado!!!")

    run_federated(client_fn_callback, server_fn_callback, num_clients = NUM_PARTITIONS)