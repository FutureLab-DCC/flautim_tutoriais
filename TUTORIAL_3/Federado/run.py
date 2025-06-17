from flautim.pytorch.common import run_federated, weighted_average
from flautim.pytorch import Model, Dataset
from flautim.pytorch.federated import Experiment
import MNISTDataset, MNISTModel, MNISTExperiment
import flautim as fl
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
from flwr.common import Context, ndarrays_to_parameters
import flwr
import pandas as pd
import numpy as np
from flwr.server import ServerConfig, ServerAppComponents
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

DATASET = "zalando-datasets/fashion_mnist"
NUM_PARTITIONS = 30

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

        net = MNISTModel.MNISTModel(context, num_classes = 10, suffix = 0)
        params = ndarrays_to_parameters(net.get_parameters())

        strategy = flwr.server.strategy.FedAvg(
                          initial_parameters=params,
                          evaluate_metrics_aggregation_fn=weighted_average,
                          fraction_fit=0.1,  # 10% clients sampled each round to do fit()
                          fraction_evaluate=0.5,  # 50% clients sample each round to do evaluate()
                          evaluate_fn=eval_fn,
                          on_fit_config_fn = fit_config,
                          on_evaluate_config_fn = fit_config
                          )
        num_rounds = 30
        config = ServerConfig(num_rounds=num_rounds)

        return ServerAppComponents(config=config, strategy=strategy)
    return create_server_fn

def generate_client_fn(context):
    
    def create_client_fn(context_flwr:  Context):

        global fds
        
        cid = int(context_flwr.node_config["partition-id"])
        
        partition = fds.load_partition(cid)

        model = MNISTModel.MNISTModel(context, num_classes = 10, suffix = cid)
        
        dataset = MNISTDataset.MNISTDataset(FM_NORMALIZATION, EVAL_TRANSFORMS, TRAIN_TRANSFORMS, partition, batch_size = 32, shuffle = False, num_workers = 0)
        
        return MNISTExperiment.MNISTExperiment(model, dataset,  context).to_client() 

    return create_client_fn
    

def evaluate_fn(context):
    def fn(server_round, parameters, config):
        """This function is executed by the strategy it will instantiate
        a model and replace its parameters with those from the global model.
        The, the model will be evaluate on the test set (recall this is the
        whole MNIST test set)."""
        
        global FM_NORMALIZATION, EVAL_TRANSFORMS, TRAIN_TRANSFORMS, DATASET
        global fds

        model = MNISTModel.MNISTModel(context, num_classes = 10, suffix = "FL-Global")
        model.set_parameters(parameters)
        
        partition = fds.load_partition(0)
        
        dataset = MNISTDataset.MNISTDataset(FM_NORMALIZATION, EVAL_TRANSFORMS, TRAIN_TRANSFORMS, partition, batch_size = 32, shuffle = False, num_workers = 0)
        dataset.test_partition = load_dataset(DATASET)["test"]
        
        experiment = MNISTExperiment.MNISTExperiment(model, dataset, context)
        
        config["server_round"] = server_round
        
        loss, _, return_dic = experiment.evaluate(parameters, config) 

        return loss, return_dic

    return fn
    
partitioner = DirichletPartitioner(
            num_partitions=NUM_PARTITIONS,
            partition_by="label",
            alpha=1.0,
            seed=42,
        )
fds = FederatedDataset(
            dataset=DATASET,
            partitioners={"train": partitioner},
        )

fds.load_partition(0)

if __name__ == '__main__':

    context = fl.init() 
    
    fl.log(f"Flautim inicializado!!!")

    
    client_fn_callback = generate_client_fn(context)
    evaluate_fn_callback = evaluate_fn(context)
    server_fn_callback = generate_server_fn(context, eval_fn = evaluate_fn_callback)
    
    fl.log(f"Experimento criado!!!")
    
    run_federated(client_fn_callback, server_fn_callback, num_clients = NUM_PARTITIONS)
