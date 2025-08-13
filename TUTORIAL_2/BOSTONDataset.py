from flautim.pytorch.Dataset import Dataset
import torch
import copy
import flautim as fl

class BostonDataset(Dataset):

    def __init__(self, file, **kwargs):
        super(BostonDataset, self).__init__(name = "Boston", **kwargs)

        # Defina o que são features e targets
        self.features = file.iloc[:, 0:13].values
        self.target = file.iloc[:, 13].values

        # Número de amostras para teste
        self.test_size = int(0.2 * len(file))

        # Defina o tipo do tensor de entrada e de saída.
        self.xdtype = torch.float32
        self.ydtype = torch.float32

        # batch_size
        self.batch_size = 10

        # shuffle
        self.shuffle = True

        # num_workers
        self.num_workers = 1

    def train(self) -> Dataset:
        # Separação das amostras para treino
        train = copy.deepcopy(self)
        train.features = self.features[:-self.test_size]
        train.target = self.target[:-self.test_size]
        return copy.deepcopy(train)

    def validation(self) -> Dataset:
        # Separação das amostras para validação
        test = copy.deepcopy(self)
        test.features = self.features[-self.test_size:]
        test.target = self.target[-self.test_size:]
        return copy.deepcopy(test)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.float32), torch.LongTensor([self.target[idx]])