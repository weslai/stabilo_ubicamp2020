## create a DataFrameDataset
## to use DataFrame as a Data Source

from torch.utils.data import Dataset
import torch

class LetterDataset(Dataset):
    """Stage 1 Letters Dataset"""

    def __init__(self, mode, train_datasets: list, test_datasets: list, transform = None):
        """
        Args:
            mode: train or test
            datasets: List from the whole dataset
            labels: List from the whole dataset
            alpha: percentage for Testing data
            transform: Not sure would be used, usually for Images
        """
        self.mode = mode
        self.train = train_datasets
        self.test = test_datasets
        self.transform = transform

        print('number of trainset:', len(self.train))
        print('number of testset:', len(self.test))

    def __len__(self):
        if self.mode is 'train':
            return len(self.train)
        else:
            return len(self.test)

    def __getitem__(self, idx):
        if self.mode is 'train':
            label = self.train[idx][1]
            sample = self.train[idx][0]
            sample_tensor = torch.tensor(sample)
        else:
            label = self.test[idx][1]
            sample = self.test[idx][0]
            sample_tensor = torch.tensor(sample)

        return sample_tensor, label

## get train dataset
def get_train_dataset(datalist: list, transform= None):
    """
    Args:
        datasets: the whole Dataset

        alpha: for testing data
        transform: None

    Returns: training dataset
    """
    datatuple = datalist[0]
    train_dataset = datatuple[0]
    test_dataset = datatuple[1]

    ### wichtig: train_list, train_idx, train_len also by test
    ### train_idx is increasing
    trainset = LetterDataset('train', train_dataset, test_dataset, transform)

    return trainset
## get test dataset
def get_test_dataset(datalist: list, transform= None):
    """
    Args:
        datasets: the whole Dataset
        alpha: for testing data
        transform: None (not used in Testing dataset)

    Returns: test dataset
    """
    datatuple = datalist[0]
    train_dataset = datatuple[0]
    test_dataset = datatuple[1]

    testset = LetterDataset('test', train_dataset, test_dataset, transform)

    return testset


