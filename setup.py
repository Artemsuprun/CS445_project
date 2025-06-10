import torch
from torch.utils.data import TensorDataset
import torchvision.datasets as dsets
import torchvision.transforms as transforms


def setup_train(dataset):
    if dataset == 'MNIST':
        # Get the MNIST dataset
        train_set = dsets.MNIST(root='./data/MNIST', train=True, transform=transforms.ToTensor(), download=True)
    elif dataset == 'FMNIST':
        # Get the Fashion MNIST dataset
        train_set = dsets.FashionMNIST(root='./data/FMNIST', train=True, transform=transforms.ToTensor(), download=True)

    train_input = []
    train_target = []
    for img, _ in train_set:
       input_img = img.clone()
       target_img = img.clone()
       train_input.append(input_img)
       train_target.append(target_img)
    train_input = torch.stack(train_input)
    train_target = torch.stack(train_target)

    return TensorDataset(train_input, train_target)


def setup_test(dataset):
    if dataset == 'MNIST':
        # Get the MNIST dataset
        test_set = dsets.MNIST(root='./data/MNIST', train=False, transform=transforms.ToTensor(), download=True)
    elif dataset == 'FMNIST':
        # Get the Fashion MNIST dataset
        test_set = dsets.FashionMNIST(root='./data/FMNIST', train=False, transform=transforms.ToTensor(), download=True)
    
    test_input = []
    test_target = []
    for img, _ in test_set:
       input_img = img.clone()
       target_img = img.clone()
       test_input.append(input_img)
       test_target.append(target_img)
    test_input = torch.stack(test_input)
    test_target = torch.stack(test_target)

    return TensorDataset(test_input, test_target)