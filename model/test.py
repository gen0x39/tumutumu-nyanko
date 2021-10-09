import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import torch.optim as optim
import utils
from tqdm import tqdm
from datetime import datetime as dt
from model import NetworkMNIST as Network


parser = argparse.ArgumentParser("mnist")
parser.add_argument('--epochs', type=int, default=600, help='num of training epochs)')
parser.add_argument('--learning_rate', type=float, default=0.01, help='init learning rate')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--save', type=str, default='weight', help='experiment name')
parser.add_argument('--model_path', type=str, default='weight/weight.pt', help='path of pretrained model')
args = parser.parse_args()


def main():
    device = torch.device("cpu")
    
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, ), (0.5, ))])

    testset = MNIST(root='./data',
                    train=False,
                    download=True,
                    transform=transform)

    testloader = DataLoader(testset,
                            batch_size=100,
                            shuffle=False,
                            num_workers=2)

    classes = tuple(np.linspace(0, 9, 10, dtype=np.uint8))

    # model
    model = Network()
    model.to(device)
    utils.load(model, args.model_path)

    # define loss function and optimier
    criterion = nn.CrossEntropyLoss()
    criterion.to(device)
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=0.99,
    )
    
    # test
    correct = 0
    total = 0
    with torch.no_grad():
        for (images, labels) in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy: {:.2f} %'.format(100 * float(correct/total)))


if __name__ == '__main__':
    start_time = time.time()
    main()
    print('elapsed time: {:.3f} [sec]'.format(time.time() - start_time))