'''
PyTorch MNIST sample
'''
import argparse
import time
import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import torch.optim as optim
from torch.autograd import Variable

from net import Net


def parser():
    '''
    argument
    '''
    parser = argparse.ArgumentParser(description='PyTorch MNIST')
    parser.add_argument('--epochs', '-e', type=int, default=2,
                        help='number of epochs to train (default: 2)')
    parser.add_argument('--lr', '-l', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
    parser.add_argument('--save', type=str, default='weight', help='experiment name')
    parser.add_argument('--model_path', type=str, default='weight/weight.pt', help='path of pretrained model')
    args = parser.parse_args()
    return args

# load model from pt file
def load(model, model_path):
    model.load_state_dict(torch.load(model_path))

def main():
    '''
    main
    '''
    device = torch.device("cpu")
    args = parser()

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
    net = Net()
    net.to(device)
    load(net, args.model_path)

    # define loss function and optimier
    criterion = nn.CrossEntropyLoss()
    criterion.cuda()
    optimizer = optim.SGD(net.parameters(),
                          lr=args.lr, momentum=0.99, nesterov=True)

    # test
    correct = 0
    total = 0
    with torch.no_grad():
        for (images, labels) in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy: {:.2f} %'.format(100 * float(correct/total)))


if __name__ == '__main__':
    start_time = time.time()
    main()
    print('elapsed time: {:.3f} [sec]'.format(time.time() - start_time))