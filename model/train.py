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
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs)')
parser.add_argument('--learning_rate', type=float, default=0.01, help='init learning rate')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--save', type=str, default='weight', help='experiment name')
args = parser.parse_args()


def main():
    # --- Setting ---
    # gpu is available
    if not torch.cuda.is_available():
        sys.exit(1)
    
    torch.cuda.set_device(args.gpu)

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, ), (0.5, ))])

    trainset = MNIST(root='./data',
                     train=True,
                     download=True,
                     transform=transform)
    testset = MNIST(root='./data',
                    train=False,
                    download=True,
                    transform=transform)

    trainloader = DataLoader(trainset,
                             batch_size=100,
                             shuffle=True,
                             num_workers=2)
    testloader = DataLoader(testset,
                            batch_size=100,
                            shuffle=False,
                            num_workers=2)

    classes = tuple(np.linspace(0, 9, 10, dtype=np.uint8))

    # model
    model = Network()
    model.cuda()

    # define loss function and optimier
    criterion = nn.CrossEntropyLoss()
    criterion.cuda()
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=0.99,
    )

    now = dt.now()
    start = now.strftime('%p%I:%M:%S')

    # train
    for epoch in range(args.epochs):
        with tqdm(total=len(trainloader),unit='batch') as progress_bar:
            progress_bar.set_description(f"Epoch[{epoch}/{args.epochs}](training) start: " + start)

            for i, (inputs, labels) in enumerate(trainloader, 0):
                inputs = inputs.cuda()
                labels = labels.cuda()
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                progress_bar.set_postfix({"loss":loss.item(), "now":now.strftime('%p%I:%M:%S')})
                progress_bar.update(1)
            
            utils.save(model, os.path.join(args.save, 'weight.pt'))
    print('Finished Training')

    # test
    correct = 0
    total = 0
    with torch.no_grad():
        for (images, labels) in testloader:
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy: {:.2f} %'.format(100 * float(correct/total)))


if __name__ == '__main__':
    start_time = time.time()
    main()
    print('elapsed time: {:.3f} [sec]'.format(time.time() - start_time))