import torch
from model import AlexNet
import config
import torch.nn as nn

import torchvision
import torchvision.transforms as transforms
import engine
from torch.utils.data import DataLoader

if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose(
        [transforms.Resize((227, 227)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=config.BATCH_SIZE,
                             shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=config.BATCH_SIZE,
                            shuffle=False, num_workers=2)
    dataiter = iter(testloader)
    images, labels = next(dataiter)

    model = AlexNet(config.NUM_CLASSES)
    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=config.LEARNING_RATE,
                                weight_decay=0.005, momentum=0.9)

    for epoch in range(config.NUM_EPOCHS):
        print(f'Epoch : {epoch + 1}/{config.NUM_EPOCHS} .')

        training_loss = engine.train_fn(model=model, data_loader=trainloader, loss_fn=loss_fn,
                                        optimizer=optimizer, device=device)

        validation_accuracy = engine.eval_fn(model, testloader, device)

        print("Training Loss is : ", training_loss)
        print("Validation Accuracy is : ", validation_accuracy)
