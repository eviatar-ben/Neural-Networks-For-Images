import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import wandb
import torch.optim as optim

from Exs.Ex1 import BaseLineNet
from Exs.Ex1 import Net

RUN = 'BaseLine'
# RUN = 'MyNet'

EPOCHS = 15
LR = 0.001


# -----------------------------------------------Helpers-------------------------------------------------

def load_data():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 4

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=0)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=0)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainset, trainloader, testset, testloader, classes


def init_w_and_b():
    wandb.init(
        # Set the project where this run will be logged
        project="NN4I_Ex1 ",
        # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
        name=f"experiment_{RUN}_{EPOCHS}_epochs",
        # Track hyperparameters and run metadata
        config={
            "learning_rate": LR,
            "architecture": "CNN",
            "dataset": "CIFAR-10",
            "epochs": EPOCHS,
        })


def load_and_test(testloader, PATH='./cifar_net.pth'):
    if RUN == 'BaseLine':
        net = BaseLineNet.BaseLineNet()
    else:
        net = Net.Net()

    net.load_state_dict(torch.load(PATH))
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()

    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for i, data in enumerate(testloader):
            images, labels = data
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # cross entropy loss
            loss = criterion(outputs, labels)
            if i % 20 == 19:  # print every 10 mini-batches
                wandb.log({"test_acc": 100 * correct // total})
                wandb.log({"test_loss": loss})

    acc = 100 * correct // total
    wandb.log({"final_acc": 100 * correct // total})
    print(f'Accuracy of the network on the 10000 test images: {acc} %')


def build_and_train(trainloader):
    if RUN == 'BaseLine':
        net = BaseLineNet.BaseLineNet()
    else:
        net = Net.Net()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9)

    for epoch in range(EPOCHS):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
                # wandb.log({"acc": acc, "loss": loss})
                wandb.log({"epoch": epoch, "train_loss": loss})

    print('Finished Training')

    PATH = f'./cifar_net_{RUN}_{EPOCHS}_{LR}.pth'
    torch.save(net.state_dict(), PATH)


def main():
    init_w_and_b()
    trainset, trainloader, testset, testloader, classes = load_data()
    build_and_train(trainloader)
    load_and_test(testloader)
    wandb.finish()


if __name__ == '__main__':
    wandb.login()
    main()
