import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import wandb
import torch.optim as optim

from Exs.Ex1 import BaseLineNet
from Exs.Ex1 import Multiple_FC_Net
from Exs.Ex1 import Net

# RUN = 'BaseLine'
# RUN = 'Multiple_FC_Net'
# RUN = 'ComplexNet'
RUN = 'Net'


# WB = False
WB = True

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
    if WB:
        wandb.init(
            # Set the project where this run will be logged
            project="NN4I_Ex1 ",
            # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
            name=f"{RUN}_{EPOCHS}_epochs",
            notes='',
            # Track hyperparameters and run metadata
            config={
                "learning_rate": LR,
                "architecture": "CNN",
                "dataset": "CIFAR-10",
                "epochs": EPOCHS,
            })


def load_and_test(testloader, valloader=None, PATH=''):
    if RUN == 'BaseLine':
        PATH = f'./{RUN}_{EPOCHS}.pth'
        net = BaseLineNet.BaseLineNet()
    elif RUN == 'Multiple_FC_Net':
        PATH = f'./{RUN}_{EPOCHS}.pth'
        net = Multiple_FC_Net.MultipleFCNet()
    elif RUN == "Net":
        PATH = f'./{RUN}_{EPOCHS}.pth'
        net = Net.Net()
    elif RUN == 'ComplexNet':
        PATH = f'./{RUN}_{EPOCHS}.pth'
        net = Net.ComplexNet()

    else:
        net = Net.Net()
        raise Exception

    net.load_state_dict(torch.load(PATH))
    net.eval()  # set to evaluation mode

    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    test_loss = 0.0

    with torch.no_grad():
        for i, data in enumerate(testloader):
            images, labels = data
            outputs = net(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_loss /= len(testloader.dataset)
    test_acc = 100.0 * correct / total

    if valloader:
        with torch.no_grad():
            running_val_loss = 0.0
            for i, data in enumerate(valloader, 0):
                inputs, labels = data
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()
            val_loss = running_val_loss / len(valloader)

    if WB:
        wandb.log({"test_loss": test_loss})
        wandb.log({"test_acc": test_acc})
    print(f'Accuracy of the network on the 10000 test images: {test_acc} %')


def calculate_test_loss(net, testloader):
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    test_loss = 0.0

    with torch.no_grad():
        for i, data in enumerate(testloader):
            images, labels = data
            outputs = net(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_loss /= len(testloader.dataset)
    test_acc = 100.0 * correct / total

    if WB:
        wandb.log({"test_loss": test_loss})
        wandb.log({"test_acc": test_acc})
    print(f'Accuracy of the network on the 10000 test images: {test_acc} %')


def build_and_train(trainloader, testloader):
    if RUN == 'BaseLine':
        net = BaseLineNet.BaseLineNet()
    elif RUN == 'Multiple_FC_Net':
        net = Multiple_FC_Net.MultipleFCNet()
    elif RUN == "Net":
        net = Net.Net()
    elif RUN == 'ComplexNet':
        net = Net.ComplexNet()
    else:
        net = Net.Net()

    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(net.parameters())
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
                if WB:
                    # wandb.log({"acc": acc, "loss": loss})
                    wandb.log({"epoch": epoch, "train_loss": loss})
        calculate_test_loss(net, testloader)

    print('Finished Training')
    wandb.log({"number of parameters": sum(p.numel() for p in net.parameters() if p.requires_grad)})
    print(sum(p.numel() for p in net.parameters() if p.requires_grad))
    PATH = f'./{RUN}_{EPOCHS}.pth'
    torch.save(net.state_dict(), PATH)


def main():
    init_w_and_b()
    trainset, trainloader, testset, testloader, classes = load_data()
    build_and_train(trainloader, testloader)
    # load_and_test(testloader)
    if WB:
        wandb.finish()


if __name__ == '__main__':
    import sys

    if WB:
        wandb.login()
    if len(sys.argv) == 2:
        param = {1: 'BaseLine', 2: 'Multiple_FC_Net', 3: 'Net'}
        RUN = param[int(sys.argv[1])]
    main()
