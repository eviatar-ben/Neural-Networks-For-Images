import torch
import wandb


def load_data(batch_size=64):
    from torchvision import datasets, transforms

    train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())

    trainloader = torch.utils.data.DataLoader(dataset=train_data,
                                              batch_size=batch_size,
                                              shuffle=True)

    test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

    testloader = torch.utils.data.DataLoader(dataset=test_data,
                                             batch_size=batch_size,
                                             shuffle=True)
    return trainloader, testloader


def init_w_and_b(group, project, d="", description="", run="", LR=1e-3, EPOCHS=10, architecture="GAN"):
    wandb.init(
        # Set the project where this run will be logged
        group=group,
        project=project,
        name=f"{description}{run}{d}",
        notes='',
        # Track hyperparameters and run metadata
        config={
            "learning_rate": LR,
            "architecture": architecture,
            "dataset": "MNIST",
            "epochs": EPOCHS,

        })
