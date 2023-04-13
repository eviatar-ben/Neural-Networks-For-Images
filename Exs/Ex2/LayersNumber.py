import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image
import wandb
from Exs.Ex2.AEs import Encoder1L, Encoder2L, Encoder3L, Encoder4L, Decoder1L, Decoder2L, Decoder3L, Decoder4L

# WB = False
WB = True

PLOT = not WB

RUN = 'AE'
EPOCHS = 10
LR = 1e-3
DESCRIPTION = "Same d multiple number of layers"
D = 15
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_data():
    from torchvision import datasets, transforms

    train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())

    trainloader = torch.utils.data.DataLoader(dataset=train_data,
                                              batch_size=64,
                                              shuffle=True)

    test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

    testloader = torch.utils.data.DataLoader(dataset=test_data,
                                             batch_size=64,
                                             shuffle=True)
    return trainloader, testloader


def plot_images(outputs):
    import matplotlib.pyplot as plt
    for k in range(0, EPOCHS, 4):
        plt.figure(figsize=(9, 2))
        plt.gray()
        imgs = outputs[k][1].detach().numpy()
        recon = outputs[k][2].detach().numpy()
        for i, item in enumerate(imgs):
            if i >= 9: break
            plt.subplot(2, 9, i + 1)
            plt.imshow(item)
            plt.show()

        for i, item in enumerate(recon):
            if i >= 9: break
            plt.subplot(2, 9, 9 + i + 1)  # row_length + i + 1
            plt.imshow(item)
            plt.show()
    return


def test(testloader, encoder, decoder):
    test_loss = 0
    criterion = nn.MSELoss()
    with torch.no_grad():
        for img, _ in testloader:
            img = img.to(device)

            latent = encoder(img).to(device)
            output = decoder(latent).to(device)

            loss = criterion(output, img)
            test_loss += loss.item()
        print('\nTest set: Avg. test loss: {:.4f}\n'.format(test_loss / len(testloader)))
    return test_loss / len(testloader)


def train(trainloader, encoder, decoder, epoch, outputs):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=LR)
    train_loss = 0
    flag = True
    for img, _ in trainloader:
        optimizer.zero_grad()

        img = img.to(device)

        latent = encoder(img).to(device)
        output = decoder(latent).to(device)

        loss = criterion(output, img)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        if flag:
            outputs.append((epoch, img[0], output[0]))
            flag = False
    print('\nTrain set: Avg. train loss: {:.4f}\n'.format(train_loss / len(trainloader)))
    return train_loss / len(trainloader)


def train_and_test(trainloader, testloader, d=15, nl=3):
    if nl == 1:
        encoder = Encoder1L(d).to(device)
        decoder = Decoder1L(d).to(device)
        print("encoder1")
    elif nl == 2:
        encoder = Encoder2L(d).to(device)
        decoder = Decoder2L(d).to(device)
        print("encoder2")
    elif nl == 3:
        encoder = Encoder3L(d).to(device)
        decoder = Decoder3L(d).to(device)
        print("encoder3")
    elif nl == 4:
        encoder = Encoder4L(d).to(device)
        decoder = Decoder4L(d).to(device)
        print("encoder4")
    else:
        raise Exception("l must be in [1-4]")

    outputs = []
    for epoch in range(EPOCHS):
        train_loss = train(trainloader, encoder, decoder, epoch, outputs)
        test_loss = test(testloader, encoder, decoder)
        if WB:
            wandb.log({"train_loss": train_loss, 'test_loss': test_loss}, step=epoch)
    # if PLOT:
    #     plot_images(outputs)


def init_w_and_b(d=D):
    if WB:
        wandb.init(
            # Set the project where this run will be logged
            group="number of layers [1-4]",
            project="NN4I_Ex2 ",
            name=f"{DESCRIPTION}{RUN}{d}",
            notes='',
            # Track hyperparameters and run metadata
            config={
                "learning_rate": LR,
                "architecture": "AE",
                "dataset": "MNIST",
                "epochs": EPOCHS,

            })


def main():
    trainloader, testloader = load_data()
    for nl in range(1, 5):
        # set wandb new plot per current nl value
        init_w_and_b(nl)
        train_and_test(trainloader, testloader, nl)
        if WB:
            wandb.finish()


if __name__ == '__main__':
    if WB:
        wandb.login()
    main()
