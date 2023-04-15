import torch
import torch.nn as nn
import wandb
from torchvision.utils import save_image
from Exs.Ex2.AEs import Encoder1L, Encoder2L, Encoder3L, Encoder4L, Decoder1L, Decoder2L, Decoder3L, Decoder4L
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision

# TRAIN = True
TRAIN = False

WB = True
# WB = False

RUN = '3L'
EPOCHS = 10
LR = 1e-3
DESCRIPTION = "Pearson Correlation & Latent Space "
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def init_w_and_b():
    if WB:
        wandb.init(
            # Set the project where this run will be logged
            group="Pearson Correlation [10-100]",
            project="NN4I_Ex2 ",
            name=f"{DESCRIPTION}{RUN}",
            notes='',
            # Track hyperparameters and run metadata
            config={
                "learning_rate": LR,
                "architecture": "AE",
                "dataset": "MNIST",
                "epochs": EPOCHS,

            })


def load_data():
    from torchvision import datasets, transforms

    train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())

    trainloader = torch.utils.data.DataLoader(dataset=train_data,
                                              batch_size=train_data.__len__(),
                                              shuffle=True)

    test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

    testloader = torch.utils.data.DataLoader(dataset=test_data,
                                             batch_size=64,
                                             shuffle=True)
    return trainloader, testloader


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


def correlate(encoder, trainloader):
    img, _ = next(iter(trainloader))
    with torch.no_grad():
        latent = encoder(img)
        corr_matrix = latent.T.corrcoef()
        corr_matrix_norm = corr_matrix.norm()
        return corr_matrix_norm.item()


def train_and_correlate(trainloader, d):
    if TRAIN:
        encoder, decoder = Encoder3L(d), Decoder3L(d)
        outputs = []
        for epoch in range(EPOCHS):
            train_loss = train(trainloader, encoder, decoder, epoch, outputs)
            print('\nTrain set: Avg. test loss: {:.4f}\n'.format(train_loss))

        # save model
        torch.save(encoder.state_dict(), f'./models/encoder{RUN}_{d}D.pth')
        torch.save(decoder.state_dict(), f'./models/decoder{RUN}_{d}D.pth')
    else:
        encoder, decoder = Encoder3L(d), Decoder3L(d)
        encoder.load_state_dict(torch.load(f'./models/encoder{RUN}_{d}D.pth'))
        # decoder.load_state_dict(torch.load(f'./models/decoder{RUN}_{d}D.pth'))

    corr_matrix_norm = correlate(encoder, trainloader)
    return corr_matrix_norm


def explore(trainloader):
    norms_matrix = []
    latent_dimensions = [10, 15, 20, 25, 30, 35, 40, 80, 100]
    for d in latent_dimensions:
        corr_matrix_norm = train_and_correlate(trainloader, d)
        print(f"latent: {d}, pearson: {corr_matrix_norm}")
        norms_matrix.append(corr_matrix_norm)
        if WB:
            correlation_and_latent_dimension = [[d, n] for d, n in zip(latent_dimensions, norms_matrix)]
            table = wandb.Table(data=correlation_and_latent_dimension, columns=["Latent Dimension",
                                                                                "Pearson Correlation (matrix norm)"])
            wandb.log({"Correlation_and_Latent_Dimension": wandb.plot.line(table, "Latent Dimension",
                                                                           "Pearson Correlation (matrix norm)",
                                                                           title="Pearson Correlation (per coordinate)"
                                                                                 "and Latent Dimension")})
            # wandb.log({"Latent Dimension": d, 'Pearson Correlation (matrix norm)': corr_matrix_norm})
    return norms_matrix


def main():
    if WB:
        init_w_and_b()
    trainloader, _ = load_data()
    explore(trainloader)


if __name__ == '__main__':
    main()
