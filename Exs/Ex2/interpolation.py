import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image
from Exs.Ex2.AEs import Encoder1L, Encoder2L, Encoder3L, Encoder4L, Decoder1L, Decoder2L, Decoder3L, Decoder4L

TRAIN = True
RUN = '3L'
EPOCHS = 10
LR = 1e-3
DESCRIPTION = "Same d multiple number of layers"
D = 40
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

def get_two_dif_digits(trainloader, d= False):
    if d:
        i1, d1 = next(iter(trainloader))
        i2, d2 = next(iter(trainloader))
        while d1 != d[0]:
            i1, d1 = next(iter(trainloader))
        while d2 != d[1]:
            i2, d2 = next(iter(trainloader))
        return i1, i2
    i1, d1 = next(iter(trainloader))
    i2, d2 = next(iter(trainloader))
    while d2 != d1:
        i2, d2 = next(iter(trainloader))
    return i1, i2

def interpolate(encoder, decoder, trainloader, num_of_steps = 25):
    """
    D((E(I1)*a)+(E(I2)*(1-a))) for a [0,1]
    """
    import numpy as np
    alphas = np.linspace(0, 1, num_of_steps)
    i1, i2 = get_two_dif_digits(trainloader)
    ei1 = encoder(i1)
    ei2 = encoder(i2)
    interpolation = []
    for alpha in alphas:
        noise = alpha * ei1 + (1 - alpha) * ei1
        interpolation += [decoder(noise.unsqueeze(0)).view(-1, 28)]

    out = torch.concat(interpolation, 1)
    save_image(out, f"./plots/interpolation.png")



    interpolation = []
    for alpha in alphas:


def train_and_interpolate(trainloader):
    if TRAIN:
        encoder, decoder = Encoder3L(), Decoder3L()
        outputs = []
        for epoch in range(EPOCHS):
            train_loss = train(trainloader, encoder, decoder, epoch, outputs)
            print('\nTrain set: Avg. test loss: {:.4f}\n'.format(train_loss))

        # save model
        torch.save(encoder.state_dict(), f'./models/encoder{RUN}.pth')
        torch.save(decoder.state_dict(), f'./models/decoder{RUN}.pth')
    else:
        pass
    interpolate(encoder, decoder, trainloader)


def main():
    trainloader, _ = load_data()
    train_and_interpolate(trainloader)


if __name__ == '__main__':
    main()
