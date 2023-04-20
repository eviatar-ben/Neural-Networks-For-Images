import torch
import torch.nn as nn
from torchvision.utils import save_image
from Exs.Ex2.AEs import Encoder1L, Encoder2L, Encoder3L, Encoder4L, Decoder1L, Decoder2L, Decoder3L, Decoder4L
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision

TRAIN = True
# TRAIN = False

RUN = '3L'
EPOCHS = 10
LR = 1e-3
DESCRIPTION = "Same d multiple number of layers"
D = 3
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


def get_two_dif_digits(trainloader, encoder, specific_digits=False, digits=[2, 3]):
    images, digits = next(iter(trainloader))
    encoded_images = encoder(images)
    encoded_images_and_digits = [(i, d) for i, d in (zip(encoded_images, digits))]
    if specific_digits:
        d1, d2 = torch.tensor(digits[0]), torch.tensor(digits[1])
        for i, d in encoded_images_and_digits:
            if torch.eq(d1, d):
                ei1 = i
                break
        for i, d in encoded_images_and_digits:
            if torch.eq(d2, d):
                ei2 = i
                return ei1, ei2, d1, d2

    ei1 = encoded_images_and_digits[0][0]
    d1 = encoded_images_and_digits[0][1]
    ed1 = encoded_images_and_digits[0][1]

    for i, d in encoded_images_and_digits:
        if not torch.eq(ed1, d):
            ei2 = i
            d2 = d
            return ei1, ei2, d1, d2
    raise Exception


def interpolate(encoder, decoder, trainloader, num_of_steps=25, same_img=False):
    """
    D((E(I1)*a)+(E(I2)*(1-a))) for a [0,1]
    """
    import numpy as np
    alphas = np.linspace(0, 1, num_of_steps)
    if same_img:
        ei1, ei2, d1, d2 = same_img
    else:
        ei1, ei2, d1, d2 = get_two_dif_digits(trainloader, encoder, (3, 2))
    interpolations = []
    for alpha in alphas:
        interpolate_input = alpha * ei1 + (1 - alpha) * ei2
        interpolations += [decoder(interpolate_input.unsqueeze(0)).view(-1, 28)]

    out = torch.concat(interpolations, 1)
    save_image(out, f"./plots/interpolations/interpolation_D{D}_[{d2}-{d1}].png")


def train_and_interpolate(trainloader, d=D):
    # from Exs.Ex2.TransferLearning import Encoder, Decoder
    if TRAIN:
        encoder, decoder = Encoder3L(d), Decoder3L(d)
        outputs = []
        for epoch in range(EPOCHS):
            train_loss = train(trainloader, encoder, decoder, epoch, outputs)
            print(f"Epoch : {epoch}")
            # print('\nTrain set: Avg. test loss: {:.4f}\n'.format(train_loss))

        # save model
        torch.save(encoder.state_dict(), f'./models/encoder{RUN}_{d}D.pth')
        torch.save(decoder.state_dict(), f'./models/decoder{RUN}_{d}D.pth')
    else:
        encoder, decoder = Encoder3L(d), Decoder3L(d)
        encoder.load_state_dict(torch.load(f'./models/encoder{RUN}_{d}D.pth'))
        decoder.load_state_dict(torch.load(f'./models/decoder{RUN}_{d}D.pth'))
    # same_image = get_two_dif_digits(trainloader, encoder, specific_digits=[1, 8])
    interpolate(encoder, decoder, trainloader)


def main():
    trainloader, _ = load_data()
    train_and_interpolate(trainloader)
    # for d in [5, 10,15, 20, 25, 30, 35, 40, 80, 90, 100 ]:


if __name__ == '__main__':
    main()
