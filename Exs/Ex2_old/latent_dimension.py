import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image
import wandb

WB = False
# WB = True

RUN = 'AE'
EPOCHS = 9
LR = 1e-3
DESCRIPTION = "Latent_dimension_higher_dims"
D = 10
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# todo check maybe the loss size order is small because of batch normalize  self.bn

class Encoder(nn.Module):

    def __init__(self, d=D):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=2, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=0)
        self.relu3 = nn.ReLU()

        self.flatten = nn.Flatten(start_dim=1)

        self.fc1 = nn.Linear(3 * 3 * 32, d)
        self.relu4 = nn.ReLU()

    def forward(self, x):
        # con' layers
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        # Flatten
        x = self.flatten(x)
        # Fully-connected layer
        x = self.relu4(x)
        x = self.fc1(x)

        # x = self.fc2(x)
        # x = nn.Tanh(x)
        return x


class Decoder(nn.Module):
    def __init__(self, d=D):
        super().__init__()
        # self.fc2 = nn.Linear(d, 128)
        self.relu4 = nn.ReLU()
        self.fc1 = nn.Linear(d, 3 * 3 * 32)

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32, 3, 3))

        self.relu3 = nn.ReLU()
        self.conv3 = nn.ConvTranspose2d(32, 16, 3, stride=2, output_padding=0)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        # x = self.fc2(x)
        x = self.relu4(x)
        x = self.fc1(x)

        x = self.unflatten(x)

        x = self.relu3(x)
        x = self.conv3(x)
        x = self.relu2(x)
        x = self.conv2(x)
        x = self.relu1(x)
        x = self.conv1(x)
        x = torch.sigmoid(x)

        return x


def load_data():
    batch_size_train = 128
    batch_size_test = 128
    trainloader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./data', train=True, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor()
                                       # , torchvision.transforms.Normalize((0.5,), (0.5,))
                                   ])),
        batch_size=batch_size_train, shuffle=True, pin_memory=True)

    testloader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./data', train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor()
                                       # , torchvision.transforms.Normalize((0.5,), (0.5,))
                                   ])),
        batch_size=batch_size_test, shuffle=True, pin_memory=True)
    return trainloader, testloader


def plot_images(output, img):
    import matplotlib.pyplot as plt
    # plt.figure(figsize=(2, 2))
    plt.gray()
    plt.imshow(output[0][0].detach().numpy())
    plt.imshow(img[0][0].detach().numpy())
    # imgs = output[k][1].detach().numpy()
    # recon = outputs[k][2].detach().numpy()
    # for i, item in enumerate(imgs):
    #     if i >= 9: break
    #     plt.subplot(2, 9, i + 1)
    #     # item = item.reshape(-1, 28,28) # -> use for Autoencoder_Linear
    #     # item: 1, 28, 28
    #     plt.imshow(item[0])
    #
    # for i, item in enumerate(recon):
    #     if i >= 9: break
    #     plt.subplot(2, 9, 9 + i + 1)  # row_length + i + 1
    #     # item = item.reshape(-1, 28,28) # -> use for Autoencoder_Linear
    #     # item: 1, 28, 28
    #     plt.imshow(item[0])
    return


def train_and_test(trainloader, testloader, d=D):
    # random_seed = 1
    # torch.backends.cudnn.enabled = False
    # torch.manual_seed(random_seed)

    encoder = Encoder(d).to(device)
    decoder = Decoder(d).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=LR)

    def train():
        cur_train_loss = 0
        # model.train()  # todo: check this out
        for img, _ in trainloader:
            optimizer.zero_grad()

            img = img.to(device)

            latent = encoder(img).to(device)
            output = decoder(latent).to(device)

            loss = criterion(output, img)
            loss.backward()
            optimizer.step()
            cur_train_loss += loss.item()
        print('\nTest set: Avg. train loss: {:.4f}\n'.format(cur_train_loss / len(trainloader)))
        return cur_train_loss / len(trainloader)

    def test():
        cur_test_loss = 0
        with torch.no_grad():
            for img, _ in testloader:
                latent = encoder(img).to(device)
                output = decoder(latent).to(device)
                cur_test_loss += criterion(output, img).item()
            # plot_images(output, img)
        cur_test_loss /= len(testloader.dataset)
        print('\nTrain set: Avg. test loss: {:.4f}\n'.format(cur_test_loss))

        return cur_test_loss

    for epoch in range(EPOCHS):
        train_loss = train()
        test_loss = test()
        if WB:
            wandb.log({"train_loss": train_loss, 'test_loss': test_loss}, step=epoch)


def init_w_and_b(d=D):
    if WB:
        wandb.init(
            # Set the project where this run will be logged
            group="Searching for looses anomaly",
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
    for d in range(15, 16):
        # set wandb new plot per current d value
        init_w_and_b(d)
        train_and_test(trainloader, testloader, d)
        if WB:
            wandb.finish()


if __name__ == '__main__':
    if WB:
        wandb.login()
    main()
