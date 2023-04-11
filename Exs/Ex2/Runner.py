import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image
import wandb

# WB = False
WB = True

RUN = 'AE'
EPOCHS = 9
LR = 1e-3
# DESCRIPTION = "NonLinear"
DESCRIPTION = "First"
D = 10
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):

    def __init__(self, d=D):
        super(Encoder, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=0),
            nn.ReLU(True))

        self.flatten = nn.Flatten(start_dim=1)

        self.fc = nn.Sequential(
            nn.Linear(3 * 3 * 32, 128),
            nn.ReLU(True),
            nn.Linear(128, d),
            nn.Tanh())

    def forward(self, x):
        x = self.layers(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


class Decoder(nn.Module):
    def __init__(self, d=D):
        super().__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(d, 128),
            nn.ReLU(True),
            nn.Linear(128, 3 * 3 * 32),
            nn.ReLU(True)
        )

        self.unflatten = nn.Unflatten(dim=1,
                                      unflattened_size=(32, 3, 3))

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3,
                               stride=2, output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2,
                               padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 3, stride=2,
                               padding=1, output_padding=1)
        )

    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x


def load_data():
    batch_size_train = 64
    batch_size_test = 1000
    trainloader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./data', train=True, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=batch_size_train, shuffle=True)

    testloader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./data', train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=batch_size_test, shuffle=True)
    return trainloader, testloader


def train_and_test(trainloader, testloader, d=D):
    # set wandb new plot per current d value

    random_seed = 1
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)

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
        return cur_train_loss / len(trainloader)

    def test():
        # model.eval()  # todo: check this out
        test_loss = 0
        with torch.no_grad():
            for img, _ in testloader:
                latent = encoder(img)
                output = decoder(latent)
                test_loss += criterion(output, img).item()

        test_loss /= len(testloader.dataset)
        print('\nTest set: Avg. loss: {:.4f}\n'.format(test_loss))

        return test_loss

    for epoch in range(EPOCHS):
        train_loss = train()
        test_loss = test()
        if WB:
            wandb.log({"train_loss": train_loss, 'test_loss': test_loss})
            # wandb.log({"acc": acc, "loss": loss})


def init_w_and_b():
    if WB:
        wandb.init(
            # Set the project where this run will be logged
            group="Auto-Encoding",
            project="NN4I_Ex2 ",
            # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
            name=f"{DESCRIPTION}{RUN}_{EPOCHS}_epochs",
            notes='checking if log is work properly',
            # Track hyperparameters and run metadata
            config={
                "learning_rate": LR,
                "architecture": "AE",
                "dataset": "MNIST",
                "epochs": EPOCHS,

            })


def main():
    init_w_and_b()
    trainloader, testloader = load_data()
    for d in range(5, 15):
        train_and_test(trainloader, testloader, d)
    if WB:
        wandb.finish()


if __name__ == '__main__':
    if WB:
        wandb.login()
    main()
