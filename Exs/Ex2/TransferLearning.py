import torch
import torch.nn as nn
import wandb

WB = False
# WB = True

PLOT = not WB

RUN = 'MLP'
EPOCHS = 10
LR = 1e-3
DESCRIPTION = "Transfer learning"
D = 10
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

DATA_FRACTION = 10
FRACTIONS_NUMBER = 10


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


class MLP(nn.Module):
    def __init__(self, d=D):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(d, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10)
        )

    def forward(self, x):
        return self.layers(x)


def load_data():
    from torchvision import datasets, transforms

    train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())

    trainloader = torch.utils.data.DataLoader(dataset=train_data,
                                              batch_size=DATA_FRACTION,
                                              shuffle=True)

    test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

    testloader = torch.utils.data.DataLoader(dataset=test_data,
                                             batch_size=64,
                                             shuffle=True)
    return trainloader, testloader


def test(testloader, encoder, mlp):
    test_loss = 0
    criterion = nn.CrossEntropyLoss()  # maybe MSE?
    with torch.no_grad():
        for img, digits in testloader:
            img = img.to(device)

            latent = encoder(img).to(device)
            output = mlp(latent).to(device)

            loss = criterion(output, torch.nn.functional.one_hot(digits, 10).float())
            test_loss += loss.item()
        print('\nTest set: Avg. test loss: {:.4f}\n'.format(test_loss / len(testloader)))
    return test_loss / len(testloader)


def train(trainloader, mlp, mlp_encoder, encoder, decoder):
    criterion = nn.CrossEntropyLoss()  # maybe MSE?
    mlp_optimizer = torch.optim.Adam(list(mlp.parameters()), lr=LR)
    ae_optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=LR)
    mlp_train_loss = 0
    ae_train_loss = 0
    data_fractions = 0
    for img, digits in trainloader:
        img = img.to(device)

        mlp_optimizer.zero_grad()

        ae_optimizer.zero_grad()

        latent_for_mlp = encoder(img).to(device)
        mlp_output = mlp(latent_for_mlp).to(device)

        latent_for_ae = encoder(img).to(device)
        ae_output = decoder(latent_for_ae).to(device)

        mlp_loss = criterion(mlp_output, torch.nn.functional.one_hot(digits, 10).float())
        mlp_loss.backward()
        mlp_optimizer.step()
        mlp_train_loss += mlp_loss.item()

        ae_loss = criterion(ae_output, torch.nn.functional.one_hot(digits, 10).float())
        ae_loss.backward()
        ae_optimizer.step()
        ae_train_loss += ae_loss.item()

        if data_fractions > FRACTIONS_NUMBER:  # "Do this with a small fraction of the annotated training data (~ tens of images)"
            break
        data_fractions += 1
    print('\nMLP Train set: Avg. train loss: {:.4f}\n'.format(mlp_train_loss / (FRACTIONS_NUMBER * DATA_FRACTION)))
    return mlp_train_loss / len(trainloader)


def train_mlp_and_test(trainloader, testloader, d=D):
    # random_seed = 1
    # torch.backends.cudnn.enabled = False
    # torch.manual_seed(random_seed)
    encoder = Encoder(d).to(device)
    mlp_encoder = Encoder(d).to(device)
    decoder = Decoder(d).to(device)
    mlp = MLP(d).to(device)
    outputs = []
    for epoch in range(EPOCHS):
        train_loss = train(trainloader, mlp, mlp_encoder, encoder, decoder)
        # test_loss = test(testloader, encoder, mlp)
        if WB:
            pass  # wandb.log({"train_loss": train_loss, 'test_loss': test_loss}, step=epoch)
    # if PLOT:
    #     plot_images(outputs)


def init_w_and_b(d=D):
    if WB:
        wandb.init(
            # Set the project where this run will be logged
            group="Transfer Learning",
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
    if WB:
        init_w_and_b()
    trainloader, testloader = load_data()
    train_mlp_and_test(trainloader, testloader, d=D)


if __name__ == '__main__':
    if WB:
        wandb.login()
    main()
