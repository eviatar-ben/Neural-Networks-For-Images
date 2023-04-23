import torch
import torch.nn as nn
import wandb

# WB = False
WB = True

PLOT = not WB

# PRE_TRAIN = True
PRE_TRAIN = False

RUN = 'MLP'
EPOCHS = 10
LR = 1e-3
DESCRIPTION = "Losses in small batches pre-trained encoder from previous task"
D = 15
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

DATA_FRACTION = 10
FRACTIONS_NUMBER = 4


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
            nn.Linear(32, 10),
            torch.nn.Sigmoid()  # maybe without sigmoid
        )

    def forward(self, x):
        return self.layers(x)


def init_w_and_b(d=D):
    if WB:
        wandb.init(
            # Set the project where this run will be logged
            group="Transfer Learning , pre-trained encoder",
            project="NN4I_Ex2 ",
            name=f"{DESCRIPTION} {RUN} D{d}",
            notes='',
            # Track hyperparameters and run metadata
            config={
                "learning_rate": LR,
                "architecture": "AE",
                "dataset": "MNIST",
                "epochs": EPOCHS,

            })


def load_data(batch_size=DATA_FRACTION):
    from torchvision import datasets, transforms

    train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())

    trainloader = torch.utils.data.DataLoader(dataset=train_data,
                                              batch_size=batch_size,
                                              shuffle=True)

    test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

    testloader = torch.utils.data.DataLoader(dataset=test_data,
                                             batch_size=64,
                                             shuffle=True)
    return trainloader, testloader


def test(testloader, learnable_encoder, learnable_encoder_mlp, non_learnable_encoder, non_learnable_encoder_mlp):
    test_loss = 0
    criterion = nn.CrossEntropyLoss()  # maybe MSE?
    # with torch.no_grad():
    #     for img, digits in testloader:
    #         img = img.to(device)
    #
    #         latent = encoder(img).to(device)
    #         output = mlp(latent).to(device)
    #
    #         loss = criterion(output, torch.nn.functional.one_hot(digits, 10).float())
    #         test_loss += loss.item()
    #     print('\nTest set: Avg. test loss: {:.4f}\n'.format(test_loss / len(testloader)))

    non_learnable_encoder_criterion = nn.CrossEntropyLoss()  # maybe MSE?
    learnable_encoder_criterion = nn.CrossEntropyLoss()  # maybe MSE?

    non_learnable_encoder_test_loss = 0
    learnable_encoder_test_loss = 0
    data_fractions = 0
    with torch.no_grad():
        for img, digits in testloader:
            img = img.to(device)
            # non learnable encoder:
            latent_non_learnable_encoder = non_learnable_encoder(img).to(device)
            output_non_learnable_encoder = non_learnable_encoder_mlp(latent_non_learnable_encoder).to(device)

            non_learnable_encoder_loss = non_learnable_encoder_criterion(output_non_learnable_encoder,
                                                                         torch.nn.functional.one_hot(digits,
                                                                                                     10).float())
            non_learnable_encoder_test_loss += non_learnable_encoder_loss.item()

            # learnable encoder:
            latent_learnable_encoder = learnable_encoder(img).to(device)
            output_learnable_encoder = learnable_encoder_mlp(latent_learnable_encoder).to(device)

            learnable_encoder_loss = learnable_encoder_criterion(output_learnable_encoder.squeeze(),
                                                                 torch.nn.functional.one_hot(digits, 10).float())
            learnable_encoder_test_loss += learnable_encoder_loss.item()

            if data_fractions > FRACTIONS_NUMBER:
                # "Do this with a small fraction of the annotated training data (~ tens of images)"
                break
            data_fractions += 1
    # print(f"non learnable encoder Train loss{non_learnable_encoder_train_loss}")
    # print(f"learnable encoder Train loss{learnable_encoder_train_loss}")

    return learnable_encoder_test_loss, non_learnable_encoder_test_loss


def train(trainloader, learnable_encoder, learnable_encoder_mlp, non_learnable_encoder, non_learnable_encoder_mlp):
    non_learnable_encoder_criterion = nn.CrossEntropyLoss()  # maybe MSE?
    learnable_encoder_criterion = nn.CrossEntropyLoss()  # maybe MSE?

    non_learnable_encoder_optimizer = torch.optim.Adam(list(non_learnable_encoder_mlp.parameters()), lr=LR)
    learnable_encoder_optimizer = torch.optim.Adam(list(learnable_encoder.parameters()) +
                                                   list(learnable_encoder_mlp.parameters()), lr=LR)

    non_learnable_encoder_train_loss = 0
    learnable_encoder_train_loss = 0
    data_fractions = 0
    for img, digits in trainloader:
        img = img.to(device)
        # non learnable encoder:
        non_learnable_encoder_optimizer.zero_grad()

        latent_non_learnable_encoder = non_learnable_encoder(img).to(device)
        output_non_learnable_encoder = non_learnable_encoder_mlp(latent_non_learnable_encoder).to(device)

        non_learnable_encoder_loss = non_learnable_encoder_criterion(output_non_learnable_encoder,
                                                                     torch.nn.functional.one_hot(digits, 10).float())
        non_learnable_encoder_loss.backward()
        non_learnable_encoder_optimizer.step()
        non_learnable_encoder_train_loss += non_learnable_encoder_loss.item()

        # learnable encoder:
        learnable_encoder_optimizer.zero_grad()

        latent_learnable_encoder = learnable_encoder(img).to(device)
        output_learnable_encoder = learnable_encoder_mlp(latent_learnable_encoder).to(device)

        learnable_encoder_loss = learnable_encoder_criterion(output_learnable_encoder.squeeze(),
                                                             torch.nn.functional.one_hot(digits, 10).float())
        learnable_encoder_loss.backward()
        learnable_encoder_optimizer.step()
        learnable_encoder_train_loss += learnable_encoder_loss.item()

        if data_fractions > FRACTIONS_NUMBER:
            # "Do this with a small fraction of the annotated training data (~ tens of images)"
            break
        data_fractions += 1
    # print(f"non learnable encoder Train loss{non_learnable_encoder_train_loss}")
    # print(f"learnable encoder Train loss{learnable_encoder_train_loss}")

    return learnable_encoder_train_loss, non_learnable_encoder_train_loss


def train_mlp_and_test(trainloader, testloader, d=D):
    # random_seed = 1
    # torch.backends.cudnn.enabled = False
    # torch.manual_seed(random_seed)

    learnable_encoder = Encoder(d).to(device)
    # learnable_encoder.load_state_dict()
    learnable_mlp = MLP(d).to(device)

    non_learnable_encoder = Encoder(d).to(device)
    # non_learnable_encoder.load_state_dict(torch.load(f'./models/encoder{RUN}_{d}D.pth'))  # "pretrained"
    non_learnable_encoder.load_state_dict(torch.load(f'./models/encoderAE_15D.pth'))  # "pretrained"
    non_learnable_encoder_mlp = MLP(d).to(device)
    for epoch in range(EPOCHS):
        train_losses = train(trainloader, learnable_encoder, learnable_mlp, non_learnable_encoder,
                             non_learnable_encoder_mlp)
        test_losses = test(testloader, learnable_encoder, learnable_mlp, non_learnable_encoder,
                           non_learnable_encoder_mlp)
        if WB:
            wandb.log({"Full learning train loss": train_losses[0], 'MLP learning only train loss': train_losses[1],
                       "Full learning test loss": test_losses[0], 'MLP learning only test loss': test_losses[1]},
                      step=epoch)


def pre_train(trainloader, learnable_encoder, pre_trained_mlp):
    pre_trained_encoder_criterion = nn.CrossEntropyLoss()  # maybe MSE?

    pre_trained_encoder_optimizer = torch.optim.Adam(list(learnable_encoder.parameters()), lr=LR)

    pre_trained_encoder_train_loss = 0
    data_fractions = 0
    for img, digits in trainloader:
        img = img.to(device)
        pre_trained_encoder_optimizer.zero_grad()

        latent_pre_trained_encoder = learnable_encoder(img).to(device)
        output_pre_trained_encoder = pre_trained_mlp(latent_pre_trained_encoder).to(device)

        pre_trained_encoder_loss = pre_trained_encoder_criterion(output_pre_trained_encoder.squeeze(),
                                                                 torch.nn.functional.one_hot(digits, 10).float())
        pre_trained_encoder_loss.backward()
        pre_trained_encoder_optimizer.step()
        pre_trained_encoder_train_loss += pre_trained_encoder_loss.item()

        data_fractions += 1
    # print(f"non learnable encoder Train loss{non_learnable_encoder_train_loss}")
    # print(f"learnable encoder Train loss{learnable_encoder_train_loss}")

    return pre_trained_encoder_train_loss


def get_pre_train(d=D):
    learnable_encoder = Encoder(d).to(device)
    learnable_mlp = MLP(d).to(device)
    trainloader, testloader = load_data(batch_size=64)
    for epoch in range(EPOCHS):
        train_losses = pre_train(trainloader, learnable_encoder, learnable_mlp)
        print(train_losses)
    torch.save(learnable_encoder.state_dict(), f'./models/encoder{RUN}_{d}D.pth')


def main():
    if WB:
        init_w_and_b()
    if PRE_TRAIN:
        get_pre_train(d=D)
    trainloader, testloader = load_data()
    train_mlp_and_test(trainloader, testloader, d=D)


if __name__ == '__main__':
    if WB:
        wandb.login()
    main()
