import torch
import torch.nn as nn
import wandb

# WB = False
WB = True

PLOT = not WB

RUN = 'AE'
EPOCHS = 10
LR = 1e-3
DESCRIPTION = "Latent_dimension_higher_dims"
D = 10
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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


def train_and_test(trainloader, testloader, d=D):
    # random_seed = 1
    # torch.backends.cudnn.enabled = False
    # torch.manual_seed(random_seed)
    encoder = Encoder(d).to(device)
    decoder = Decoder(d).to(device)
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
            group="Latent dimension [50-55]",
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
    for d in range(50, 56):
        # set wandb new plot per current d value
        init_w_and_b(d)
        train_and_test(trainloader, testloader, d)
        if WB:
            wandb.finish()


if __name__ == '__main__':
    if WB:
        wandb.login()
    main()
