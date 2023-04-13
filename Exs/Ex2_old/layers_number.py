import torch
import torchvision
import torch.nn as nn
import wandb

from Exs.Ex2_old.AEs import Encoder1L, Encoder2L, Encoder3L, Encoder4L, Decoder1L, Decoder2L, Decoder3L, Decoder4L

WB = False
# WB = True

RUN = 'AE'
EPOCHS = 9
LR = 1e-3
DESCRIPTION = "different layers number"
D = 10
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Stride is the number of pixels shifts over the input matrix.
# For padding p, filter size 𝑓∗𝑓 and input image size 𝑛 ∗ 𝑛 and stride ‘𝑠’
# our output image dimension will be
# [ {(𝑛 + 2𝑝 − 𝑓 + 1) / 𝑠} + 1] ∗ [ {(𝑛 + 2𝑝 − 𝑓 + 1) / 𝑠} + 1].




def load_data():
    batch_size_train = 64
    batch_size_test = 64
    trainloader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./data', train=True, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor()
                                       # ,torchvision.transforms.Normalize((0.5,), (0.5,))
                                   ])),
        batch_size=batch_size_train, shuffle=True, pin_memory=True)

    testloader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./data', train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor()
                                       # ,torchvision.transforms.Normalize((0.5,), (0.5,))
                                   ])),
        batch_size=batch_size_test, shuffle=True, pin_memory=True)
    return trainloader, testloader


def train_and_test(trainloader, testloader, d=D, cl=3):
    random_seed = 1
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)
    if cl == 1:
        encoder = Encoder1L(d).to(device)
        decoder = Decoder1L(d).to(device)
        print("encoder1")
    elif cl == 2:
        encoder = Encoder2L(d).to(device)
        decoder = Decoder2L(d).to(device)
        print("encoder2")
    elif cl == 3:
        encoder = Encoder3L(d).to(device)
        decoder = Decoder3L(d).to(device)
        print("encoder3")
    elif cl == 4:
        encoder = Encoder4L(d).to(device)
        decoder = Decoder4L(d).to(device)
        print("encoder4")
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
        criterion = nn.MSELoss()
        # model.eval()  # todo: check this out
        test_loss = 0
        with torch.no_grad():
            for img, _ in testloader:
                latent = encoder(img).to(device)
                output = decoder(latent).to(device)
                test_loss += criterion(output, img).item()

        test_loss /= len(testloader.dataset)
        print('\nTest set: Avg. loss: {:.4f}\n'.format(test_loss))

        return test_loss

    for epoch in range(EPOCHS):
        train_loss = train()
        test_loss = test()
        if WB:
            wandb.log({"train_loss": train_loss, 'test_loss': test_loss}, step=epoch)


def init_w_and_b(d=D):
    if WB:
        wandb.init(
            # Set the project where this run will be logged
            group="Auto-Encoding different number of layers [1-4]",
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
    for l in range(1, 2):
        # set wandb new plot per current d value
        d = 15
        init_w_and_b(d)
        train_and_test(trainloader, testloader, d, l)
        if WB:
            wandb.finish()


if __name__ == '__main__':
    if WB:
        wandb.login()
    main()
