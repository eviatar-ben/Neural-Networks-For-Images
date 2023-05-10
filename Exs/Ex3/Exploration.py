# Non-Saturating loss

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.utils as vutils
import wandb

WB = False
# WB = True

PLOT = not WB
LOSSES = ["Original", "Non-saturation", "L2"]
LOSS_TYPE = 2
LOSS = LOSSES[LOSS_TYPE]
RUN = f'{LOSS}_LOSS'
EPOCHS = 10
LR = 1e-3
G_D_FACTOR = 4
DESCRIPTION = f"GD_FACTOR{G_D_FACTOR}_RUN_{RUN}"
D = 15
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# input noise dimension
nz = 100

# number of gpu's available
ngpu = 1

real_label = 1
fake_label = 0


class Discriminator(nn.Module):
    def __init__(self, ngpu, nc=1, ndf=64):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, 1, 4, 2, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output.view(-1, 1).squeeze(1)


class Generator(nn.Module):
    def __init__(self, ngpu, nc=1, nz=100, ngf=64):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, kernel_size=1, stride=1, padding=2, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


def init_w_and_b(group, project, d="", description="", run="", LR=1e-3, EPOCHS=10, architecture="GAN"):
    wandb.init(
        # settings=wandb.Settings(start_method="fork"),
        # Set the project where this run will be logged
        group=group,
        project=project,
        name=f"{description}{run}{d}",
        notes='',
        # Track hyperparameters and run metadata
        config={
            "learning_rate": LR,
            "architecture": architecture,
            "dataset": "MNIST",
            "epochs": EPOCHS,
        })


def load_data(batch_size=64):
    from torchvision import datasets, transforms

    train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())

    trainloader = torch.utils.data.DataLoader(dataset=train_data,
                                              batch_size=batch_size,
                                              shuffle=True)

    test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

    testloader = torch.utils.data.DataLoader(dataset=test_data,
                                             batch_size=batch_size,
                                             shuffle=True)
    return trainloader, testloader


def train():
    dataloader, _ = load_data()
    criterion = nn.BCELoss()  # todo: maybe logit is required?

    netD = Discriminator(ngpu).to(device)
    netG = Generator(ngpu).to(device)

    optimizerD = torch.optim.Adam(netD.parameters(), lr=LR, betas=(0.5, 0.999))
    optimizerG = torch.optim.Adam(netG.parameters(), lr=LR, betas=(0.5, 0.999))

    fixed_noise = torch.randn(64, 100, 1, 1, device=device)

    for epoch in range(EPOCHS):
        for i, data in enumerate(dataloader, 0):
            #############################################################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ############################################################
            # train D with real
            netD.zero_grad()
            real_cpu = data[0].to(device)
            batch_size = real_cpu.size(0)
            label = torch.full((batch_size,), real_label, device=device).to(torch.float32)

            output = netD(real_cpu)
            if LOSS_TYPE == 2:
                errD_real = 0.5 * torch.mean((output - label) ** 2)  # criterion(output, label)
            else:
                errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            # train D with fake
            noise = torch.randn(batch_size, nz, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach())
            if LOSS_TYPE == 2:
                errD_fake = 0.5 * torch.mean((output - label) ** 2)  # criterion(output, label)
            else:
                errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            ############################################################
            # (2) Update G network:
            # i.  minimize log(1-D(G(z))) equals maximize -log(1-D(G(z))) minimize -BCELoss(1-D(G(z)))
            # ii. minimize -log(D(G(z))) equals maximize log(D(G(z))) minimize BCELoss(D(G(z)))
            # iii. minimize (D(G(z))-1)^2
            ############################################################
            # train G
            if i % G_D_FACTOR == 0:
                netG.zero_grad()
                label.fill_(real_label)
                output = netD(fake)
                if LOSS_TYPE == 0:
                    # original GAN loss
                    label.fill_(fake_label)  # fake labels are real for generator cost
                    errG = -criterion(output, label)  # note the negative sign

                elif LOSS_TYPE == 1:
                    # non-saturation GAN loss
                    errG = criterion(output, label)

                elif LOSS_TYPE == 2:
                    # least squares
                    errG = 0.5 * torch.mean((output - label) ** 2)
                else:
                    return

                errG.backward()
                D_G_z2 = output.mean().item()
                optimizerG.step()

            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                  % (epoch, EPOCHS, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # if i % 100 == 0:
            #     vutils.save_image(real_cpu, 'output/real_samples.png', normalize=True)
            #     fake = netG(fixed_noise)
            #     vutils.save_image(fake.detach(),
            #     f'output/fake_samples_epoch_{epoch}_iteration_{i}.png', normalize=True)

            # if i % 100 == 0:
            #
            #     vutils.save_image(real_cpu, f'/content/drive/MyDrive/NN4I/{LOSS}/real_samples.png', normalize=True)
            #     fake = netG(fixed_noise)
            #     vutils.save_image(fake.detach(), f'/content/drive/MyDrive/NN4I/{LOSS}/fake_samples_epoch_{epoch}_iteration_{i}_loss_{LOSS}.png', normalize=True)

        if WB:
            wandb.log({"Loss_D": errD.item(), 'Loss_G': errG.item(), },
                      step=epoch)
            # wandb.log({"Loss_D": errD.item(), 'Loss_G': errG.item(),
            #            "D(x)":D_x, "D(G(z))":D_G_z2 }, step=epoch)


def main():
    if WB:
        init_w_and_b(project="NN4I_Ex3 ", group="Loss Saturation",
                     d=DESCRIPTION, description=DESCRIPTION, run=RUN, LR=LR,
                     EPOCHS=EPOCHS, architecture="GAN")
    train()


if __name__ == '__main__':
    main()
