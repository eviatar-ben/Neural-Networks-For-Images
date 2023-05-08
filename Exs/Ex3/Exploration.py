import Exs.utils as utils
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.utils as vutils

WB = False
# WB = True

PLOT = not WB

RUN = 'GAN'
EPOCHS = 10
LR = 1e-3
DESCRIPTION = "Latent_dimension_higher_dims"
D = 15
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# checking the availability of cuda devices
device = 'cuda' if torch.cuda.is_available() else 'cpu'
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


def train():
    dataloader, _ = utils.load_data()
    criterion = nn.BCELoss()

    netD = Discriminator(ngpu).to(device)
    netG = Generator(ngpu).to(device)

    optimizerD = torch.optim.Adam(netD.parameters(), lr=LR, betas=(0.5, 0.999))
    optimizerG = torch.optim.Adam(netG.parameters(), lr=LR, betas=(0.5, 0.999))

    fixed_noise = torch.randn(64, 100, 1, 1, device=device)

    for epoch in range(EPOCHS):
        for i, data in enumerate(dataloader, 0):

            # train D with real
            netD.zero_grad()
            real_cpu = data[0].to(device)
            batch_size = real_cpu.size(0)
            label = torch.full((batch_size,), real_label, device=device).to(torch.float32)

            output = netD(real_cpu)
            # output = output.to(torch.float32)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            # train D with fake
            noise = torch.randn(batch_size, nz, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            # train G
            netG.zero_grad()
            label.fill_(real_label)
            output = netD(fake)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                  % (epoch, EPOCHS, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            if i % 100 == 0:
                vutils.save_image(real_cpu, 'output/real_samples.png', normalize=True)
                fake = netG(fixed_noise)
                vutils.save_image(fake.detach(), 'output/fake_samples_epoch_%03d.png' % (epoch), normalize=True)
        torch.save(netG.state_dict(), '.models/weights/netG_epoch_%d.pth' % (epoch))
        torch.save(netD.state_dict(), 'weights/netD_epoch_%d.pth' % (epoch))


def main():
    if WB:
        utils.init_w_and_b(project="NN4I_Ex3 ", group="Loss Saturation",
                           d=DESCRIPTION, description=DESCRIPTION, run=RUN, LR=LR,
                           EPOCHS=EPOCHS, architecture="GAN")
    train()


if __name__ == '__main__':
    main()
