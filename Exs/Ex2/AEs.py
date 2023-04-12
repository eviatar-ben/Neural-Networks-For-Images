import torch
import torch.nn as nn

D = 15


class Encoder1L(nn.Module):

    def __init__(self, d=D):
        super(Encoder1L, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=2, padding=1)
        # self.bn1 = nn.BatchNorm2d(8)
        self.relu1 = nn.ReLU(True)
        self.flatten = nn.Flatten(start_dim=1)

        self.fc1 = nn.Linear(14*14*8, 128)
        self.relu4 = nn.ReLU(True)
        self.fc2 = nn.Linear(128, d)

    def forward(self, x):
        # con' layers
        x = self.conv1(x)
        # x = self.bn1(x)
        x = self.relu1(x)
        # Flatten
        x = self.flatten(x)
        # Fully-connected layer
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.fc2(x)
        return x


class Decoder1L(nn.Module):
    def __init__(self, d=D):
        super().__init__()
        self.fc2 = nn.Linear(d, 128)
        self.relu4 = nn.ReLU(True)
        self.fc1 = nn.Linear(128, 14 * 14 * 8)
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(8, 14, 14))

        # self.bn1 = nn.BatchNorm2d(8)
        self.relu1 = nn.ReLU(True)
        self.conv1 = nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        x = self.fc2(x)
        x = self.relu4(x)
        x = self.fc1(x)

        x = self.unflatten(x)

        # x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv1(x)
        return x


class Encoder2L(nn.Module):

    def __init__(self, d=D):
        super(Encoder2L, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True))

        self.flatten = nn.Flatten(start_dim=1)

        self.fc = nn.Sequential(
            nn.Linear(3 * 3 * 16, 128),
            nn.ReLU(True),
            nn.Linear(128, d),
        )

    def forward(self, x):
        x = self.layers(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


class Decoder2L(nn.Module):
    def __init__(self, d=D):
        super().__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(d, 128),
            nn.ReLU(True),
            nn.Linear(128, 3 * 3 * 16),
            nn.ReLU(True)
        )

        self.unflatten = nn.Unflatten(dim=1,
                                      unflattened_size=(16, 3, 3))

        self.decoder_conv = nn.Sequential(
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


class Encoder3L(nn.Module):

    def __init__(self, d=D):
        super(Encoder3L, self).__init__()

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
        )

    def forward(self, x):
        x = self.layers(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


class Decoder3L(nn.Module):
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


class Encoder4L(nn.Module):

    def __init__(self, d=D):
        super(Encoder4L, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=0),
            nn.ReLU(True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=0),
            nn.ReLU(True))

        self.flatten = nn.Flatten(start_dim=1)

        self.fc = nn.Sequential(
            nn.Linear(3 * 3 * 64, 128),
            nn.ReLU(True),
            nn.Linear(128, d),
        )

    def forward(self, x):
        x = self.layers(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


class Decoder4L(nn.Module):
    def __init__(self, d=D):
        super().__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(d, 128),
            nn.ReLU(True),
            nn.Linear(128, 3 * 3 * 32),
            nn.ReLU(True)
        )

        self.unflatten = nn.Unflatten(dim=1,
                                      unflattened_size=(64, 3, 3))

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3,
                               stride=2, output_padding=0),
            nn.ReLU(True),
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
