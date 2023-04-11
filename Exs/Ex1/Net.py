import torch.nn as nn

CLASSES_NUM = 10


# class ComplexNet(nn.Module):
#     def __init__(self):
#         super(ComplexNet, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm2d(32)
#         self.relu1 = nn.ReLU()
#         self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm2d(32)
#         self.relu2 = nn.ReLU()
#         self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
#         self.bn3 = nn.BatchNorm2d(64)
#         self.relu3 = nn.ReLU()
#         self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
#         self.bn4 = nn.BatchNorm2d(64)
#         self.relu4 = nn.ReLU()
#         self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
#         # Fully-connected layer
#         self.fc1 = nn.Linear(in_features=65536, out_features=10)
#
#     def forward(self, x):
#         import torch
#         # Convolutional layers
#         x = self.conv1(x)
#         # x = self.bn1(x)
#         # x = self.relu1(x)
#         x = self.conv2(x)
#         # x = self.bn2(x)
#         # x = self.relu2(x)
#         # x = self.pool1(x)
#         x = self.conv3(x)
#         # x = self.bn3(x)
#         # x = self.relu3(x)
#         x = self.conv4(x)
#         # x = self.bn4(x)
#         # x = self.relu4(x)
#         # x = self.pool2(x)
#         # Flatten
#         x = torch.flatten(x, start_dim=1)
#         # x = x.view(x.size(0), -1)
#         # Fully-connected layer
#         x = self.fc1(x)
#
#         return x

class ComplexNet(nn.Module):
    def __init__(self):
        super(ComplexNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1)

        # Fully-connected layer
        self.fc1 = nn.Linear(in_features=16384, out_features=10)

    def forward(self, x):
        import torch
        # Convolutional layers
        x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.relu1(x)
        x = self.conv2(x)
        # x = self.bn2(x)
        # x = self.relu2(x)
        # x = self.pool1(x)
        x = self.conv3(x)
        # x = self.bn3(x)
        # x = self.relu3(x)
        x = self.conv4(x)
        # x = self.bn4(x)
        # x = self.relu4(x)
        # x = self.pool2(x)
        x= self.conv5(x)
        x= self.conv6(x)

        # Flatten
        x = torch.flatten(x, start_dim=1)
        # x = x.view(x.size(0), -1)
        # Fully-connected layer
        x = self.fc1(x)

        return x

# Basic Net with one Linear FC layer

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(256, 4, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(4 * 8 * 8, CLASSES_NUM)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.softmax(x)
        return x


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(3600, 10)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        import torch
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)

        x = self.fc1(x)
        x = self.softmax(x)
        return x


class AssesCascade1(nn.Module):
    def __init__(self):
        super(ComplexNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        # self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        # self.bn2 = nn.BatchNorm2d(32)
        # self.relu2 = nn.ReLU()
        # self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # self.bn3 = nn.BatchNorm2d(64)
        # self.relu3 = nn.ReLU()
        # self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        # self.bn4 = nn.BatchNorm2d(64)
        # self.relu4 = nn.ReLU()
        # self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.conv5 = nn.Conv2d(in_channels=64, out_channels=16, kernel_size=3, padding=1)
        # Fully-connected layer
        self.fc1 = nn.Linear(in_features=32768, out_features=10)

    def forward(self, x):
        import torch
        # Convolutional layers
        x = self.conv1(x)
        # x = self.bn1(x)
        x = self.relu1(x)
        # x = self.conv2(x)
        # x = self.bn2(x)
        # x = self.relu2(x)
        # x = self.pool1(x)
        # x = self.conv3(x)
        # x = self.bn3(x)
        # x = self.relu3(x)
        # x = self.conv4(x)
        # x = self.bn4(x)
        # x = self.relu4(x)
        # x = self.pool2(x)
        # Flatten
        x = torch.flatten(x, start_dim=1)
        # x = x.view(x.size(0), -1)
        # Fully-connected layer
        x = self.fc1(x)

        return x


class AssesCascade2(nn.Module):
    def __init__(self):
        super(ComplexNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        # self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        # self.bn2 = nn.BatchNorm2d(32)
        # self.relu2 = nn.ReLU()
        # self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # self.bn3 = nn.BatchNorm2d(64)
        # self.relu3 = nn.ReLU()
        # self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        # self.bn4 = nn.BatchNorm2d(64)
        # self.relu4 = nn.ReLU()
        # self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.conv5 = nn.Conv2d(in_channels=64, out_channels=16, kernel_size=3, padding=1)
        # Fully-connected layer
        self.fc1 = nn.Linear(in_features=32768, out_features=10)

    def forward(self, x):
        import torch
        # Convolutional layers
        x = self.conv1(x)
        # x = self.bn1(x)
        x = self.relu1(x)
        # x = self.conv2(x)
        # x = self.bn2(x)
        # x = self.relu2(x)
        # x = self.pool1(x)
        # x = self.conv3(x)
        # x = self.bn3(x)
        # x = self.relu3(x)
        # x = self.conv4(x)
        # x = self.bn4(x)
        # x = self.relu4(x)
        # x = self.pool2(x)
        # Flatten
        x = torch.flatten(x, start_dim=1)
        # x = x.view(x.size(0), -1)
        # Fully-connected layer
        x = self.fc1(x)

        return x


if __name__ == '__main__':
    net = SimpleNet()
    print(sum(p.numel() for p in net.parameters() if p.requires_grad))
    pass
