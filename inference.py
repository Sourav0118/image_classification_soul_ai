import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import os
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
import torchvision.transforms as tt
import matplotlib.pyplot as plt
main_folder = os.getcwd()

def conv_2d(ni, nf, stride=1, ks=3):
    # torch.nn.Conv2d(in_channels = n, out_channels = m, kernel_size = k, stride = s, padding = p)
    return nn.Conv2d(ni, nf, ks, stride, ks//2, bias=False)

def bn_relu_conv(ni, nf):
    return nn.Sequential(nn.BatchNorm2d(ni), nn.ReLU(inplace=True), conv_2d(ni, nf))
# nn. Sequential is a construction which is used when you want to run certain layers sequentially.
# It makes the forward to be readable and compact.

class ResidualBlock(nn.Module):
    def __init__(self, ni, nf, stride=1):   # ni = no. of input channels, nf = no. of output channels
        super().__init__()
        self.bn = nn.BatchNorm2d(ni)
        self.conv1 = conv_2d(ni, nf, stride)
        self.conv2 = bn_relu_conv(nf, nf)
        self.shortcut = lambda x: x
        if ni!=nf:
            self.shortcut = conv_2d(ni, nf, stride, 1)

    def forward(self, x):
        x = F.relu(self.bn(x), inplace=True)
        r = self.shortcut(x)
        x = self.conv1(x)
        x = self.conv2(x) * 0.2
        return x.add_(r)


def make_group(N, ni, nf, stride):
    start = ResidualBlock(ni, nf, stride)
    rest = [ResidualBlock(nf, nf) for j in range(1,N)]
    return [start] + rest

class Flatten(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, x):
        return x.view(x.size(0), -1)

class WideResNet(nn.Module):
    def __init__(self, n_groups, N, n_classes, k=1, n_start=16):
        super().__init__()
        layers = [conv_2d(3, n_start)]
        n_channels = [n_start]

        for i in range(n_groups):
            n_channels.append(n_start*(2**i)*k)
            stride = 2 if i>0 else 1
            layers += make_group(N, n_channels[i], n_channels[i+1], stride)

        layers += [nn.BatchNorm2d(n_channels[3]),
                   nn.ReLU(inplace=True),
                   nn.AdaptiveAvgPool2d(1),
                   Flatten(),
                   nn.Linear(n_channels[3], n_classes)]

        self.features = nn.Sequential(*layers)

    def forward(self, x):
        return self.features(x)

def wrn_22():
    return WideResNet(3, N=3, n_classes=10, k=6)

model = wrn_22()

class SingleImageDataset(Dataset):
    def __init__(self, img, transform=None):
        self.image = img
        self.transform = transform
        self.classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    def __len__(self):
        return 1  # Since we only have one image

    def __getitem__(self, idx):
        # Return the image and the corresponding label (if available)
        if self.transform:
            image = self.transform(self.image)  # Apply transformations if provided
        else:
            image = self.image

        return image

stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
tfms = tt.Compose([tt.ToTensor(), tt.Normalize(*stats)])
def evaluate(img):
    model = wrn_22()
    dataset = SingleImageDataset(img, transform=tfms)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    model_path = os.path.join( main_folder, 'best_model.pt' )
    if torch.cuda.is_available():
        state_dict = torch.load(model_path)
    else:
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    for x in dataloader:
        with torch.no_grad():
            output = model(x)
        pred = torch.argmax(output, dim=1)
    return dataset.classes[pred[0].item()]
