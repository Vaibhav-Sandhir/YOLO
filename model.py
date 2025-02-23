import torch
import torch.nn as nn

architecture_config = [
    # Tuple is 1 conv layer (kernel_size, filters, stride, padding)
    # M is max pool layer of 
    # List is a collection of tuples with a number for multiplicity
    (7, 64, 2, 3),
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias = False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leaky_relu(self.batch_norm(self.conv(x)))


class YOLOv1(nn.Module):
    def __init__(self, in_channels):
        super(YOLOv1, self).__init__()
        self.architecture = architecture_config
        self.in_channels = in_channels
        self.darknet = self.create_darknet(self.architecture)
        self.fcs = self.create_fcs()

    def forward(self, x):
        x = self.darknet(x)
        return self.fcs(torch.flatten(x, start_dim = 1))

    def create_darket(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == tuple:
                layers.append(self.create_CNNBlock(x, in_channels))
                in_channels = x[1]
            elif x == "M":
                layers.append(nn.MaxPool2d(kernel_size = 2, stride = 2))
            elif type(x) == list:
                conv1 = x[0]
                conv2 = x[1]
                repeats = x[2]
                for _ in range(repeats):
                    layers.append(self.create_CNNBlock(conv1, in_channels))
                    layers.append(self.create_CNNBlock(conv2, conv1[1]))
                    in_channels = conv2[1]

        return nn.Sequential(*layers)

    def create_fcs(self, split_size, num_boxes, num_classes):
        S, B, C = split_size, num_boxes, num_classes
        fcs = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features = 1024 * S * S, out_features = 496),
            nn.Dropout(0.0),
            nn.LeakyReLU(0.1),
            nn.Linear(in_features = 496, out_features = S * S * (C + B * 5))
        )

    def create_CNNBlock(self, x, in_channels):
        return CNNBlock(in_channels = in_channels,
                        out_channels = x[1],
                        kernel_size = x[0],
                        stride = x[2],
                        padding = x[3]
                        )



        
