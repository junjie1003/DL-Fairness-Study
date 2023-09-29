import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torchvision.models import resnet18


class ResNet18(nn.Module):
    def __init__(self, num_classes=2, pretrained=False):
        super(ResNet18, self).__init__()

        self.resnet = resnet18(pretrained=pretrained)
        self.num_features = self.resnet.fc.in_features

        # Get feature extractor
        self.feature_extractor = nn.Sequential(*list(self.resnet.children())[:-1])

        # Get label predictor
        self.label_predictor = nn.Linear(self.num_features, num_classes)
        print(f"ResNet18 - num_classes: {num_classes} pretrained: {pretrained}")

    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        logits = self.label_predictor(features)

        return logits, features


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        # make sure input tensor is flattened
        x = x.view(x.shape[0], -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class Generator(nn.Module):
    def __init__(self, channels=3, dropout_prob=0.5):
        super(Generator, self).__init__()

        encoder_list = [
            # 3 × 224 × 224
            nn.Conv2d(channels, 32, kernel_size=4, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(32),
            nn.ReLU(),
            # 32 × 112 × 112
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            # 64 × 56 × 56
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(128),
            nn.ReLU(),
            # 128 × 28 × 28
        ]

        bottleneck_list = [ResnetBlock(128, use_dropout=True, dropout_prob=dropout_prob)]

        decoder_list = [
            # 128 × 28 × 28
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            # 64 × 56 × 56
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(32),
            nn.ReLU(),
            # 32 × 112 × 112
            nn.ConvTranspose2d(32, channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
            # 3 × 224 × 224
        ]

        self.encoder = nn.Sequential(*encoder_list)
        self.bottleneck = nn.Sequential(*bottleneck_list)
        self.decoder = nn.Sequential(*decoder_list)

    def forward(self, x):
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x)
        return x


# Define a simplified resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, norm_layer=nn.BatchNorm2d, use_dropout=False, dropout_prob=0.5, use_bias=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, norm_layer, use_dropout, dropout_prob, use_bias)

    def build_conv_block(self, dim, norm_layer, use_dropout, dropout_prob, use_bias):
        conv_block = [nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=use_bias), norm_layer(dim), nn.ReLU(True)]

        if use_dropout:
            conv_block += [nn.Dropout(dropout_prob)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


if __name__ == "__main__":
    model = Generator(channels=3)
    summary(model=model, input_size=(3, 224, 224), batch_size=64, device="cpu")
