from torch import nn
from torchvision.models.resnet import BasicBlock, Bottleneck


class FCNBase(nn.Module):

    def __init__(self):
        super(FCNBase, self).__init__()

    @staticmethod
    def align_shape(x, desired_shape):
        left = int((x.shape[2] - desired_shape[2]) / 2)
        top = int((x.shape[3] - desired_shape[3]) / 2)
        x = x[:, :, left:left + desired_shape[2], top:top + desired_shape[3]]
        return x

    def load_pretrained_model(self, pretrained_model):
        own_state = self.state_dict()
        for name, param in pretrained_model.items():
            if name in own_state:
                own_state[name].copy_(param.data)

    def forward(self, x):
        raise NotImplementedError


class WaterNetV1(FCNBase):

    def __init__(self):

        super(WaterNetV1, self).__init__()

        # ResNet 34
        block = BasicBlock
        layers = [3, 4, 6, 3]

        # ResNet 50
        # block = Bottleneck
        # layers = [3, 4, 6, 3]

        self.inplanes = 64

        # Conv module
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.make_conv_layer(block, 64, layers[0])
        self.layer2 = self.make_conv_layer(block, 128, layers[1], stride=2)
        self.layer3 = self.make_conv_layer(block, 256, layers[2], stride=2)
        self.layer4 = self.make_conv_layer(block, 512, layers[3], stride=2)

        self.deconv1 = self.make_deconv_layer(512, 256)
        self.deconv2 = self.make_deconv_layer(256, 128, merge_flag=True)
        self.deconv3 = self.make_deconv_layer(128, 64, merge_flag=True)
        self.deconv4 = self.make_deconv_layer(64, 32, merge_flag=True, stride=4)

        # Output mask
        self.fuse1 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.fuse2 = nn.Conv2d(32, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_conv_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    @staticmethod
    def make_deconv_layer(in_planes, out_planes, merge_flag=False, stride=2):

        layers = []
        layers.append(
            nn.ConvTranspose2d(in_planes, out_planes, padding=1, output_padding=1, kernel_size=3, stride=2)
        )
        layers.append(nn.BatchNorm2d(out_planes))
        layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def forward(self, x):

        # input_shape = x.shape
        input_shape = x.shape

        x = self.conv1(x)  # 1/2
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x) # 1/2
        f0 = x

        x = self.layer1(x)
        x = self.layer2(x)  # 1/2
        f1 = x

        x = self.layer3(x)  # 1/2
        f2 = x

        x = self.layer4(x)

        x = self.deconv1(x)
        x = FCNBase.align_shape(x, f2.shape)
        # print(x.shape)
        x = x + f2

        x = self.deconv2(x)
        x = FCNBase.align_shape(x, f1.shape)
        # print(x.shape)
        x = x + f1

        x = self.deconv3(x)
        x = FCNBase.align_shape(x, f0.shape)
        # print(x.shape)
        x = x + f0

        x = self.deconv4(x)
        x = FCNBase.align_shape(x, input_shape)
        # print(x.shape)

        x = self.fuse1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.fuse2(x)
        x = self.sigmoid(x)

        return x
