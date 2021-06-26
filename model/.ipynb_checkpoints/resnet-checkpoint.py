import torch
import torch.nn as nn
affine_par = True


def weights_init(m):
    # Initialize filters with Gaussian random weights
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv7x7(in_planes, out_planes, stride=2):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=7, stride=stride,
                     padding=3, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, fist_dilation=1, multi_grid=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation * multi_grid, dilation=dilation * multi_grid, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=False)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu_inplace(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, in_channel):
        self.inplanes = 128
        super(ResNet, self).__init__()
        self.conv1 = conv3x3(in_channel, 64, stride=2)
        self.bn1 = nn.BatchNorm2d(64, momentum=0.95)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(64, 64)
        self.bn2 = nn.BatchNorm2d(64, momentum=0.95)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv3 = conv3x3(64, 128)
        self.bn3 = nn.BatchNorm2d(128, momentum=0.95)
        self.relu3 = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

#         self.relu = nn.ReLU(inplace=False)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)  # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4, multi_grid=(1, 1, 1))

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, multi_grid=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, affine=affine_par))

        layers = []
        generate_multi_grid = lambda index, grids: grids[index % len(grids)] if isinstance(grids, tuple) else 1
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample,
                            multi_grid=generate_multi_grid(0, multi_grid)))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, dilation=dilation, multi_grid=generate_multi_grid(i, multi_grid)))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def freeze(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                

class ResNet101(nn.Module):

    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = ResNet(Bottleneck, [3, 4, 23, 3], in_channel=3)

        if pretrained:
            saved_state_dict = torch.load('/datasets/KITTI/depth_prediction/pretrained/resnet101-imagenet.pth',
                                          map_location="cpu")
            new_params = self.backbone.state_dict().copy()
            for i in saved_state_dict:
                i_parts = i.split('.')
                if not i_parts[0] == 'fc':
                    new_params['.'.join(i_parts[0:])] = saved_state_dict[i]

            self.backbone.load_state_dict(new_params)

    def forward(self, input):
        return self.backbone(input)
    
    

class ResNet101_early(nn.Module):

    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = ResNet(Bottleneck, [3, 4, 23, 3], in_channel=4)

        if pretrained:
            saved_state_dict = torch.load('/datasets/KITTI/depth_prediction/pretrained/resnet101-imagenet.pth',
                                          map_location="cpu")
            new_params = self.backbone.state_dict().copy()
            for i in saved_state_dict:
                i_parts = i.split('.')
#                 print(i_parts)
                if not i_parts[0] == 'fc' and not i_parts[0] == 'conv1' and not i_parts[0] == 'bn1':
                    new_params['.'.join(i_parts[0:])] = saved_state_dict[i]

            self.backbone.load_state_dict(new_params)

    def forward(self, input):
        return self.backbone(input)
    
class ResNet101_radar(nn.Module):

    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = ResNet(Bottleneck, [3, 4, 23, 3], in_channel=1)

        if pretrained:
            saved_state_dict = torch.load('/datasets/KITTI/depth_prediction/pretrained/resnet101-imagenet.pth',
                                          map_location="cpu")
            new_params = self.backbone.state_dict().copy()
            for i in saved_state_dict:
                i_parts = i.split('.')
#                 print(i_parts)
                if not i_parts[0] == 'fc' and not i_parts[0] == 'conv1' and not i_parts[0] == 'bn1':
                    new_params['.'.join(i_parts[0:])] = saved_state_dict[i]

            self.backbone.load_state_dict(new_params)

    def forward(self, input):
        return self.backbone(input)
    
    
class ResNet26_radar(nn.Module):

    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = ResNet(Bottleneck, [2,2,2,2], in_channel=1)

    def forward(self, input):
        return self.backbone(input)