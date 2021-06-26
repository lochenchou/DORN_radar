import torch
import torch.nn as nn
import torch.nn.functional as F

class FullImageEncoder(nn.Module):
    def __init__(self, h, w, kernel_size):
        super(FullImageEncoder, self).__init__()
#         self.global_pooling = nn.AvgPool2d(kernel_size, stride=kernel_size, padding=kernel_size // 2)  # KITTI 16 16
        self.global_pooling = nn.AvgPool2d(kernel_size, stride=kernel_size)  # KITTI 16 16
        self.dropout = nn.Dropout2d(p=0.5)

        self.h = h // kernel_size 
        self.w = w // kernel_size
   
        self.global_fc = nn.Linear(2048 * self.h * self.w, 512)  # kitti 4x5
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(512, 512, 1)  # 1x1 conv
        self.relu1 = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.global_pooling(x)
        x = self.dropout(x)
        x = x.view(-1, 2048 * self.h * self.w)  # kitti 4x5
        x = self.relu(self.global_fc(x))
        x = x.view(-1, 512, 1, 1)
        x = self.conv1(x)
        x = self.relu1(x)
        return x


class SceneUnderstandingModule(nn.Module):
    def __init__(self, ord_num, size, kernel_size, pyramid=[6, 12, 18]):
        # pyramid kitti [6, 12, 18] nyu [4, 8, 12]
        super(SceneUnderstandingModule, self).__init__()
        assert len(size) == 2
        assert len(pyramid) == 3
        self.size = size
        h, w = self.size
        self.encoder = FullImageEncoder(h // 8, w // 8, kernel_size)
        self.aspp1 = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=1, padding=0, dilation=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=1, padding=0, dilation=1),
            nn.ReLU(inplace=True)
        )
        self.aspp2 = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, padding=pyramid[0], dilation=pyramid[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=1, padding=0, dilation=1),
            nn.ReLU(inplace=True)
        )
        self.aspp3 = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, padding=pyramid[1], dilation=pyramid[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=1, padding=0, dilation=1),
            nn.ReLU(inplace=True)
        )
        self.aspp4 = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, padding=pyramid[2], dilation=pyramid[2]),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=1, padding=0, dilation=1),
            nn.ReLU(inplace=True)
        )
        self.concat_process = nn.Sequential(
            nn.Dropout2d(p=0.5),
            nn.Conv2d(512*5, 2048, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5),
            nn.Conv2d(2048, ord_num * 2, kernel_size=1)
        )

    def forward(self, x):
        N, C, H, W = x.shape
#         print('Scene understanding module')
#         print('input shape: {}'.format(x.shape))
        x1 = self.encoder(x)
#         print('after full image encoder, {}'.format(x1.shape))
        x1 = F.interpolate(x1, size=(H, W), mode="bilinear", align_corners=True)
#         print('after upsampling, {}'.format(x1.shape))
        x2 = self.aspp1(x)
#         print('after ASPP-0, {}'.format(x2.shape))
        x3 = self.aspp2(x)
#         print('after ASPP-6, {}'.format(x3.shape))
        x4 = self.aspp3(x)
#         print('after ASPP-12, {}'.format(x4.shape))
        x5 = self.aspp4(x)
#         print('after ASPP-18, {}'.format(x5.shape))

        x6 = torch.cat((x1, x2, x3, x4, x5), dim=1)
#         print('after concat, {}'.format(x6.shape))
        out = self.concat_process(x6)
#         print('after last conv, {}'.format(out.shape))
        out = F.interpolate(out, size=self.size, mode="bilinear", align_corners=True)
#         print('after upsampling, {}'.format(out.shape))
        return out