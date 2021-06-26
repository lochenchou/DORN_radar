import torch
import torch.nn as nn
import torch.nn.functional as F
from model.resnet import ResNet101, ResNet101_radar, ResNet26_radar
from model.ordinal_regression_layer import OrdinalRegressionLayer

affine_par = True


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


    def forward(self, x):
        # x for RGB image feature, y for radar feature
        N, C, H, W = x.shape
        x1 = self.encoder(x)
        x1 = F.interpolate(x1, size=(H, W), mode="bilinear", align_corners=True)
        x2 = self.aspp1(x)
        x3 = self.aspp2(x)
        x4 = self.aspp3(x)
        x5 = self.aspp4(x)
        out = torch.cat((x1, x2, x3, x4, x5), dim=1)
        
        return out


class RaDORN(torch.nn.Module):

    def __init__(self, ord_num=80, input_size=(350, 800), kernel_size=16, pyramid=[6, 12, 18], pretrained=True):
        super().__init__()
        assert len(input_size) == 2
        assert isinstance(kernel_size, int)
        self.size = input_size
        self.ord_num = ord_num
        self.resnet101 = ResNet101(pretrained=pretrained)
        self.resnet26_radar = ResNet26_radar(pretrained=pretrained)
        self.scene_understanding_module = SceneUnderstandingModule(ord_num, size=input_size,
                                                                 kernel_size=kernel_size,
                                                                 pyramid=pyramid)
        
        self.conv_radar = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=1, padding=0, dilation=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=1, padding=0, dilation=1),
            nn.ReLU(inplace=True)
        )
        self.concat_process = nn.Sequential(
            nn.Dropout2d(p=0.5),
            nn.Conv2d(512*6, 2048, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5),
            nn.Conv2d(2048, ord_num * 2, kernel_size=1)
        )
        
        self.ord_regression_layer = OrdinalRegressionLayer()
        
        
    def forward(self, image, radar):
        """
        :input: image: torch.Tensor, (N,3,H,W)
                target: target depth, torch.Tensor, (N,H,W)
                
        :return:prob: probability of each label, torch.Tensor, (N,K,H,W), K is the ordinal number 
                label: predicted label, torch.Tensor, (N,H,W)
        """
        N, C, H, W = image.shape
        img_feat = self.resnet101(image)
        radar_feat = self.resnet26_radar(radar)
        img_feat = self.scene_understanding_module(img_feat) # (N, 2K, H, W) > (N, 160, 385, 513)
        radar_feat = self.conv_radar(radar_feat)
        feat = torch.cat((img_feat, radar_feat), dim=1)
    
        feat = self.concat_process(feat)
        feat = F.interpolate(feat, size=self.size, mode="bilinear", align_corners=True)
        prob = self.ord_regression_layer(feat) # (N, K, H, W)
        
        # calculate label
        label = torch.sum((prob >= 0.5), dim=1).view(-1, 1, H, W) # (N, 1, H, W)
    
        return prob, label
    


    