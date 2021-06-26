import torch
from model.resnet import ResNet101
from model.modules import SceneUnderstandingModule
from model.ordinal_regression_layer import OrdinalRegressionLayer


class DORN(torch.nn.Module):

    def __init__(self, ord_num=80, input_size=(350, 800), kernel_size=16, pyramid=[6, 12, 18], pretrained=True):
        super().__init__()
        assert len(input_size) == 2
        assert isinstance(kernel_size, int)
        self.ord_num = ord_num
        self.resnet101 = ResNet101(pretrained=pretrained)
        self.scene_understanding_module = SceneUnderstandingModule(ord_num, size=input_size,
                                                                 kernel_size=kernel_size,
                                                                 pyramid=pyramid)
        self.ord_regression_layer = OrdinalRegressionLayer()


    def forward(self, image, target=None):
        """
        :input: image: torch.Tensor, (N,3,H,W)
                target: target depth, torch.Tensor, (N,H,W)
                
        :return:prob: probability of each label, torch.Tensor, (N,K,H,W), K is the ordinal number 
                label: predicted label, torch.Tensor, (N,H,W)
        """
        N, C, H, W = image.shape
#         print('input image.shape {}'.format(image.shape))
        feat = self.resnet101(image)
#         print('after resnet 101 shape {}'.format(feat.shape))
        feat = self.scene_understanding_module(feat) # (N, 2K, H, W) > (N, 160, 385, 513)
#         print('after scene_understanding_module shape {}'.format(feat.shape))
        prob = self.ord_regression_layer(feat) # (N, K, H, W)
        
        # calculate label
        label = torch.sum((prob >= 0.5), dim=1).view(-1, 1, H, W) # (N, 1, H, W)
    
        return prob, label
    

