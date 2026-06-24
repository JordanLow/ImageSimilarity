import torch
import torch.nn as nn
import torchvision.models as models

'''
Documentation of the dimensions of this model:

Model1 is DINOv3 with DINO_ViT_b_16 weights, outputs 768 features
Model2 is EfficientNetv2, outputs 1000 features
Model3 is Swin, outputs 1000 features

The original input is cloned and fed into all 3
The outputs are concatenated for a layer with 2768 features

The next layer is a fully connected layer accepting 2768 features and outputting 2000 features
This layer uses ReLu as the activation function.

The last layer is another FC layer from 2000 to 1000 features. The output of this layer is the final output.

'''

class ModelComboDINO(nn.Module):
    def __init__(self):
        super(ModelComboDINO, self).__init__()
        self.model1 = torch.hub.load('facebookresearch/dinov3', 'dinov3_vitl16', skip_validation=True, weights='./dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth')        
        #self.model1 = models.vit_b_16(weights='IMAGENET1K_SWAG_LINEAR_V1')
        self.model2 = models.efficientnet_v2_m(weights='DEFAULT')
        # self.model1 = models.vgg16(weights='DEFAULT')
        # self.model2 = models.resnet50(weights='IMAGENET1K_V2')
        # models.inception_v3(weights='IMAGENET1K_V1')
        self.model3 = models.swin_t(weights='DEFAULT')
        for param in self.model1.parameters():
            param.requires_grad = False
        for param in self.model2.parameters():
            param.requires_grad = False
        for param in self.model3.parameters():
            param.requires_grad = False
        num_features = 1000
        dino_features = 1024
        self.fc1 = nn.Linear(num_features*2+dino_features, num_features*2)
        self.fc2 = nn.Linear(num_features*2, num_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.model1(x.clone())
        x2 = self.model2(x.clone())
        x3 = self.model3(x.clone())

        if self.training:
            x = torch.cat((x1, x2, x3), dim=1)  # .logits
        else:
            x = torch.cat((x1, x2, x3), dim=1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x