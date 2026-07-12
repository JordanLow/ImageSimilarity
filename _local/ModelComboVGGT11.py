import torch
import torch.nn as nn
import torchvision.models as models
from vggt.models.vggt import VGGT

'''
Documentation of the dimensions of this model:

Model1 is VGGT

The original input is fed directly into VGGT
The outputs is discarded and instead we grab the 11th layer of VGGT
We flatten the features into a 1d array and feed it into the next layer

The next layer is a fully connected layer accepting the previous dimension of features (as given by vggt.embed_dim) and outputting 2000 features
This layer uses ReLu as the activation function.

The last layer is another FC layer from 2000 to 1000 features. The output of this layer is the final output.

'''

class EarlyExitException(Exception):
    pass

class ModelComboVGGT11(nn.Module):
    def __init__(self):
        super(ModelComboVGGT11, self).__init__()
        self.vggt = VGGT.from_pretrained('facebook/VGGT-1B')
        self.extracted_features = {}
        for param in self.vggt.parameters():
            param.requires_grad = False
        
        def early_exit_hook(module, input, output):
            # Output might be a tuple, safely grab the tensor
            self.extracted_features['layer_11'] = output[0] if isinstance(output, tuple) else output
            
            # Instantly abort the rest of the VGGT forward pass!
            raise EarlyExitException()

        self.vggt.aggregator.frame_blocks[10].register_forward_hook(early_exit_hook)

        vggt_dim = 1024
        num_features = 1000

        self.fc1 = nn.Linear(vggt_dim, num_features*2)
        self.fc2 = nn.Linear(num_features*2, num_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        self.extracted_features.clear()

        x = x.unsqueeze(1)

        try:
            _ = self.vggt(x)
        except EarlyExitException:
            # The exception was successfully caught, meaning the model 
            # stopped at layer 11 just like we wanted.
            pass 

        # Retrieve the safely extracted features
        features = self.extracted_features['layer_11']

        features_1d = features.mean(dim=1)

        x = self.relu(self.fc1(features_1d))
        x = self.fc2(x)
        return x