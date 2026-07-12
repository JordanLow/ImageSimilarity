import torch
import torch.nn as nn
from vggt.models.vggt import VGGT

class ModelComboVGGTCombinedNoGrad(nn.Module):
    def __init__(self):
        super(ModelComboVGGTCombinedNoGrad, self).__init__()
        self.vggt = VGGT.from_pretrained(
            'facebook/VGGT-1B'
        )
        
        # 1. Freeze weights
        for param in self.vggt.parameters():
            param.requires_grad = False
            
        # 2. Set to evaluation mode to disable dropout in the backbone
        self.vggt.eval() 
        
        self.target_layers = [4, 11, 17, 23] 
        vggt_dim = 2048
        combined_dim = vggt_dim * len(self.target_layers)
        num_features = 1000

        self.fc1 = nn.Linear(combined_dim, num_features*2)
        self.fc2 = nn.Linear(num_features*2, num_features)
        self.relu = nn.ReLU()
        
        # Determine best dtype for Autocast
        self.dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    def forward(self, x):

        x = x.unsqueeze(1)

        # 3. Use mixed precision for the backbone pass, but allow gradient tracking for trainable layers
        with torch.amp.autocast('cuda', dtype=self.dtype):
            raws, psidx = self.vggt.aggregator(x)
            
            extracted_features = []

            for layer_idx in self.target_layers:
                tokens = raws[layer_idx]
                
                # Slice spatial tokens
                spatial_tokens = tokens[:, :, psidx:]
                
                # Apply Global Average Pooling across the token dimension
                pooled_features = spatial_tokens.mean(dim=2)
                extracted_features.append(pooled_features)

            # Concatenate along the feature dimension (-1) safely
            features = torch.cat(extracted_features, dim=-1)
            features = features.squeeze(1)
        
        # Ensure features are in float32 for stable linear layer training,
        # but gradient tracking is implicitly enabled because no_grad() is not used for this part.
        features = features.to(torch.float32) 

        # Now feed these features to the trainable layers
        x = self.relu(self.fc1(features))
        x = self.fc2(x)
        return x
