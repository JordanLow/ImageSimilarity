import torch
import torch.nn as nn
from vggt.models.vggt import VGGT

class ModelComboVGGTCombined(nn.Module):
    def __init__(self):
        super(ModelComboVGGTCombined, self).__init__()
        self.vggt = VGGT.from_pretrained('facebook/VGGT-1B')        
        # 1. Freeze weights
        for param in self.vggt.parameters():
            param.requires_grad = False
            
        # 2. Set to evaluation mode to disable dropout in the backbone
        self.vggt.eval() 
        
        self.target_layers = [4, 11, 17, 23] 
        vggt_dim = 2048
        combined_dim = vggt_dim * len(self.target_layers)
        num_features = 1000
        process_features = 1000

        self.layer_processors = nn.ModuleList([ 
            nn.Linear(vggt_dim, process_features)
            for _ in self.target_layers
        ])

        self.fc1 = nn.Linear(process_features * len(self.target_layers), num_features*2)
        self.fc2 = nn.Linear(num_features*2, num_features)
        self.relu = nn.ReLU()
        
        # Determine best dtype for Autocast
        self.dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    def forward(self, x):

        x = x.unsqueeze(1)

        # 3. Disable gradient tracking and use mixed precision for the backbone pass
        with torch.no_grad():
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

        feature_list = []
        for i, features in enumerate(extracted_features):
            features = features.mean(dim=1)
            features_for_head = features.clone().detach()
            features_for_head.requires_grad_(True)
            feature_list.append(self.relu(self.layer_processors[i](features_for_head)))

        fused_features = torch.cat(feature_list, dim=-1).to(torch.float32)

        # Now feed these re-attached features to the trainable layers
        x = self.relu(self.fc1(fused_features))
        x = self.fc2(x)
        return x
