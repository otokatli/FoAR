import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms
import numpy as np

from policy.tokenizer import Sparse3DEncoder, ForceEncoder
from policy.transformer import Transformer
from policy.diffusion import DiffusionUNetPolicy, DiffusionUNetLowdimPolicy

from torchvision.models import resnet18


class MultimodalResNetBinaryClassifier(nn.Module):
    def __init__(self, num_obs_force=100, num_classes=1, pretrained=True):
        super(MultimodalResNetBinaryClassifier, self).__init__()
        self.resnet = resnet18(pretrained=pretrained)
        
        # Add layers to process force_torque data
        self.force_fc = nn.Sequential(
            nn.Linear(6 * num_obs_force, 128),
            nn.ReLU(),
            nn.Linear(128, 256)
        )
        
        # Modify the final fully connected layer
        self.fc = nn.Linear(self.resnet.fc.in_features + 256, num_classes)
        self.resnet.fc = nn.Identity()  # Remove the original fc layer

    def forward(self, x, force_torque):
        # Process image through ResNet
        x = self.resnet(x)
        
        # Process force_torque data
        force_features = self.force_fc(force_torque)
        
        # Concatenate features
        combined_features = torch.cat((x, force_features), dim=1)
        
        # Final classification
        output = torch.sigmoid(self.fc(combined_features))
        return output

class FoAR(nn.Module):
    def __init__(
            self,
            num_action = 20,
            input_dim = 6,
            obs_feature_dim = 512,
            action_dim = 10,
            hidden_dim = 512,
            nheads = 8,
            num_encoder_layers = 4,
            num_decoder_layers = 1,
            dim_feedforward = 2048,
            dropout = 0.1,
            num_obs_force = 200
            ):
        super().__init__()
        num_obs = 1
        self.sparse_encoder = Sparse3DEncoder(input_dim, obs_feature_dim)
        self.force_encoder = ForceEncoder(num_obs_force, input_dim, obs_feature_dim)
        self.sparse_transformer = Transformer(hidden_dim, nheads, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout)
        self.force_transformer = Transformer(hidden_dim, nheads, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout)
        self.action_decoder = DiffusionUNetPolicy(action_dim, num_action, num_obs, obs_feature_dim * 2) # concat readout from two transformer layers
        self.sparse_readout_embed = nn.Embedding(1, hidden_dim)
        self.force_readout_embed = nn.Embedding(1, hidden_dim)
        self.flat_embed = nn.Embedding(1, hidden_dim)
        self.classifier = MultimodalResNetBinaryClassifier(num_obs_force=num_obs_force)
        self.classifier_criterion = nn.BCELoss()

    def forward(self, force_torque, color_list, cloud, actions=None, is_cut=None, batch_size=24):
        src, pos, src_padding_mask = self.sparse_encoder(cloud, batch_size=batch_size)
        
        # Process sparse data
        sparse_readout = self.sparse_transformer(src, src_padding_mask, self.sparse_readout_embed.weight, pos)[-1]
        sparse_readout = sparse_readout[:, 0]

        force_torque_feature, force_torque_pos, force_torque_padding_mask = self.force_encoder(force_torque, batch_size=batch_size)

        force_readout = self.force_transformer(force_torque_feature, force_torque_padding_mask, self.force_readout_embed.weight, force_torque_pos)[-1]
        force_readout = force_readout[:, 0]

        force_torque = force_torque.view(batch_size, -1)
        prop = self.classifier(color_list, force_torque)

        weight = prop.expand(-1, force_readout.size(1))

        # Replace relevant indices with transformer output
        force_readout = weight * force_readout + (1 - weight) * self.flat_embed.weight

        combined_readout = torch.cat([sparse_readout, force_readout], dim=1)

        if actions is not None:
            loss_action = self.action_decoder.compute_loss(combined_readout, actions)
            prop = prop.squeeze()
            loss_cls = self.classifier_criterion(prop, is_cut)
            return loss_action, loss_cls
        else:
            with torch.no_grad():
                action_pred = self.action_decoder.predict_action(combined_readout)
            return prop.squeeze(), action_pred
        