import torch
import torchvision
from typing import List, Callable
from robomimic.models.obs_core import VisualCore
from torchvision import models
from torch import nn
from torchvision import transforms
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor
from transformers import ViTImageProcessor, ViTModel
import mail_cfg

class VIT(nn.Module):
    def __init__(self, input_shape: List[int], output_size: int):
        super().__init__()
        self.preprocess = nn.Sequential(
            transforms.Resize(224)
        )
        self.vit = ViTModel.from_pretrained("facebook/dino-vits8", device_map="cuda")
        self.vit.requires_grad_(False)

        hidden_dim=512

        self.fc1 = nn.Linear(in_features=384, out_features=hidden_dim)
        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=output_size)

    def forward(self, x):
        x = self.preprocess(x) 
        dino_embedding = self.vit(pixel_values=x)
        dino_last_hidden_states = dino_embedding.last_hidden_state[:,0]
        hidden = nn.functional.relu(self.fc1(dino_last_hidden_states))
        logits = self.fc2(hidden)
        
        return logits

def get_vision_model(input_shape: List[int], output_size: int):
    if mail_cfg.USE_VIT:
        return VIT(input_shape, output_size)
    return get_resnet(input_shape, output_size)


def get_resnet(input_shape: List[int], output_size: int):
    """Get ResNet model from torchvision.models
    Args:
        input_shape: Shape of input image (C, H, W).
        output_size: Size of output feature vector.
    """

    resnet = VisualCore(
        input_shape=input_shape,
        backbone_class="ResNet18Conv",
        backbone_kwargs=dict(
            input_coord_conv=False,
            pretrained=False,
        ),
        pool_class="SpatialSoftmax",
        pool_kwargs=dict(
            num_kp=32,
            learnable_temperature=False,
            temperature=1.0,
            noise_std=0.0,
            output_variance=False,
        ),
        flatten=True,
        feature_dimension=output_size,
    )

    return resnet


# def _get_old_resnet(name, weights=None, **kwargs):
#     """
#     name: resnet18, resnet34, resnet50
#     weights: "IMAGENET1K_V1", "r3m"
#     """
#     # load r3m weights
#     if (weights == "r3m") or (weights == "R3M"):
#         return get_r3m(name=name, **kwargs)
#
#     func = getattr(torchvision.models, name)
#     resnet = func(weights=weights, **kwargs)
#
#     num_fc_in = resnet.fc.in_features
#
#     resnet.fc = torch.nn.Linear(num_fc_in, 64)
#     # resnet.fc = torch.nn.Identity()
#
#     return resnet
#
# def get_r3m(name, **kwargs):
#     """
#     name: resnet18, resnet34, resnet50
#     """
#     import r3m
#     r3m.device = 'cpu'
#     model = r3m.load_r3m(name)
#     r3m_model = model.module
#     resnet_model = r3m_model.convnet
#     resnet_model = resnet_model.to('cpu')
#     return resnet_model