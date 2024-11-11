import torch.nn as nn
import torchvision.models as models


class ResNetSimCLR(nn.Module):

    def __init__(self, base_model, pretrained, out_dim):
        super().__init__()

        self.backbone = self._get_basemodel(base_model, pretrained)
        d_feats = self.backbone.fc.in_features

        # skip the original fc layer
        self.backbone.fc = nn.Identity()

        # add mlp projection head
        self.head = nn.Sequential(
            nn.Linear(d_feats, d_feats),
            nn.ReLU(),
            nn.Linear(d_feats, out_dim),
        )

    def _get_basemodel(self, model_name, pretrained):
        weights = pretrained or None
        resnet_dict = {
            "resnet18": lambda: models.resnet18(weights=weights),
            "resnet50": lambda: models.resnet50(weights=weights),
        }

        if model_name not in resnet_dict:
            raise ValueError(
                f"Invalid '{model_name}' backbone. Use one of: {list(resnet_dict.keys())}"
            )

        return resnet_dict[model_name]()

    def forward(self, x):
        x = self.backbone(x)
        return self.head(x)
