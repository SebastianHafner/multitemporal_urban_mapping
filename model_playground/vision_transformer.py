"""
Override timm.models.vision_transformer.VisionTransformer to
output all output tokens (excluding class or distill tokens)
This works for the vision_transformer_hybrid models as well
"""
import torch
import timm.models.vision_transformer


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):

    def forward_features(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        """
        x = self.norm(x)
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]
        """
        if self.dist_token is None:
            return x[:, 1:]
        else:
            return x[:, 2:]

    def forward(self, x):
        """
        x = self.forward_features(x)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])  # x must be a tuple
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
        return x
        """
        x = self.forward_features(x)
        return x


# class VisionTransformerLogit():
#     def __init__(self, variant='vit_small_r26_s32_384', pretrained=True):
#         self.model = timm.create_model(variant, pretrained=pretrained)
#
#     def forward(self, x):
#         x = self.model.forward_features(x)
#         x = self.model.forward_head(x, pre_logits=True)
#         return x


timm.models.vision_transformer.VisionTransformer = VisionTransformer