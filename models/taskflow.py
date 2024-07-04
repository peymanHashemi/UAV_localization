import torch.nn as nn
from .Backbone.backbone import make_backbone
from .Head.head import make_head
import os
import torch


class Model(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.backbone = make_backbone(opt)
        self.seg_backbone = make_backbone(opt)

        opt.in_planes = self.backbone.output_channel
        opt.seg_in_planes = self.seg_backbone.output_channel
        
        self.concat_dim = self.backbone.output_channel + self.seg_backbone.output_channel
        # Define the reduction layer to reduce the concatenated features back to the original dimension
        self.reduction_layer = nn.Linear(self.concat_dim, self.backbone.output_channel)

        # Define the reduction layer using a 1x1 convolutional layer
        # self.reduction_layer = nn.Conv2d(self.concat_dim, self.backbone.output_channel, kernel_size=1)

        self.head = make_head(opt)
        self.opt = opt

    def forward(self, drone_image, seg_drone_image, satellite_image, seg_satellite_image):
        if self.opt.segmentaion:
            if drone_image is None:
                drone_res = None
            else:
                drone_features = self.backbone(drone_image)
                seg_drone_features = self.seg_backbone(seg_drone_image)
                # drone_res = self.head(torch.max(drone_features, seg_drone_features))
                # drone_res = self.head((drone_features + seg_drone_features)/2)
                concat_drone_features = torch.cat((drone_features, seg_drone_features), dim=2)
                reduced_drone_features = self.reduction_layer(concat_drone_features)
                drone_res = self.head(reduced_drone_features)

            if satellite_image is None:
                satellite_res = None
            else:
                satellite_features = self.backbone(satellite_image)
                seg_satellite_features = self.seg_backbone(seg_satellite_image)
                # satellite_res = self.head((satellite_features + seg_satellite_features)/2)
                # satellite_res = self.head(torch.max(satellite_features, seg_satellite_features))


                concat_satellite_features = torch.cat((satellite_features, seg_satellite_features), dim=2)
                reduced_satellite_features = self.reduction_layer(concat_satellite_features)
                satellite_res = self.head(reduced_satellite_features)
        
            return drone_res,satellite_res
        



        else:
            if drone_image is None:
                drone_res = None
            else:
                drone_features = self.backbone(drone_image)
                drone_res = self.head(drone_features)
            if satellite_image is None:
                satellite_res = None
            else:
                satellite_features = self.backbone(satellite_image)
                satellite_res = self.head(satellite_features)
        
            return drone_res,satellite_res
    
    def load_params(self, load_from):
        pretran_model = torch.load(load_from)
        model2_dict = self.state_dict()
        state_dict = {k: v for k, v in pretran_model.items() if k in model2_dict.keys() and v.size() == model2_dict[k].size()}
        model2_dict.update(state_dict)
        self.load_state_dict(model2_dict)


def make_model(opt):
    model = Model(opt)
    if os.path.exists(opt.load_from):
        model.load_params(opt.load_from)
    return model
