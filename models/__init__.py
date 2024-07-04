
#################################################
        # self.max_channels = max(self.backbone_channels, self.seg_backbone_channels)
        # self.min_channels = min(self.backbone_channels, self.seg_backbone_channels)

        # # Define a convolutional layer to adjust the smaller feature map to match the larger one
        # self.adjust_layer = nn.Conv2d(self.min_channels, self.max_channels, kernel_size=1)

        # # Define the reduction layer using a 1x1 convolutional layer
        # self.reduction_layer = nn.Conv2d(2 * self.max_channels, self.backbone_channels, kernel_size=1)
    #     self.head = make_head(opt)
    #     self.opt = opt

    # def forward(self, drone_image, seg_drone_image, satellite_image, seg_satellite_image):
    #     if self.opt.segmentaion:
    #         if drone_image is None:
    #             drone_res = None
    #         else:
    #             drone_features = self.backbone(drone_image)
    #             seg_drone_features = self.seg_backbone(seg_drone_image)

    #             # Adjust the smaller feature map to match the larger one
    #             if self.backbone_channels < self.seg_backbone_channels:
    #                 drone_features = self.adjust_layer(drone_features)
    #             else:
    #                 seg_drone_features = self.adjust_layer(seg_drone_features)

    #             # Concatenate the features along the channel dimension
    #             concat_drone_features = torch.cat((drone_features, seg_drone_features), dim=1)
    #             reduced_drone_features = self.reduction_layer(concat_drone_features)
    #             drone_res = self.head(reduced_drone_features)

    #         if satellite_image is None:
    #             satellite_res = None
    #         else:
    #             satellite_features = self.backbone(satellite_image)
    #             seg_satellite_features = self.seg_backbone(seg_satellite_image)

    #             # Adjust the smaller feature map to match the larger one
    #             if self.backbone_channels < self.seg_backbone_channels:
    #                 satellite_features = self.adjust_layer(satellite_features)
    #             else:
    #                 seg_satellite_features = self.adjust_layer(seg_satellite_features)

    #             # Concatenate the features along the channel dimension
    #             concat_satellite_features = torch.cat((satellite_features, seg_satellite_features), dim=1)
    #             reduced_satellite_features = self.reduction_layer(concat_satellite_features)
    #             satellite_res = self.head(reduced_satellite_features)
        
    #         return drone_res, satellite_res
################################################################