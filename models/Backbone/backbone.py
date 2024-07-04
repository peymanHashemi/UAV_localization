import torch.nn as nn
import timm
from .RKNet import RKNet
from .cvt import get_cvt_models
import torch

def make_backbone(opt):
    backbone_model = Backbone(opt)
    return backbone_model


class Backbone(nn.Module):
    def __init__(self, opt):
        super().__init__()
        if opt.segmentaion==True: 
            self.opt = opt
            self.seg_img_size = (opt.h,opt.w)
            self.backbone,self.output_channel = self.init_backbone(opt.seg_backbone)
        else:
            self.opt = opt  
            self.img_size = (opt.h,opt.w)
            self.backbone,self.output_channel = self.init_backbone(opt.backbone)

    def init_backbone(self, backbone):
        #models
        if backbone=="resnet50":
            backbone_model = timm.create_model('resnet50', pretrained=True)
            output_channel = 2048
        elif backbone=="RKNet":
            backbone_model = RKNet()
            output_channel = 2048
        elif backbone=="senet":
            backbone_model = timm.create_model('legacy_seresnet50', pretrained=True)
            output_channel = 2048
        elif backbone=="Pvtv2b2":
            backbone_model = timm.create_model("pvt_v2_b2", pretrained=True)
            output_channel = 512
        elif backbone=="Convnext-T":
            backbone_model = timm.create_model("convnext_tiny", pretrained=True)
            output_channel = 768

        # 9 
        # vit-models
        elif backbone=="ViTS-224":
            backbone_model = timm.create_model("vit_small_patch16_224", pretrained=True, img_size=self.img_size)
            output_channel = 384
        elif backbone=="ViTS-384":
            backbone_model = timm.create_model("vit_small_patch16_384", pretrained=True)
            output_channel = 384
        elif backbone=="vit_small_patch16_224.augreg_in21k_ft_in1k":
            backbone_model = timm.create_model("vit_small_patch16_224.augreg_in21k_ft_in1k", pretrained=True)
            output_channel = 384
        elif backbone=="vit_small_r26_s32_224.augreg_in21k_ft_in1k":
            backbone_model = timm.create_model("vit_small_r26_s32_224.augreg_in21k_ft_in1k", pretrained=True)
            output_channel = 384
        elif backbone=="ViTB-224":
            backbone_model = timm.create_model("vit_base_patch16_224", pretrained=True)
            output_channel = 768
            #done

        # 8 
        #deit-models
        elif backbone=="DeitS-224":
            backbone_model = timm.create_model("deit_small_distilled_patch16_224", pretrained=True)
            output_channel = 384
        elif backbone=="DeitB-224":
            backbone_model = timm.create_model("deit_base_distilled_patch16_224", pretrained=True)
            output_channel = 384

        #swin-models
        elif backbone=="SwinB-224":
            backbone_model = timm.create_model("swin_base_patch4_window7_224", pretrained=True)
            output_channel = 768
        elif backbone=="Swinv2S-256":
            backbone_model = timm.create_model("swinv2_small_window8_256", pretrained=True)
            output_channel = 768
        elif backbone=="Swinv2T-256":
            backbone_model = timm.create_model("swinv2_tiny_window16_256", pretrained=True)
            output_channel = 768

        #efficientnet-models
        elif backbone=="EfficientNet-B2":
            backbone_model = timm.create_model("efficientnet_b2", pretrained=True)
            output_channel = 1408
        elif backbone=="EfficientNet-B3":
            backbone_model = timm.create_model("efficientnet_b3", pretrained=True)
            output_channel = 1536
        elif backbone=="EfficientNet-B5":
            backbone_model = timm.create_model("tf_efficientnet_b5", pretrained=True)
            output_channel = 2048
        elif backbone=="EfficientNet-B6":
            backbone_model = timm.create_model("tf_efficientnet_b6", pretrained=True)
            output_channel = 2304
        
        # 1
        #mvit-models
        elif backbone=="mvitv2_base.fb_in1k":
            backbone_model = timm.create_model("mvitv2_base.fb_in1k", pretrained=True)
            output_channel = 768
        elif backbone=="mvitv2_base":
            backbone_model = timm.create_model("mvitv2_base", pretrained=True)
            output_channel = 768
        elif backbone=="mvitv2_base_cls.fb_inw21k":
            backbone_model = timm.create_model("mvitv2_base_cls.fb_inw21k", pretrained=True)
            output_channel = 768
        elif backbone=="mvitv2_small_cls":
            backbone_model = timm.create_model("mvitv2_small_cls")
            output_channel = 768
        elif backbone=="mvitv2_small.fb_in1k":
            backbone_model = timm.create_model("mvitv2_small.fb_in1k", pretrained=True)
            output_channel = 768
        elif backbone=="mvitv2_tiny.fb_in1k":
            backbone_model = timm.create_model("mvitv2_tiny.fb_in1k", pretrained=True)
            output_channel = 768
            #done

            
        # 7
        #repvit-models
        elif backbone=="repvit_m2_3.dist_450e_in1k":
            backbone_model = timm.create_model("repvit_m2_3.dist_450e_in1k", pretrained=True)
            output_channel = 640
            #done
        
        # 10
        #eva-models
        elif backbone=="eva02_small_patch14_224.mim_in22k":
            backbone_model = timm.create_model("eva02_small_patch14_224.mim_in22k", pretrained=True)
            output_channel = 384
        elif backbone=="eva02_base_patch14_224.mim_in22k":
            backbone_model = timm.create_model("eva02_base_patch14_224.mim_in22k", pretrained=True)
            output_channel = 768
            
        elif backbone=="fastvit_sa36":
            backbone_model = timm.create_model("fastvit_sa36", pretrained=True)
            output_channel = 1024

        # 11
        #conv-models
        elif backbone=="convformer_m36.sail_in22k_ft_in1k":
            backbone_model = timm.create_model("convformer_m36.sail_in22k_ft_in1k", pretrained=True)
            output_channel = 576
        elif backbone=="convformer_s36.sail_in22k_ft_in1k":
            backbone_model = timm.create_model("convformer_s36.sail_in22k_ft_in1k", pretrained=True)
            output_channel = 512
            #donnnne

        # 2
        #coformer-models
        elif backbone=="caformer_s36.sail_in22k":
            backbone_model = timm.create_model("caformer_s36.sail_in22k", pretrained=True)
            output_channel = 512
        elif backbone=="caformer_s36.sail_in22k_ft_in1k":
            backbone_model = timm.create_model("caformer_s36.sail_in22k_ft_in1k", pretrained=True)
            output_channel = 512  
        elif backbone=="caformer_m36.sail_in22k_ft_in1k":
            backbone_model = timm.create_model("caformer_m36.sail_in22k_ft_in1k", pretrained=True)
            output_channel = 576
        elif backbone=="caformer_b36.sail_in22k_ft_in1k":
            backbone_model = timm.create_model("caformer_b36.sail_in22k_ft_in1k", pretrained=True)
            output_channel = 768
            #done
        
        # 3
        #tinyvit-models
        elif backbone=="tiny_vit_21m_224":
            backbone_model = timm.create_model("tiny_vit_21m_224", pretrained=True)
            output_channel = 576
        elif backbone=="tiny_vit_21m_224.dist_in22k_ft_in1k":
            backbone_model = timm.create_model("tiny_vit_21m_224.dist_in22k_ft_in1k", pretrained=True)
            output_channel = 576
        elif backbone=="tiny_vit_11m_224.dist_in22k_ft_in1k":
            backbone_model = timm.create_model("tiny_vit_11m_224.dist_in22k_ft_in1k", pretrained=True)
            output_channel = 448
            #done
        
        # 4 
        #maxvit-models
        elif backbone=="maxvit_rmlp_small_rw_224.sw_in1k":
            backbone_model = timm.create_model("maxvit_rmlp_small_rw_224.sw_in1k", pretrained=True)
            output_channel = 768
        elif backbone=="maxvit_small_tf_224.in1k":
            backbone_model = timm.create_model("maxvit_small_tf_224.in1k", pretrained=True)
            output_channel = 768
        elif backbone=="maxvit_rmlp_base_rw_224.sw_in12k_ft_in1k":
            backbone_model = timm.create_model("maxvit_rmlp_base_rw_224.sw_in12k_ft_in1k", pretrained=True)
            output_channel = 768
            #done

            
        
        # 12
        #xcit-models
        elif backbone=="xcit_small_12_p8_224.fb_dist_in1k":
            backbone_model = timm.create_model("xcit_small_12_p8_224.fb_dist_in1k", pretrained=True)
            output_channel = 384
        
        # 5
        #coatnet-models
        elif backbone=="coatnet_rmlp_1_rw2_224.sw_in12k_ft_in1k":
            backbone_model = timm.create_model("coatnet_rmlp_1_rw2_224.sw_in12k_ft_in1k", pretrained=True)
            output_channel = 768
        elif backbone=="coatnet_2_rw_224.sw_in12k_ft_in1k":
            backbone_model = timm.create_model("coatnet_2_rw_224.sw_in12k_ft_in1k", pretrained=True)
            output_channel = 1024
            #done

        # 6
        elif backbone=="volo_d2_224.sail_in1k":
            backbone_model = timm.create_model("volo_d2_224.sail_in1k", pretrained=True)
            output_channel = 512
            #dooooooooone

        elif backbone=="repvit_m2_3.dist_450e_in1k":
            backbone_model = timm.create_model("repvit_m2_3.dist_450e_in1k", pretrained=True)
            output_channel = 640
            #done

        
        elif backbone=="vgg16":
            backbone_model = timm.create_model("vgg16", pretrained=True)
            output_channel = 512
        elif backbone=="cvt13":
            backbone_model, channels = get_cvt_models(model_size="cvt13")
            output_channel = channels[-1]
            checkpoint_weight = "/home/dmmm/VscodeProject/FPI/pretrain_model/CvT-13-384x384-IN-22k.pth"
            backbone_model = self.load_checkpoints(checkpoint_weight, backbone_model)
        else:
            raise NameError("{} not in the backbone list!!!".format(backbone))
        return backbone_model,output_channel
    
    def load_checkpoints(self, checkpoint_path, model):
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        filter_ckpt = {k: v for k, v in ckpt.items() if "pos_embed" not in k}
        missing_keys, unexpected_keys = model.load_state_dict(filter_ckpt, strict=False)
        print("Load pretrained backbone checkpoint from:", checkpoint_path)
        print("missing keys:", missing_keys)
        print("unexpected keys:", unexpected_keys)
        return model

    def forward(self, image):
        features = self.backbone.forward_features(image)
        shape = features.shape
        if len(shape) >= 4:
            features = features.reshape(shape[0], shape[2]*shape[3], shape[1])
        # import pdb; pdb.set_trace()
        return features