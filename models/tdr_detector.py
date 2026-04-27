import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, resnet50
from .tdr_head import TDRHead

class SimpleFPN(nn.Module):
    """
    简化的特征金字塔网络 (FPN)
    用于提取多尺度特征
    """
    def __init__(self, in_channels_list, out_channels):
        """
        初始化
        
        Args:
            in_channels_list: 来自 Backbone 的各层输出通道数
            out_channels: FPN 输出通道数
        """
        super(SimpleFPN, self).__init__()
        self.out_channels = out_channels
        
        # 侧向连接卷积层
        self.lateral_convs = nn.ModuleList()
        for in_channels in in_channels_list:
            self.lateral_convs.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=1)
            )
        
        # 输出卷积层
        self.fpn_convs = nn.ModuleList()
        for _ in range(len(in_channels_list)):
            self.fpn_convs.append(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            )
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 来自 Backbone 的各层输出特征列表
            
        Returns:
            fpn_feats: FPN 输出的多尺度特征列表
        """
        # 计算侧向连接特征
        lateral_feats = []
        for i, conv in enumerate(self.lateral_convs):
            lateral_feats.append(conv(x[i]))
        
        # 自顶向下融合
        fpn_feats = [lateral_feats[-1]]
        for i in range(len(lateral_feats) - 2, -1, -1):
            # 上采样
            upsampled = F.interpolate(fpn_feats[-1], size=lateral_feats[i].shape[2:], mode='nearest')
            # 融合
            fused = lateral_feats[i] + upsampled
            fpn_feats.append(fused)
        
        # 反转顺序，使其从低层到高层
        fpn_feats = fpn_feats[::-1]
        
        # 应用输出卷积
        for i, conv in enumerate(self.fpn_convs):
            fpn_feats[i] = conv(fpn_feats[i])
        
        return fpn_feats

class TDRDetector(nn.Module):
    """
    TDR-QAF 系统的顶层检测器
    集成 Backbone、Neck 和 TDRHead
    """
    def __init__(self, num_classes=10, embed_dims=256 ,
                 num_decoder_layers=6, num_depth_dense=48, max_queries=400, debug=False):
        """
        初始化
        
        Args:
            num_classes (int): 类别数（例如 nuScenes 的 10 类）
            embed_dims (int): Transformer 的隐藏层维度
            num_decoder_layers (int): Decoder 的层数（如 6 层）
        """
        super(TDRDetector, self).__init__()
        self.num_classes = num_classes
        self.embed_dims = embed_dims
        self.num_decoder_layers = num_decoder_layers
        
        # 定义 Backbone (ResNet50)
        try:
            backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        except Exception:
            backbone = resnet50(weights=None)
        # 提取 Backbone 的主要层
        self.backbone = nn.ModuleDict({
            'conv1': backbone.conv1,
            'bn1': backbone.bn1,
            'relu': backbone.relu,
            'maxpool': backbone.maxpool,
            'layer1': backbone.layer1,  # output channels: 256
            'layer2': backbone.layer2,  # output channels: 512
            'layer3': backbone.layer3,  # output channels: 1024
            'layer4': backbone.layer4   # output channels: 2048
        })
        
        # 定义 Neck (简化的 FPN)
        in_channels_list = [256, 512, 1024, 2048]  # 对应 layer1-layer4 的输出通道
        out_channels = 256  # 输出通道数
        self.neck = SimpleFPN(in_channels_list, out_channels)
        
        # 实例化 TDRHead
        self.pts_bbox_head = TDRHead(
            num_classes=num_classes,
            in_channels=out_channels,
            embed_dims=embed_dims,
            num_decoder_layers=num_decoder_layers,  # 传递参数
            num_depth_dense=num_depth_dense,        # 传递参数
            max_queries=max_queries,                # 传递参数
            debug=debug
        )
    
    def extract_feat(self, imgs):
        """
        提取图像特征
        
        Args:
            imgs: 多视角图像 [B * N_cam, 3, H, W]
            
        Returns:
            feat: 提取的特征 [B * N_cam, C, H_feat, W_feat]
        """
        # Backbone 前向传播
        x = self.backbone.conv1(imgs)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        # 提取各层特征
        layer1_feat = self.backbone.layer1(x)
        layer2_feat = self.backbone.layer2(layer1_feat)
        layer3_feat = self.backbone.layer3(layer2_feat)
        layer4_feat = self.backbone.layer4(layer3_feat)
        
        # 经过 FPN
        fpn_feats = self.neck([layer1_feat, layer2_feat, layer3_feat, layer4_feat])
        
        # 返回全部 4 层特征列表
        return fpn_feats
    
    def forward_train(self, imgs, boxes_2d, cam_intrinsics, cam_extrinsics, temporal_depth_prior=None):
        """
        训练时的前向传播
        
        Args:
            imgs: 多视角图像 [B, N_cam, 3, H, W]
            boxes_2d: 多视角2D候选框 [B, N_cam, Num_boxes, 4] (xyxy格式)
            cam_intrinsics: 相机内参 [B, N_cam, 3, 3] 或 [B, N_cam, 4, 4]
            cam_extrinsics: 相机外参 [B, N_cam, 4, 4] (Camera to Ego)
            temporal_depth_prior: 时序深度先验 [B, N_cam, Num_boxes] 或 None
            
        Returns:
            cls_scores: 类别概率 [B, Num_Query, num_classes]
            bbox_preds: 3D 边界框预测 [B, Num_Query, 10]
        """
        B, N_cam, _, H, W = imgs.shape
        
        # 将多视角图像合并为 [B * N_cam, 3, H, W]
        imgs_reshaped = imgs.view(-1, 3, H, W)
        
        # 提取多尺度特征
        mlvl_feats = self.extract_feat(imgs_reshaped)  # 返回的是 list
        
        # 遍历每层特征，分别进行维度重塑
        mlvl_feats_reshaped = []
        for feat in mlvl_feats:
            C, H_feat, W_feat = feat.shape[1], feat.shape[2], feat.shape[3]
            feat_reshaped = feat.view(B, N_cam, C, H_feat, W_feat)
            mlvl_feats_reshaped.append(feat_reshaped)
        
        # 送入 TDRHead（现在传入的是特征列表）
        cls_scores, bbox_preds = self.pts_bbox_head(
            mlvl_feats_reshaped, boxes_2d, cam_intrinsics, cam_extrinsics, temporal_depth_prior
        )
        
        return cls_scores, bbox_preds
    
    def forward(self, imgs, boxes_2d, cam_intrinsics, cam_extrinsics, temporal_depth_prior=None):
        """
        前向传播
        
        Args:
            imgs: 多视角图像 [B, N_cam, 3, H, W]
            boxes_2d: 多视角2D候选框 [B, N_cam, Num_boxes, 4] (xyxy格式)
            cam_intrinsics: 相机内参 [B, N_cam, 3, 3] 或 [B, N_cam, 4, 4]
            cam_extrinsics: 相机外参 [B, N_cam, 4, 4] (Camera to Ego)
            temporal_depth_prior: 时序深度先验 [B, N_cam, Num_boxes] 或 None
            
        Returns:
            cls_scores: 类别概率 [B, Num_Query, num_classes]
            bbox_preds: 3D 边界框预测 [B, Num_Query, 10]
        """
        return self.forward_train(imgs, boxes_2d, cam_intrinsics, cam_extrinsics, temporal_depth_prior)
