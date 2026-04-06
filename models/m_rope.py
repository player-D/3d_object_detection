import torch
import torch.nn as nn

class MRoPE(nn.Module):
    def __init__(self, embed_dims):
        """
        多模态旋转位置编码 (M-RoPE)
        将 TDR-Lifting 生成的 3D Reference Points 转换为复数域的旋转正余弦嵌入
        
        Args:
            embed_dims (int): 隐藏层特征维度（如 256）
        """
        super(MRoPE, self).__init__()
        self.embed_dims = embed_dims
        
        # 计算各轴分配的维度数
        # 采用 3D RoPE 机制，将 embed_dims 按比例分配给 x, y, z 轴
        # 建议比例为：x 占 1/3，y 占 1/3，z 占 1/3
        # 确保分配的维度数为偶数，如有余数补在 x 或 y 上
        dim_per_axis = embed_dims // 3
        remainder = embed_dims % 3
        
        self.dim_x = dim_per_axis + (remainder // 2)
        self.dim_y = dim_per_axis + (remainder % 2)
        self.dim_z = dim_per_axis
        
        # 确保各轴维度为偶数
        if self.dim_x % 2 != 0:
            self.dim_x += 1
            self.dim_y -= 1
        if self.dim_y % 2 != 0:
            self.dim_y += 1
            self.dim_z -= 1
        if self.dim_z % 2 != 0:
            self.dim_z += 1
            self.dim_x -= 1
        
        # 生成频率基底
        # 针对 x, y, z 分别生成类似于传统 Transformer (Vaswani et al.) 的频率基底 inv_freq
        inv_freq_x = 1.0 / (10000 ** (torch.arange(0, self.dim_x, 2, dtype=torch.float32) / self.dim_x))
        inv_freq_y = 1.0 / (10000 ** (torch.arange(0, self.dim_y, 2, dtype=torch.float32) / self.dim_y))
        inv_freq_z = 1.0 / (10000 ** (torch.arange(0, self.dim_z, 2, dtype=torch.float32) / self.dim_z))
        
        # 注册为缓冲区，确保随模型移动到设备
        self.register_buffer("inv_freq_x", inv_freq_x)
        self.register_buffer("inv_freq_y", inv_freq_y)
        self.register_buffer("inv_freq_z", inv_freq_z)
    
    def forward(self, reference_points):
        """
        前向传播
        
        Args:
            reference_points: 形状 [B, Num_Anchors, 3] (对应归一化后的 x, y, z)
            
        Returns:
            pos_sin: 正弦嵌入 [B, Num_Anchors, embed_dims]
            pos_cos: 余弦嵌入 [B, Num_Anchors, embed_dims]
        """
        B, Num_Anchors, _ = reference_points.shape
        device = reference_points.device
        
        # 确保频率基底在正确的设备上
        inv_freq_x = self.inv_freq_x.to(device)
        inv_freq_y = self.inv_freq_y.to(device)
        inv_freq_z = self.inv_freq_z.to(device)
        
        # 提取 x, y, z 坐标
        x = reference_points[..., 0]  # [B, Num_Anchors]
        y = reference_points[..., 1]  # [B, Num_Anchors]
        z = reference_points[..., 2]  # [B, Num_Anchors]
        
        # 计算频率特征
        # 将输入的 reference_points 展开，在对应轴上与频率基底进行张量乘法
        freq_x = torch.einsum('ba,d->bad', x, inv_freq_x)  # [B, Num_Anchors, dim_x/2]
        freq_y = torch.einsum('ba,d->bad', y, inv_freq_y)  # [B, Num_Anchors, dim_y/2]
        freq_z = torch.einsum('ba,d->bad', z, inv_freq_z)  # [B, Num_Anchors, dim_z/2]
        
        # 生成正弦和余弦嵌入
        sin_x = torch.sin(freq_x)
        cos_x = torch.cos(freq_x)
        sin_y = torch.sin(freq_y)
        cos_y = torch.cos(freq_y)
        sin_z = torch.sin(freq_z)
        cos_z = torch.cos(freq_z)
        
        # 拼接各轴的正弦和余弦嵌入
        # 对于每个轴，将 sin 和 cos 交错排列
        sin_x = torch.stack([sin_x, sin_x], dim=-1).flatten(-2)  # [B, Num_Anchors, dim_x]
        cos_x = torch.stack([cos_x, cos_x], dim=-1).flatten(-2)  # [B, Num_Anchors, dim_x]
        sin_y = torch.stack([sin_y, sin_y], dim=-1).flatten(-2)  # [B, Num_Anchors, dim_y]
        cos_y = torch.stack([cos_y, cos_y], dim=-1).flatten(-2)  # [B, Num_Anchors, dim_y]
        sin_z = torch.stack([sin_z, sin_z], dim=-1).flatten(-2)  # [B, Num_Anchors, dim_z]
        cos_z = torch.stack([cos_z, cos_z], dim=-1).flatten(-2)  # [B, Num_Anchors, dim_z]
        
        # 拼接所有轴的嵌入
        pos_sin = torch.cat([sin_x, sin_y, sin_z], dim=-1)  # [B, Num_Anchors, embed_dims]
        pos_cos = torch.cat([cos_x, cos_y, cos_z], dim=-1)  # [B, Num_Anchors, embed_dims]
        
        return pos_sin, pos_cos