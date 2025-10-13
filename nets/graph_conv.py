import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from .darknet import BaseConv, Bottleneck

class MultiScaleAttn(nn.Module):
    def __init__(
        self,
        level_channels: list,         # [c3, c4, c5]
        c_text: int = 512,
        attn_embed_dim: int = 128,
        attn_heads: int = 8,
        gcn_hidden: int = 64,
        k: int = 3,
    ) -> None:
        super().__init__()
        assert len(level_channels) == 3, "level_channels should be [c3, c4, c5]"
        c3, c4, c5 = level_channels

        # 每个尺度一套FeatureUpdate（共享于P/Q该尺度）
        self.update_l3 = FeatureUpdate(c_in=c3, c_out=c3, n=1, c_hidden=attn_embed_dim,
                                       num_head=1, c_text=c_text, gcn_hidden=gcn_hidden,
                                       attn_embed_dim=attn_embed_dim)
        self.update_l4 = FeatureUpdate(c_in=c4, c_out=c4, n=1, c_hidden=attn_embed_dim,
                                       num_head=1, c_text=c_text, gcn_hidden=gcn_hidden,
                                       attn_embed_dim=attn_embed_dim)
        self.update_l5 = FeatureUpdate(c_in=c5, c_out=c5, n=1, c_hidden=attn_embed_dim,
                                       num_head=1, c_text=c_text, gcn_hidden=gcn_hidden,
                                       attn_embed_dim=attn_embed_dim)

        # 文本增强：使用 P3'、P4、P5 三尺度更新文本
        self.image_pooling_attn = ImagePoolingAttn(
            ec=attn_embed_dim,
            ch=(c3, c4, c5),
            ct=c_text,
            nh=attn_heads,
            k=k,
            scale=False,
        )

    def forward(self, P_feats: list, Q_feats: list, text_feat: torch.Tensor):
        # 解包并校验
        assert len(P_feats) == 3 and len(Q_feats) == 3, "Expect three scales for P and Q"
        P3, P4, P5 = P_feats
        Q3, Q4, Q5 = Q_feats

        # 1) 更新最高分辨率尺度 P3、Q3
        P3_upd = self.update_l3(P3, text_feat)
        Q3_upd = self.update_l3(Q3, text_feat)

        # 2) 用 [P3', P4, P5] 更新文本特征
        text_feat_upd = self.image_pooling_attn([P3_upd, P4, P5], text_feat)

        # 3) 用新的文本特征依次更新 P4、P5、Q4、Q5
        P4_upd = self.update_l4(P4, text_feat_upd)
        P5_upd = self.update_l5(P5, text_feat_upd)
        Q4_upd = self.update_l4(Q4, text_feat_upd)
        Q5_upd = self.update_l5(Q5, text_feat_upd)

        return [P3_upd, P4_upd, P5_upd, Q3_upd, Q4_upd, Q5_upd], text_feat_upd

class FeatureUpdate(nn.Module):
    def __init__(
        self,
        c_in: int,
        c_out: int,
        n: int = 1,
        c_hidden: int = 128,
        num_head: int = 1,
        c_text: int = 512,
        gcn_hidden: int = 64,
        attn_embed_dim: int = 128,
    ) -> None:
        super().__init__()
        assert c_in % 2 == 0, "c_in must be divisible by 2 to split channels"
        self.c_in = c_in
        self.c_half = c_in // 2
        self.c_out = c_out

        # 后半部分的bottleneck序列
        self.m = nn.ModuleList(
            Bottleneck(self.c_half, self.c_half, shortcut=True, expansion=0.5) for _ in range(n)
        )

        # 图像-文本注意力，输入输出均为c_half
        self.attn = ScoreCompute(
            c1=self.c_half,
            c2=self.c_half,
            num_heads=num_head,
            embed_dim=attn_embed_dim,
            guide_dim=c_text,
            scale=False,
        )

        # 特征融合：(part1, part2, bottleneck(part2), attn) 共4块，每块c_half通道
        self.feat_fusion = BaseConv((n + 3) * self.c_half, self.c_out, ksize=1, stride=1, act=False)

        # 图卷积更新模块：通道保持不变
        self.graph_update = GraphUpdate(self.c_out, gcn_hidden, self.c_out)

    def forward(self, x: torch.Tensor, text_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, c_in, H, W]
            text_feat: [B, num_classes, c_text]
        Returns:
            fused_feat: [B, c_out, H, W]
        """
        # 1) 按通道划分为两部分
        feats_part1, feats_part2 = torch.chunk(x, 2, dim=1)

        # 2) 对后半特征应用bottleneck（extend方式）
        feat_list = [feats_part1, feats_part2]
        feat_list.extend(m(feat_list[-1]) for m in self.m)

        # 3) 图像-文本注意力，得到attn特征和得分图
        attn_feat, score_map = self.attn(feat_list[-1], text_feat)

        # 4) 新特征拼接并融合
        feat_list.append(attn_feat)
        fused_feat = self.feat_fusion(torch.cat(feat_list, 1))

        # 5) 基于得分图的GCN特征更新
        fused_feat = update_features_with_gcn(
            fused_feat, score_map, self.graph_update, k_ratio=0.005, similarity_threshold=0.5
        )

        return fused_feat


class ScoreCompute(nn.Module):
    def __init__(self, c1, c2, num_heads=1, embed_dim=128, guide_dim=512, scale=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        from .darknet import BaseConv  # local import to avoid circular at module load
        self.img_conv = BaseConv(c1, embed_dim, ksize=1, stride=1, act=False) if c1 != embed_dim else None
        self.text_linear = nn.Linear(guide_dim, embed_dim)
        self.bias = nn.Parameter(torch.zeros(num_heads))
        self.proj_conv = BaseConv(c1, c2, ksize=3, stride=1, act=False)
        self.scale = nn.Parameter(torch.ones(1, num_heads, 1, 1)) if scale else 1.0

    def forward(self, img_feat, text_feat):
        bs, _, h, w = img_feat.shape

        if text_feat.dtype != self.text_linear.weight.dtype:
            text_feat = text_feat.to(self.text_linear.weight.dtype)

        text_feat = self.text_linear(text_feat)
        text_feat = text_feat.view(bs, -1, self.num_heads, self.head_dim)

        img_embed = self.img_conv(img_feat) if self.img_conv is not None else img_feat
        img_embed = img_embed.view(bs, self.num_heads, self.head_dim, h, w)

        attn_weight = torch.einsum("bmchw,bnmc->bmhwn", img_embed, text_feat)
        attn_weight = attn_weight.max(dim=-1)[0]
        attn_weight = attn_weight / (self.head_dim**0.5)
        attn_weight = attn_weight + self.bias[None, :, None, None]
        attn_weight = attn_weight.sigmoid() * self.scale
        score_map = attn_weight.mean(dim=1, keepdim=True)

        img_feat = self.proj_conv(img_feat)
        img_feat = img_feat.view(bs, self.num_heads, -1, h, w)
        img_feat = img_feat * attn_weight.unsqueeze(2)

        return img_feat.view(bs, -1, h, w), score_map.squeeze(1)

class GraphUpdate(nn.Module):
    """Graph-based feature update module."""

    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.gcn1 = GCNConv(in_channels, hidden_channels)
        self.gcn2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        """Forward pass for graph convolution."""
        # print(x.device,edge_index.device)
        edge_index = edge_index.to(x.device)
        q = self.gcn1(x, edge_index)
        x = F.relu(q)
        x = self.gcn2(x, edge_index)
        return x


def select_top_k_pixels(score, k_ratio=0.04):
    # 选择4%的像素点
    batch_size, height, width = score.shape
    num_pixels = height * width # [4, 64, 64]
    num_select = int(num_pixels * k_ratio)

    # 展平二维得分图到一维
    flat_score = score.view(batch_size, -1)  # [batch, height * width]

    # 返回索引
    _, top_indices = torch.topk(flat_score, num_select, dim=1)  # [batch, num_select]

    # 将一维索引转为二维坐标
    h_indices = top_indices // width
    w_indices = top_indices % width
    top_indices_2d = torch.stack([h_indices, w_indices], dim=-1)  # [batch, num_select, 2]

    return top_indices_2d, top_indices


def build_edge_index_from_features(features, threshold=0.55):
    # 计算选中特征像素的相似度
    norm_features = F.normalize(features, p=2, dim=1)  # L2归一化   
    similarity_matrix = torch.mm(norm_features, norm_features.t())  # 计算相似度矩阵 [num_nodes, num_nodes]
    
    # 超过threshold的像素间建边
    edges = torch.nonzero(similarity_matrix > threshold, as_tuple=False).t()  # 每列表示一条边的两个节点索引 [2, num_edges]
    
    # 没有边连接的节点建自环
    if edges.size(1) == 0:
        num_nodes = features.size(0)
        self_loops = torch.arange(num_nodes).repeat(2, 1)  # [2, num_nodes]
        edges = self_loops

    # 返回边索引矩阵
    return edges


def update_features_with_gcn(z, score, graph_update_module, k_ratio=0.001, similarity_threshold=0.6):
    batch_size, channels, height, width = z.shape
    
    # 得到top0.7%像素索引
    top_indices_2d, flat_indices = select_top_k_pixels(score, k_ratio)
    
    # 提取选中像素的特征
    flat_z = z.view(batch_size, channels, -1)  # 展平特征图 [4, 256, 64 * 64]
    top_features = flat_z.gather(2, flat_indices.unsqueeze(1).expand(-1, channels, -1))  # 提取像素 [4, 256, num_select]
    
    # 将选中特征转换为图卷积节点输入格式
    top_features = top_features.permute(0, 2, 1).reshape(-1, channels)  # [batch * num_select, channel]
    
    # 建边
    edge_index = build_edge_index_from_features(top_features, similarity_threshold)
    
    # 图卷积 特征增强
    updated_features = graph_update_module(top_features, edge_index)  # [batch * num_select, channel]
    
    # 将特征还原为原始格式
    updated_features = updated_features.view(batch_size, -1, channels).permute(0, 2, 1)  # [batch, channel, num_select]
    
    # 将增强后的特征替换回原始特征图
    new_z = z.clone()
    updated_features = updated_features.to(new_z.dtype)
    # 使用非就地操作scatter，避免梯度计算问题
    new_z_flat = new_z.view(batch_size, channels, -1)
    new_z_flat = new_z_flat.scatter(2, flat_indices.unsqueeze(1).expand(-1, channels, -1), updated_features)
    new_z = new_z_flat.view(batch_size, channels, height, width)
    
    return new_z


class ImagePoolingAttn(nn.Module):
    def __init__(self, ec=256, ch=(), ct=512, nh=8, k=3, scale=False):
        super().__init__()

        nf = len(ch)  # 计算特征图的数量
        # 将文本嵌入投影到嵌入空间
        self.query = nn.Sequential(nn.LayerNorm(ct), nn.Linear(ct, ec))
        
        # 将图像特征投影到嵌入空间
        self.key = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.value = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        
        self.proj = nn.Linear(ec, ct)
        self.scale = nn.Parameter(torch.tensor([0.0]), requires_grad=True) if scale else 1.0
        self.projections = nn.ModuleList([nn.Conv2d(in_channels, ec, kernel_size=1) for in_channels in ch])
        self.im_pools = nn.ModuleList([nn.AdaptiveMaxPool2d((k, k)) for _ in range(nf)])
        
        self.ec = ec      # 嵌入通道数
        self.nh = nh      # 注意力头数
        self.nf = nf      # 特征图数量
        self.hc = ec // nh # 每个注意力头的通道数
        self.k = k        # 池化核大小

    def forward(self, x, text):
        bs = x[0].shape[0]  # batch_size
        assert len(x) == self.nf  # 确保特征图数量与预期一致
        
        # 确保text的数据类型与模型权重一致
        # 获取模型的目标数据类型（从第一个投影层获取）
        target_dtype = next(self.projections[0].parameters()).dtype
        if text.dtype != target_dtype:
            text = text.to(target_dtype)
        
        num_patches = self.k**2  # 池化后patch数
        
        # 对每个尺度的特征图进行投影和池化
        x = [pool(proj(x)).view(bs, -1, num_patches) for (x, proj, pool) in zip(x, self.projections, self.im_pools)]
        
        # 特征拼接和维度调整
        x = torch.cat(x, dim=-1).transpose(1, 2) 
        
        # 生成查询、键、值
        q = self.query(text)  
        k = self.key(x)      
        v = self.value(x)    
        
        # 重塑为多头注意力格式
        q = q.reshape(bs, -1, self.nh, self.hc)
        k = k.reshape(bs, -1, self.nh, self.hc)  
        v = v.reshape(bs, -1, self.nh, self.hc) 
        
        # 计算注意力权重
        aw = torch.einsum("bnmc,bkmc->bmnk", q, k)
        aw = aw / (self.hc**0.5)
        aw = F.softmax(aw, dim=-1)  # 在patch维度上归一化，使权重和为1
        
        # 使用注意力权重对值进行加权求和
        x = torch.einsum("bmnk,bkmc->bnmc", aw, v)
        
        x = self.proj(x.reshape(bs, -1, self.ec)) 
        return x * self.scale + text  # 返回增强后的文本嵌入
