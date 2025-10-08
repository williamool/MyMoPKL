import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


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