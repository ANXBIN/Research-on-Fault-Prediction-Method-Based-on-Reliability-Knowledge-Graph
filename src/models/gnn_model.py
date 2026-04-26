#!/usr/bin/env python3
"""
GNN模型：普通GNN和GNN+知识图谱融合
架构：特征嵌入 + 消息传递层 + 交叉注意力 + 图卷积
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import numpy as np
from sklearn.neighbors import NearestNeighbors

from src.models.mlp_model import BaseModelWrapper


class FeatureEmbedding(nn.Module):
    """特征嵌入层：线性投影 + BatchNorm"""

    def __init__(self, in_dim, hidden_dim, dropout=0.2):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.proj(x)


class ImprovedGCNLayer(nn.Module):
    """改进的GCN层：带BatchNorm和ReLU"""

    def __init__(self, in_dim, out_dim, dropout=0.2):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        x = self.linear(x)
        x = torch.matmul(adj, x)
        x = self.bn(x)
        x = F.relu(x)
        return self.dropout(x)


class GraphBlock(nn.Module):
    """图卷积块：双层GCN + 残差连接"""

    def __init__(self, hidden_dim, dropout=0.2):
        super().__init__()
        self.conv1 = ImprovedGCNLayer(hidden_dim, hidden_dim, dropout)
        self.conv2 = ImprovedGCNLayer(hidden_dim, hidden_dim, dropout)

    def forward(self, x, adj):
        identity = x
        x = self.conv1(x, adj)
        x = self.conv2(x, adj)
        return x + identity


class GNNKG1D_V2(nn.Module):
    """GNN-KG V2：KG嵌入投影增强 + 深层融合

    在V1基础上：
    1. 增加KG嵌入投影层（33→64维），丰富KG信息
    2. 增加深层残差融合块，提升表达能力
    3. 使用更大的隐藏维度
    """

    def __init__(self, in_channels, kg_embedding_dim, num_classes,
                 hidden_dim=256, dropout=0.3):
        super().__init__()

        self.in_channels = in_channels
        self.kg_embedding_dim = kg_embedding_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # KG嵌入投影增强层（33→hidden_dim//2）
        self.kg_proj = nn.Sequential(
            nn.Linear(kg_embedding_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 特征和增强KG拼接后统一投影
        self.fusion_embed = nn.Sequential(
            nn.Linear(in_channels + hidden_dim // 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 图卷积（与V1相同）
        self.conv1 = ImprovedGCNLayer(hidden_dim, hidden_dim, dropout)
        self.conv2 = ImprovedGCNLayer(hidden_dim, hidden_dim, dropout)

        # V2新增：深层残差融合块
        self.deep_fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
        )

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x, kg_embedding, adj=None):
        if adj is None:
            adj = torch.eye(x.size(0), device=x.device)

        # 1. KG嵌入投影增强
        kg_enhanced = self.kg_proj(kg_embedding)

        # 2. 特征和增强KG拼接
        x_fused = torch.cat([x, kg_enhanced], dim=1)
        x_fused = self.fusion_embed(x_fused)

        # 3. 图卷积 + 残差
        h = self.conv1(x_fused, adj)
        h = self.conv2(h, adj)
        h = h + x_fused  # 残差

        # 4. 深层残差融合
        h2 = self.deep_fusion(h)
        h = h + h2

        return self.classifier(h)


class GNN1D(nn.Module):
    """GNN模型 - 简化版

    架构：特征嵌入 → 双路图卷积 → 分类器
    """

    def __init__(self, in_channels, num_classes, hidden_dim=256,
                 num_layers=3, dropout=0.3):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.feature_embed = FeatureEmbedding(in_channels, hidden_dim, dropout)

        self.conv1 = ImprovedGCNLayer(hidden_dim, hidden_dim, dropout)
        self.conv2 = ImprovedGCNLayer(hidden_dim, hidden_dim, dropout)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x, adj=None):
        if adj is None:
            adj = torch.eye(x.size(0), device=x.device)

        x = self.feature_embed(x)
        h = self.conv1(x, adj)
        h = self.conv2(h, adj)
        h = h + x
        return self.classifier(h)


class GNNKG1D(nn.Module):
    """GNN + KG融合模型 - 简化版

    架构：特征+KG concat → 单层强图卷积 → 分类
    简化设计减少过拟合风险
    """

    def __init__(self, in_channels, kg_embedding_dim, num_classes,
                 hidden_dim=256, num_layers=3, dropout=0.3):
        super().__init__()

        self.in_channels = in_channels
        self.kg_embedding_dim = kg_embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # 特征和KG拼接后统一投影
        self.fusion_embed = nn.Sequential(
            nn.Linear(in_channels + kg_embedding_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 单层强图卷积（双路）
        self.conv1 = ImprovedGCNLayer(hidden_dim, hidden_dim, dropout)
        self.conv2 = ImprovedGCNLayer(hidden_dim, hidden_dim, dropout)

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x, kg_embedding, adj=None):
        if adj is None:
            adj = torch.eye(x.size(0), device=x.device)

        # 特征和KG拼接
        x_fused = torch.cat([x, kg_embedding], dim=1)
        x_fused = self.fusion_embed(x_fused)

        # 双路图卷积 + 残差
        h = self.conv1(x_fused, adj)
        h = self.conv2(h, adj)
        h = h + x_fused  # 残差

        return self.classifier(h)


def build_batch_adjacency(X_batch, k=30):
    """构建邻接矩阵：高斯核相似度 + 对称归一化拉普拉斯"""
    batch_size = len(X_batch)
    k = min(max(k, 1), batch_size - 1)

    X_norm = X_batch / (np.linalg.norm(X_batch, axis=1, keepdims=True) + 1e-8)

    nn_model = NearestNeighbors(n_neighbors=k + 1, algorithm='ball_tree', metric='euclidean')
    nn_model.fit(X_norm)
    distances, indices = nn_model.kneighbors(X_norm)

    adj = np.zeros((batch_size, batch_size), dtype=np.float32)
    for i in range(batch_size):
        neighbor_indices = indices[i, 1:]
        dists = distances[i, 1:]
        weights = np.exp(-dists ** 2 / 2)
        adj[i, neighbor_indices] = weights
        adj[neighbor_indices, i] = weights

    adj = adj + np.eye(batch_size, dtype=np.float32)

    deg = np.sum(adj, axis=1, keepdims=True)
    deg_inv_sqrt = 1.0 / np.sqrt(deg + 1e-8)
    adj = adj * deg_inv_sqrt * deg_inv_sqrt.T

    return torch.tensor(adj, dtype=torch.float)


class GNNModel(BaseModelWrapper):
    """GNN模型包装类"""

    def __init__(self, config_path='config.yaml'):
        super().__init__(config_path, 'gnn')
        model_cfg = self.config['models'].get('gnn', {})
        self.hidden_dim = model_cfg.get('hidden_dim', 256)
        self.num_layers = model_cfg.get('num_layers', 4)
        self.batch_size = model_cfg.get('batch_size', 256)
        self.scheduler = None

    def build_model(self, n_features, n_classes):
        self.model = GNN1D(
            in_channels=n_features,
            num_classes=n_classes,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-4
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.epochs, eta_min=1e-6
        )
        return self.model

    def train_epoch(self, X, y):
        self.model.train()
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        total_loss = 0
        total_correct = 0
        total_samples = 0

        for start in range(0, len(X), self.batch_size):
            end = min(start + self.batch_size, len(X))
            batch_idx = indices[start:end]
            X_batch = X[batch_idx]
            y_batch = y[batch_idx]
            adj = build_batch_adjacency(X_batch, k=30).to(self.device)
            X_tensor = torch.tensor(X_batch, dtype=torch.float).to(self.device)
            y_tensor = torch.tensor(y_batch, dtype=torch.long).to(self.device)
            self.optimizer.zero_grad()
            out = self.model(X_tensor, adj)
            loss = F.nll_loss(F.log_softmax(out, dim=1), y_tensor)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total_loss += loss.item() * len(y_batch)
            pred = out.argmax(dim=1)
            total_correct += (pred == y_tensor).sum().item()
            total_samples += len(y_batch)

        self.scheduler.step()
        return total_loss / total_samples, total_correct / total_samples

    @torch.no_grad()
    def evaluate(self, X, y):
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        all_preds = []

        for start in range(0, len(X), self.batch_size):
            end = min(start + self.batch_size, len(X))
            X_batch = X[start:end]
            y_batch = y[start:end]
            adj = build_batch_adjacency(X_batch, k=30).to(self.device)
            X_tensor = torch.tensor(X_batch, dtype=torch.float).to(self.device)
            y_tensor = torch.tensor(y_batch, dtype=torch.long).to(self.device)
            out = self.model(X_tensor, adj)
            loss = F.nll_loss(F.log_softmax(out, dim=1), y_tensor)
            total_loss += loss.item() * len(y_batch)
            pred = out.argmax(dim=1)
            total_correct += (pred == y_tensor).sum().item()
            total_samples += len(y_batch)
            all_preds.extend(pred.cpu().numpy())

        return {
            'loss': total_loss / total_samples,
            'accuracy': total_correct / total_samples
        }, np.array(all_preds)

    def predict(self, X):
        self.model.eval()
        all_preds = []
        for start in range(0, len(X), self.batch_size):
            X_batch = X[start:start + self.batch_size]
            adj = build_batch_adjacency(X_batch, k=30).to(self.device)
            X_tensor = torch.tensor(X_batch, dtype=torch.float).to(self.device)
            out = self.model(X_tensor, adj)
            all_preds.extend(out.argmax(dim=1).cpu().numpy())
        return np.array(all_preds)

    def save_model(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'fault_to_idx': self.fault_to_idx,
            'config': {
                'hidden_dim': self.hidden_dim,
                'num_layers': self.num_layers,
                'dropout': self.dropout,
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size
            }
        }, path)
        print(f"[INFO] 模型已保存至: {path}")

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.fault_to_idx = checkpoint['fault_to_idx']
        self.batch_size = checkpoint.get('config', {}).get('batch_size', 256)
        print(f"[INFO] 模型已从: {path} 加载")


class GNNKGModel(BaseModelWrapper):
    """GNN + KG融合模型包装类"""

    def __init__(self, config_path='config.yaml'):
        super().__init__(config_path, 'gnn')
        model_cfg = self.config['models'].get('gnn', {})
        self.hidden_dim = model_cfg.get('hidden_dim', 256)
        self.num_layers = model_cfg.get('num_layers', 4)
        self.batch_size = model_cfg.get('batch_size', 256)
        self.kg_embedding_dim = model_cfg.get('kg_embedding_dim', 33)
        self.scheduler = None

    def build_model(self, n_features, n_classes):
        self.model = GNNKG1D(
            in_channels=n_features,
            kg_embedding_dim=self.kg_embedding_dim,
            num_classes=n_classes,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-4
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.epochs, eta_min=1e-6
        )
        return self.model

    def train_epoch(self, X, y, kg_embeddings):
        self.model.train()
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        total_loss = 0
        total_correct = 0
        total_samples = 0

        for start in range(0, len(X), self.batch_size):
            end = min(start + self.batch_size, len(X))
            batch_idx = indices[start:end]
            X_batch = X[batch_idx]
            y_batch = y[batch_idx]
            kg_batch = kg_embeddings[batch_idx]
            adj = build_batch_adjacency(X_batch, k=30).to(self.device)
            X_tensor = torch.tensor(X_batch, dtype=torch.float).to(self.device)
            kg_tensor = torch.tensor(kg_batch, dtype=torch.float).to(self.device)
            y_tensor = torch.tensor(y_batch, dtype=torch.long).to(self.device)
            self.optimizer.zero_grad()
            out = self.model(X_tensor, kg_tensor, adj)
            loss = F.nll_loss(F.log_softmax(out, dim=1), y_tensor)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total_loss += loss.item() * len(y_batch)
            pred = out.argmax(dim=1)
            total_correct += (pred == y_tensor).sum().item()
            total_samples += len(y_batch)

        self.scheduler.step()
        return total_loss / total_samples, total_correct / total_samples

    @torch.no_grad()
    def evaluate(self, X, y, kg_embeddings):
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        all_preds = []

        for start in range(0, len(X), self.batch_size):
            end = min(start + self.batch_size, len(X))
            X_batch = X[start:end]
            y_batch = y[start:end]
            kg_batch = kg_embeddings[start:end]
            adj = build_batch_adjacency(X_batch, k=30).to(self.device)
            X_tensor = torch.tensor(X_batch, dtype=torch.float).to(self.device)
            kg_tensor = torch.tensor(kg_batch, dtype=torch.float).to(self.device)
            y_tensor = torch.tensor(y_batch, dtype=torch.long).to(self.device)
            out = self.model(X_tensor, kg_tensor, adj)
            loss = F.nll_loss(F.log_softmax(out, dim=1), y_tensor)
            total_loss += loss.item() * len(y_batch)
            pred = out.argmax(dim=1)
            total_correct += (pred == y_tensor).sum().item()
            total_samples += len(y_batch)
            all_preds.extend(pred.cpu().numpy())

        return {
            'loss': total_loss / total_samples,
            'accuracy': total_correct / total_samples
        }, np.array(all_preds)

    def predict(self, X, kg_embeddings):
        self.model.eval()
        all_preds = []
        for start in range(0, len(X), self.batch_size):
            X_batch = X[start:start + self.batch_size]
            kg_batch = kg_embeddings[start:start + self.batch_size]
            adj = build_batch_adjacency(X_batch, k=30).to(self.device)
            X_tensor = torch.tensor(X_batch, dtype=torch.float).to(self.device)
            kg_tensor = torch.tensor(kg_batch, dtype=torch.float).to(self.device)
            out = self.model(X_tensor, kg_tensor, adj)
            all_preds.extend(out.argmax(dim=1).cpu().numpy())
        return np.array(all_preds)

    def save_model(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'fault_to_idx': self.fault_to_idx,
            'config': {
                'hidden_dim': self.hidden_dim,
                'num_layers': self.num_layers,
                'kg_embedding_dim': self.kg_embedding_dim,
                'dropout': self.dropout,
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size
            }
        }, path)
        print(f"[INFO] 模型已保存至: {path}")

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.fault_to_idx = checkpoint['fault_to_idx']
        self.batch_size = checkpoint.get('config', {}).get('batch_size', 256)
        print(f"[INFO] 模型已从: {path} 加载")


class GNNKGModelV2(BaseModelWrapper):
    """GNN-KG V2：KG嵌入投影增强 + 深层残差融合

    在V1基础上改进：
    1. KG嵌入投影层（33→hidden_dim//2），丰富KG信息
    2. 深层残差融合块，提升表达能力
    3. 使用静态故障嵌入（与V1相同），不依赖标签
    """

    def __init__(self, config_path='config.yaml'):
        super().__init__(config_path, 'gnn_kg_v2')
        model_cfg = self.config['models'].get('gnn_kg_v2', {})
        self.hidden_dim = model_cfg.get('hidden_dim', 256)
        self.num_layers = model_cfg.get('num_layers', 3)
        self.kg_embedding_dim = model_cfg.get('kg_embedding_dim', 33)
        self.batch_size = model_cfg.get('batch_size', 256)
        self.scheduler = None

    def build_model(self, n_features, n_classes, fault_to_idx=None):
        self.model = GNNKG1D_V2(
            in_channels=n_features,
            kg_embedding_dim=self.kg_embedding_dim,
            num_classes=n_classes,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-4
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.epochs, eta_min=1e-6
        )
        return self.model

    def train_epoch(self, X, y, kg_embeddings):
        self.model.train()
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        total_loss = 0
        total_correct = 0
        total_samples = 0

        for start in range(0, len(X), self.batch_size):
            end = min(start + self.batch_size, len(X))
            batch_idx = indices[start:end]
            X_batch = X[batch_idx]
            y_batch = y[batch_idx]
            kg_batch = kg_embeddings[batch_idx]
            adj = build_batch_adjacency(X_batch, k=30).to(self.device)
            X_tensor = torch.tensor(X_batch, dtype=torch.float).to(self.device)
            kg_tensor = torch.tensor(kg_batch, dtype=torch.float).to(self.device)
            y_tensor = torch.tensor(y_batch, dtype=torch.long).to(self.device)
            self.optimizer.zero_grad()
            out = self.model(X_tensor, kg_tensor, adj)
            loss = F.nll_loss(F.log_softmax(out, dim=1), y_tensor)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total_loss += loss.item() * len(y_batch)
            pred = out.argmax(dim=1)
            total_correct += (pred == y_tensor).sum().item()
            total_samples += len(y_batch)

        self.scheduler.step()
        return total_loss / total_samples, total_correct / total_samples

    @torch.no_grad()
    def evaluate(self, X, y, kg_embeddings):
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        all_preds = []

        for start in range(0, len(X), self.batch_size):
            end = min(start + self.batch_size, len(X))
            X_batch = X[start:end]
            y_batch = y[start:end]
            kg_batch = kg_embeddings[start:end]
            adj = build_batch_adjacency(X_batch, k=30).to(self.device)
            X_tensor = torch.tensor(X_batch, dtype=torch.float).to(self.device)
            kg_tensor = torch.tensor(kg_batch, dtype=torch.float).to(self.device)
            y_tensor = torch.tensor(y_batch, dtype=torch.long).to(self.device)
            out = self.model(X_tensor, kg_tensor, adj)
            loss = F.nll_loss(F.log_softmax(out, dim=1), y_tensor)
            total_loss += loss.item() * len(y_batch)
            pred = out.argmax(dim=1)
            total_correct += (pred == y_tensor).sum().item()
            total_samples += len(y_batch)
            all_preds.extend(pred.cpu().numpy())

        return {'loss': total_loss / total_samples, 'accuracy': total_correct / total_samples}, np.array(all_preds)

    def predict(self, X, kg_embeddings):
        self.model.eval()
        all_preds = []
        for start in range(0, len(X), self.batch_size):
            X_batch = X[start:start + self.batch_size]
            kg_batch = kg_embeddings[start:start + self.batch_size]
            adj = build_batch_adjacency(X_batch, k=30).to(self.device)
            X_tensor = torch.tensor(X_batch, dtype=torch.float).to(self.device)
            kg_tensor = torch.tensor(kg_batch, dtype=torch.float).to(self.device)
            out = self.model(X_tensor, kg_tensor, adj)
            all_preds.extend(out.argmax(dim=1).cpu().numpy())
        return np.array(all_preds)

    def save_model(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'fault_to_idx': self.fault_to_idx,
            'config': {
                'hidden_dim': self.hidden_dim,
                'kg_embedding_dim': self.kg_embedding_dim,
                'dropout': self.dropout,
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size
            }
        }, path)
        print(f"[INFO] 模型已保存至: {path}")

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.fault_to_idx = checkpoint['fault_to_idx']
        self.batch_size = checkpoint.get('config', {}).get('batch_size', 256)
        print(f"[INFO] 模型已从: {path} 加载")


# ============================================================
# GAT: 图注意力网络
# ============================================================

class GATLayer(nn.Module):
    """图注意力层：多头注意力，用adj作为邻居mask"""

    def __init__(self, in_dim, out_dim, heads=4, dropout=0.3, concat=True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.concat = concat

        self.W = nn.Linear(in_dim, out_dim * heads, bias=False)
        self.a_src = nn.Parameter(torch.zeros(heads, out_dim))
        self.a_dst = nn.Parameter(torch.zeros(heads, out_dim))
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        N = x.size(0)
        h = self.heads
        d = self.out_dim

        Wh = self.W(x).view(N, h, d)

        attn_src = (Wh * self.a_src).sum(dim=-1)
        attn_dst = (Wh * self.a_dst).sum(dim=-1)
        attn = attn_src.unsqueeze(1) + attn_dst.unsqueeze(0)
        attn = self.leaky_relu(attn)

        mask = (adj == 0).unsqueeze(-1).expand_as(attn)
        attn = attn.masked_fill(mask, -9e15)

        attn = F.softmax(attn, dim=1)
        attn = self.dropout(attn)

        out = torch.einsum('ijh,jhd->ihd', attn, Wh)

        if self.concat:
            return out.reshape(N, h * d)
        else:
            return out.mean(dim=1)


class GAT1D(nn.Module):
    """GAT模型：图注意力网络"""

    def __init__(self, in_channels, num_classes, hidden_dim=256,
                 heads=4, dropout=0.3):
        super().__init__()
        self.feature_embed = FeatureEmbedding(in_channels, hidden_dim, dropout)
        self.gat1 = GATLayer(hidden_dim, hidden_dim, heads, dropout, concat=True)
        self.gat2 = GATLayer(hidden_dim * heads, hidden_dim, heads=1, dropout=dropout, concat=False)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x, adj=None):
        if adj is None:
            adj = torch.eye(x.size(0), device=x.device)
        x_emb = self.feature_embed(x)
        h = self.gat1(x_emb, adj)
        h = self.gat2(h, adj)
        h = h + x_emb
        return self.classifier(h)


class GATKG1D(nn.Module):
    """GAT + KG融合模型：早期拼接"""

    def __init__(self, in_channels, kg_embedding_dim, num_classes,
                 hidden_dim=256, heads=4, dropout=0.3):
        super().__init__()
        self.fusion_embed = nn.Sequential(
            nn.Linear(in_channels + kg_embedding_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.gat1 = GATLayer(hidden_dim, hidden_dim, heads, dropout, concat=True)
        self.gat2 = GATLayer(hidden_dim * heads, hidden_dim, heads=1, dropout=dropout, concat=False)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x, kg_embedding, adj=None):
        if adj is None:
            adj = torch.eye(x.size(0), device=x.device)
        x_fused = self.fusion_embed(torch.cat([x, kg_embedding], dim=1))
        h = self.gat1(x_fused, adj)
        h = self.gat2(h, adj)
        h = h + x_fused
        return self.classifier(h)


class GATKG1D_V2(nn.Module):
    """GAT + KG V2：KG投影增强 + 深层融合"""

    def __init__(self, in_channels, kg_embedding_dim, num_classes,
                 hidden_dim=256, heads=4, dropout=0.3):
        super().__init__()
        self.kg_proj = nn.Sequential(
            nn.Linear(kg_embedding_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.fusion_embed = nn.Sequential(
            nn.Linear(in_channels + hidden_dim // 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.gat1 = GATLayer(hidden_dim, hidden_dim, heads, dropout, concat=True)
        self.gat2 = GATLayer(hidden_dim * heads, hidden_dim, heads=1, dropout=dropout, concat=False)
        self.deep_fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x, kg_embedding, adj=None):
        if adj is None:
            adj = torch.eye(x.size(0), device=x.device)
        kg_enhanced = self.kg_proj(kg_embedding)
        x_fused = self.fusion_embed(torch.cat([x, kg_enhanced], dim=1))
        h = self.gat1(x_fused, adj)
        h = self.gat2(h, adj)
        h = h + x_fused
        h2 = self.deep_fusion(h)
        h = h + h2
        return self.classifier(h)


class GATModel(BaseModelWrapper):
    """GAT模型包装类"""

    def __init__(self, config_path='config.yaml'):
        super().__init__(config_path, 'gat')
        model_cfg = self.config['models'].get('gat', {})
        self.hidden_dim = model_cfg.get('hidden_dim', 256)
        self.heads = model_cfg.get('heads', 4)
        self.batch_size = model_cfg.get('batch_size', 256)
        self.scheduler = None

    def build_model(self, n_features, n_classes):
        self.model = GAT1D(
            in_channels=n_features,
            num_classes=n_classes,
            hidden_dim=self.hidden_dim,
            heads=self.heads,
            dropout=self.dropout
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.epochs, eta_min=1e-6
        )
        return self.model

    def train_epoch(self, X, y):
        self.model.train()
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        total_loss = 0
        total_correct = 0
        total_samples = 0
        for start in range(0, len(X), self.batch_size):
            end = min(start + self.batch_size, len(X))
            batch_idx = indices[start:end]
            X_batch, y_batch = X[batch_idx], y[batch_idx]
            adj = build_batch_adjacency(X_batch, k=30).to(self.device)
            X_tensor = torch.tensor(X_batch, dtype=torch.float).to(self.device)
            y_tensor = torch.tensor(y_batch, dtype=torch.long).to(self.device)
            self.optimizer.zero_grad()
            out = self.model(X_tensor, adj)
            loss = F.nll_loss(F.log_softmax(out, dim=1), y_tensor)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total_loss += loss.item() * len(y_batch)
            total_correct += (out.argmax(1) == y_tensor).sum().item()
            total_samples += len(y_batch)
        self.scheduler.step()
        return total_loss / total_samples, total_correct / total_samples

    @torch.no_grad()
    def evaluate(self, X, y):
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        all_preds = []
        for start in range(0, len(X), self.batch_size):
            end = min(start + self.batch_size, len(X))
            X_batch, y_batch = X[start:end], y[start:end]
            adj = build_batch_adjacency(X_batch, k=30).to(self.device)
            X_tensor = torch.tensor(X_batch, dtype=torch.float).to(self.device)
            y_tensor = torch.tensor(y_batch, dtype=torch.long).to(self.device)
            out = self.model(X_tensor, adj)
            loss = F.nll_loss(F.log_softmax(out, dim=1), y_tensor)
            total_loss += loss.item() * len(y_batch)
            total_correct += (out.argmax(1) == y_tensor).sum().item()
            total_samples += len(y_batch)
            all_preds.extend(out.argmax(1).cpu().numpy())
        return {'loss': total_loss / total_samples, 'accuracy': total_correct / total_samples}, np.array(all_preds)

    def predict(self, X):
        self.model.eval()
        all_preds = []
        for start in range(0, len(X), self.batch_size):
            X_batch = X[start:start + self.batch_size]
            adj = build_batch_adjacency(X_batch, k=30).to(self.device)
            X_tensor = torch.tensor(X_batch, dtype=torch.float).to(self.device)
            out = self.model(X_tensor, adj)
            all_preds.extend(out.argmax(1).cpu().numpy())
        return np.array(all_preds)

    def save_model(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'fault_to_idx': self.fault_to_idx,
            'config': {
                'hidden_dim': self.hidden_dim,
                'heads': self.heads,
                'dropout': self.dropout,
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size
            }
        }, path)
        print(f"[INFO] 模型已保存至: {path}")

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.fault_to_idx = checkpoint['fault_to_idx']
        self.batch_size = checkpoint.get('config', {}).get('batch_size', 256)
        print(f"[INFO] 模型已从: {path} 加载")


class GATKGModel(BaseModelWrapper):
    """GAT + KG融合模型包装类"""

    def __init__(self, config_path='config.yaml'):
        super().__init__(config_path, 'gat_kg')
        model_cfg = self.config['models'].get('gat_kg', {})
        self.hidden_dim = model_cfg.get('hidden_dim', 256)
        self.heads = model_cfg.get('heads', 4)
        self.kg_embedding_dim = model_cfg.get('kg_embedding_dim', 33)
        self.batch_size = model_cfg.get('batch_size', 256)
        self.scheduler = None

    def build_model(self, n_features, n_classes):
        self.model = GATKG1D(
            in_channels=n_features,
            kg_embedding_dim=self.kg_embedding_dim,
            num_classes=n_classes,
            hidden_dim=self.hidden_dim,
            heads=self.heads,
            dropout=self.dropout
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.epochs, eta_min=1e-6
        )
        return self.model

    def train_epoch(self, X, y, kg_embeddings):
        self.model.train()
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        total_loss = 0
        total_correct = 0
        total_samples = 0
        for start in range(0, len(X), self.batch_size):
            end = min(start + self.batch_size, len(X))
            batch_idx = indices[start:end]
            X_batch, y_batch, kg_batch = X[batch_idx], y[batch_idx], kg_embeddings[batch_idx]
            adj = build_batch_adjacency(X_batch, k=30).to(self.device)
            X_tensor = torch.tensor(X_batch, dtype=torch.float).to(self.device)
            kg_tensor = torch.tensor(kg_batch, dtype=torch.float).to(self.device)
            y_tensor = torch.tensor(y_batch, dtype=torch.long).to(self.device)
            self.optimizer.zero_grad()
            out = self.model(X_tensor, kg_tensor, adj)
            loss = F.nll_loss(F.log_softmax(out, dim=1), y_tensor)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total_loss += loss.item() * len(y_batch)
            total_correct += (out.argmax(1) == y_tensor).sum().item()
            total_samples += len(y_batch)
        self.scheduler.step()
        return total_loss / total_samples, total_correct / total_samples

    @torch.no_grad()
    def evaluate(self, X, y, kg_embeddings):
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        all_preds = []
        for start in range(0, len(X), self.batch_size):
            end = min(start + self.batch_size, len(X))
            X_batch, y_batch, kg_batch = X[start:end], y[start:end], kg_embeddings[start:end]
            adj = build_batch_adjacency(X_batch, k=30).to(self.device)
            X_tensor = torch.tensor(X_batch, dtype=torch.float).to(self.device)
            kg_tensor = torch.tensor(kg_batch, dtype=torch.float).to(self.device)
            y_tensor = torch.tensor(y_batch, dtype=torch.long).to(self.device)
            out = self.model(X_tensor, kg_tensor, adj)
            loss = F.nll_loss(F.log_softmax(out, dim=1), y_tensor)
            total_loss += loss.item() * len(y_batch)
            total_correct += (out.argmax(1) == y_tensor).sum().item()
            total_samples += len(y_batch)
            all_preds.extend(out.argmax(1).cpu().numpy())
        return {'loss': total_loss / total_samples, 'accuracy': total_correct / total_samples}, np.array(all_preds)

    def predict(self, X, kg_embeddings):
        self.model.eval()
        all_preds = []
        for start in range(0, len(X), self.batch_size):
            X_batch = X[start:start + self.batch_size]
            kg_batch = kg_embeddings[start:start + self.batch_size]
            adj = build_batch_adjacency(X_batch, k=30).to(self.device)
            X_tensor = torch.tensor(X_batch, dtype=torch.float).to(self.device)
            kg_tensor = torch.tensor(kg_batch, dtype=torch.float).to(self.device)
            out = self.model(X_tensor, kg_tensor, adj)
            all_preds.extend(out.argmax(1).cpu().numpy())
        return np.array(all_preds)

    def save_model(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'fault_to_idx': self.fault_to_idx,
            'config': {
                'hidden_dim': self.hidden_dim,
                'heads': self.heads,
                'kg_embedding_dim': self.kg_embedding_dim,
                'dropout': self.dropout,
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size
            }
        }, path)
        print(f"[INFO] 模型已保存至: {path}")

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.fault_to_idx = checkpoint['fault_to_idx']
        self.batch_size = checkpoint.get('config', {}).get('batch_size', 256)
        print(f"[INFO] 模型已从: {path} 加载")


class GATKGModelV2(BaseModelWrapper):
    """GAT-KG V2：KG投影增强 + 深层残差融合"""

    def __init__(self, config_path='config.yaml'):
        super().__init__(config_path, 'gat_kg_v2')
        model_cfg = self.config['models'].get('gat_kg_v2', {})
        self.hidden_dim = model_cfg.get('hidden_dim', 256)
        self.heads = model_cfg.get('heads', 4)
        self.kg_embedding_dim = model_cfg.get('kg_embedding_dim', 33)
        self.batch_size = model_cfg.get('batch_size', 256)
        self.scheduler = None

    def build_model(self, n_features, n_classes):
        self.model = GATKG1D_V2(
            in_channels=n_features,
            kg_embedding_dim=self.kg_embedding_dim,
            num_classes=n_classes,
            hidden_dim=self.hidden_dim,
            heads=self.heads,
            dropout=self.dropout
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.epochs, eta_min=1e-6
        )
        return self.model

    def train_epoch(self, X, y, kg_embeddings):
        self.model.train()
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        total_loss = 0
        total_correct = 0
        total_samples = 0
        for start in range(0, len(X), self.batch_size):
            end = min(start + self.batch_size, len(X))
            batch_idx = indices[start:end]
            X_batch, y_batch, kg_batch = X[batch_idx], y[batch_idx], kg_embeddings[batch_idx]
            adj = build_batch_adjacency(X_batch, k=30).to(self.device)
            X_tensor = torch.tensor(X_batch, dtype=torch.float).to(self.device)
            kg_tensor = torch.tensor(kg_batch, dtype=torch.float).to(self.device)
            y_tensor = torch.tensor(y_batch, dtype=torch.long).to(self.device)
            self.optimizer.zero_grad()
            out = self.model(X_tensor, kg_tensor, adj)
            loss = F.nll_loss(F.log_softmax(out, dim=1), y_tensor)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total_loss += loss.item() * len(y_batch)
            total_correct += (out.argmax(1) == y_tensor).sum().item()
            total_samples += len(y_batch)
        self.scheduler.step()
        return total_loss / total_samples, total_correct / total_samples

    @torch.no_grad()
    def evaluate(self, X, y, kg_embeddings):
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        all_preds = []
        for start in range(0, len(X), self.batch_size):
            end = min(start + self.batch_size, len(X))
            X_batch, y_batch, kg_batch = X[start:end], y[start:end], kg_embeddings[start:end]
            adj = build_batch_adjacency(X_batch, k=30).to(self.device)
            X_tensor = torch.tensor(X_batch, dtype=torch.float).to(self.device)
            kg_tensor = torch.tensor(kg_batch, dtype=torch.float).to(self.device)
            y_tensor = torch.tensor(y_batch, dtype=torch.long).to(self.device)
            out = self.model(X_tensor, kg_tensor, adj)
            loss = F.nll_loss(F.log_softmax(out, dim=1), y_tensor)
            total_loss += loss.item() * len(y_batch)
            total_correct += (out.argmax(1) == y_tensor).sum().item()
            total_samples += len(y_batch)
            all_preds.extend(out.argmax(1).cpu().numpy())
        return {'loss': total_loss / total_samples, 'accuracy': total_correct / total_samples}, np.array(all_preds)

    def predict(self, X, kg_embeddings):
        self.model.eval()
        all_preds = []
        for start in range(0, len(X), self.batch_size):
            X_batch = X[start:start + self.batch_size]
            kg_batch = kg_embeddings[start:start + self.batch_size]
            adj = build_batch_adjacency(X_batch, k=30).to(self.device)
            X_tensor = torch.tensor(X_batch, dtype=torch.float).to(self.device)
            kg_tensor = torch.tensor(kg_batch, dtype=torch.float).to(self.device)
            out = self.model(X_tensor, kg_tensor, adj)
            all_preds.extend(out.argmax(1).cpu().numpy())
        return np.array(all_preds)

    def save_model(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'fault_to_idx': self.fault_to_idx,
            'config': {
                'hidden_dim': self.hidden_dim,
                'heads': self.heads,
                'kg_embedding_dim': self.kg_embedding_dim,
                'dropout': self.dropout,
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size
            }
        }, path)
        print(f"[INFO] 模型已保存至: {path}")

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.fault_to_idx = checkpoint['fault_to_idx']
        self.batch_size = checkpoint.get('config', {}).get('batch_size', 256)
        print(f"[INFO] 模型已从: {path} 加载")