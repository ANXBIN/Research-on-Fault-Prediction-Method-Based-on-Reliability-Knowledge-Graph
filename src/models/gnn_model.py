#!/usr/bin/env python3
"""
GNN模型：普通GNN和GNN+知识图谱融合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
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


class GNN1D(nn.Module):
    """GNN模型：特征嵌入 -> 双路图卷积 -> 分类器"""

    def __init__(self, in_channels, num_classes, hidden_dim=256,
                 num_layers=3, dropout=0.3):
        super().__init__()

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
    """GNN + KG融合模型：特征+KG concat -> 双路图卷积 -> 分类"""

    def __init__(self, in_channels, kg_embedding_dim, num_classes,
                 hidden_dim=256, num_layers=3, dropout=0.3):
        super().__init__()

        self.fusion_embed = nn.Sequential(
            nn.Linear(in_channels + kg_embedding_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.conv1 = ImprovedGCNLayer(hidden_dim, hidden_dim, dropout)
        self.conv2 = ImprovedGCNLayer(hidden_dim, hidden_dim, dropout)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x, kg_embedding, adj=None):
        if adj is None:
            adj = torch.eye(x.size(0), device=x.device)

        x_fused = torch.cat([x, kg_embedding], dim=1)
        x_fused = self.fusion_embed(x_fused)

        h = self.conv1(x_fused, adj)
        h = self.conv2(h, adj)
        h = h + x_fused

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


class _GNNBaseWrapper(BaseModelWrapper):
    """GNN包装器公共基类 - 处理batch训练和邻接矩阵构建"""

    def __init__(self, config_path, config_key):
        super().__init__(config_path, config_key)
        model_cfg = self.config['models'].get(config_key, {})
        self.hidden_dim = model_cfg.get('hidden_dim', 256)
        self.num_layers = model_cfg.get('num_layers', 4)
        self.batch_size = model_cfg.get('batch_size', 256)
        self.scheduler = None

    def _init_optimizer(self):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.epochs, eta_min=1e-6
        )

    def train_epoch(self, X, y, kg_embeddings=None):
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
            if kg_embeddings is not None:
                kg_batch = kg_embeddings[batch_idx]
                kg_tensor = torch.tensor(kg_batch, dtype=torch.float).to(self.device)
                out = self.model(X_tensor, kg_tensor, adj)
            else:
                out = self.model(X_tensor, adj)

            loss = F.cross_entropy(out, y_tensor, weight=self.class_weights)
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
    def evaluate(self, X, y, kg_embeddings=None):
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

            if kg_embeddings is not None:
                kg_batch = kg_embeddings[start:end]
                kg_tensor = torch.tensor(kg_batch, dtype=torch.float).to(self.device)
                out = self.model(X_tensor, kg_tensor, adj)
            else:
                out = self.model(X_tensor, adj)

            loss = F.cross_entropy(out, y_tensor, weight=self.class_weights)
            total_loss += loss.item() * len(y_batch)
            pred = out.argmax(dim=1)
            total_correct += (pred == y_tensor).sum().item()
            total_samples += len(y_batch)
            all_preds.extend(pred.cpu().numpy())

        return {
            'loss': total_loss / total_samples,
            'accuracy': total_correct / total_samples
        }, np.array(all_preds)

    def predict(self, X, kg_embeddings=None):
        self.model.eval()
        all_preds = []
        for start in range(0, len(X), self.batch_size):
            X_batch = X[start:start + self.batch_size]
            adj = build_batch_adjacency(X_batch, k=30).to(self.device)
            X_tensor = torch.tensor(X_batch, dtype=torch.float).to(self.device)

            if kg_embeddings is not None:
                kg_batch = kg_embeddings[start:start + self.batch_size]
                kg_tensor = torch.tensor(kg_batch, dtype=torch.float).to(self.device)
                out = self.model(X_tensor, kg_tensor, adj)
            else:
                out = self.model(X_tensor, adj)

            all_preds.extend(out.argmax(dim=1).cpu().numpy())
        return np.array(all_preds)


class GNNModel(_GNNBaseWrapper):
    """GNN模型包装类"""

    def __init__(self, config_path='config.yaml'):
        super().__init__(config_path, 'gnn')

    def build_model(self, n_features, n_classes):
        self.model = GNN1D(
            in_channels=n_features,
            num_classes=n_classes,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)
        self._init_optimizer()
        return self.model


class GNNKGModel(_GNNBaseWrapper):
    """GNN + KG融合模型包装类"""

    def __init__(self, config_path='config.yaml'):
        super().__init__(config_path, 'gnn')
        model_cfg = self.config['models'].get('gnn', {})
        self.kg_embedding_dim = model_cfg.get('kg_embedding_dim', 33)

    def build_model(self, n_features, n_classes):
        self.model = GNNKG1D(
            in_channels=n_features,
            kg_embedding_dim=self.kg_embedding_dim,
            num_classes=n_classes,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)
        self._init_optimizer()
        return self.model
