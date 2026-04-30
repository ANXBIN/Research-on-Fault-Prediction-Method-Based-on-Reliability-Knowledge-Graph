#!/usr/bin/env python3
"""
MLP模型：普通MLP和知识图谱增强MLP
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import yaml
import json


class BaseModelWrapper:
    """模型包装器基类 - 提取公共逻辑"""

    def __init__(self, config_path='config.yaml', config_key='mlp'):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        model_cfg = self.config['models'].get(config_key, {})
        self.hidden_dim = model_cfg.get('hidden_dim', 128)
        self.dropout = model_cfg.get('dropout', 0.3)
        self.learning_rate = model_cfg.get('learning_rate', 0.001)
        self.epochs = model_cfg.get('epochs', 100)

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')

        self.model = None
        self.optimizer = None
        self.fault_to_idx = {}
        self.class_weights = None

    def set_class_weights(self, y_train):
        """根据训练集标签分布计算类别权重，用于平衡损失函数"""
        classes = np.unique(y_train)
        weights = compute_class_weight('balanced', classes=classes, y=y_train)
        self.class_weights = torch.tensor(weights, dtype=torch.float).to(self.device)

    def save_model(self, path):
        config = {
            'hidden_dim': self.hidden_dim,
            'dropout': self.dropout,
            'learning_rate': self.learning_rate
        }
        if hasattr(self, 'kg_embedding_dim'):
            config['kg_embedding_dim'] = self.kg_embedding_dim
        if hasattr(self, 'num_layers'):
            config['num_layers'] = self.num_layers
        if hasattr(self, 'batch_size'):
            config['batch_size'] = self.batch_size
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'fault_to_idx': self.fault_to_idx,
            'config': config
        }, path)
        print(f"[INFO] 模型已保存至: {path}")

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.fault_to_idx = checkpoint['fault_to_idx']
        print(f"[INFO] 模型已从: {path} 加载")

    def _forward(self, X_tensor, kg_tensor=None):
        """统一前向传播：自动处理有无KG嵌入的情况"""
        if kg_tensor is not None:
            return self.model(X_tensor, kg_tensor)
        return self.model(X_tensor)

    def train_epoch(self, X, y, kg_embeddings=None):
        """通用训练一个epoch，支持有无KG嵌入"""
        self.model.train()
        X_tensor = torch.tensor(X, dtype=torch.float).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.long).to(self.device)
        kg_tensor = torch.tensor(kg_embeddings, dtype=torch.float).to(self.device) if kg_embeddings is not None else None

        self.optimizer.zero_grad()
        out = self._forward(X_tensor, kg_tensor)
        loss = F.cross_entropy(out, y_tensor, weight=self.class_weights)
        loss.backward()
        self.optimizer.step()

        pred = out.argmax(dim=1)
        acc = (pred == y_tensor).sum().item() / len(y_tensor)
        return loss.item(), acc

    @torch.no_grad()
    def evaluate(self, X, y, kg_embeddings=None):
        """通用评估，支持有无KG嵌入"""
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.long).to(self.device)
        kg_tensor = torch.tensor(kg_embeddings, dtype=torch.float).to(self.device) if kg_embeddings is not None else None

        out = self._forward(X_tensor, kg_tensor)
        loss = F.cross_entropy(out, y_tensor, weight=self.class_weights)
        pred = out.argmax(dim=1)
        acc = (pred == y_tensor).sum().item() / len(y_tensor)
        return {'loss': loss.item(), 'accuracy': acc}, pred.cpu().numpy()

    def predict(self, X, kg_embeddings=None):
        """通用预测，支持有无KG嵌入"""
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float).to(self.device)
        kg_tensor = torch.tensor(kg_embeddings, dtype=torch.float).to(self.device) if kg_embeddings is not None else None
        out = self._forward(X_tensor, kg_tensor)
        return out.argmax(dim=1).cpu().numpy()


class PlainMLP(nn.Module):
    """普通MLP模型"""

    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.3):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.LayerNorm(hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, out_channels)
        )

    def forward(self, x):
        return self.net(x)


class KGEnhancedMLP_V2(nn.Module):
    """知识图谱增强MLP V2 - 门控融合架构"""

    def __init__(self, in_channels, kg_embedding_dim, hidden_channels,
                 out_channels, dropout=0.3):
        super().__init__()

        self.feature_net = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.kg_net = nn.Sequential(
            nn.Linear(kg_embedding_dim, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.gate = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.Sigmoid()
        )

        self.fusion = nn.Sequential(
            nn.Linear(hidden_channels * 3, hidden_channels * 2),
            nn.LayerNorm(hidden_channels * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.LayerNorm(hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, out_channels)
        )

    def forward(self, x, kg_embedding):
        feat = self.feature_net(x)
        kg = self.kg_net(kg_embedding)

        concat = torch.cat([feat, kg], dim=1)
        gate = self.gate(concat)
        gated = gate * feat + (1 - gate) * kg

        fused = torch.cat([feat, kg, gated], dim=1)
        fused = self.fusion(fused)
        return self.classifier(fused)


class MLPModel(BaseModelWrapper):
    """MLP模型包装类"""

    def __init__(self, config_path='config.yaml'):
        super().__init__(config_path, 'mlp')

    def build_model(self, n_features, n_classes):
        self.model = PlainMLP(
            in_channels=n_features,
            hidden_channels=self.hidden_dim,
            out_channels=n_classes,
            dropout=self.dropout
        ).to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate)
        return self.model


class KGEnhancedMLPV2Model(BaseModelWrapper):
    """知识图谱增强MLP V2包装类 - 门控融合"""

    def __init__(self, config_path='config.yaml'):
        super().__init__(config_path, 'kg_enhanced_mlp_v2')
        model_cfg = self.config['models'].get('kg_enhanced_mlp_v2', {})
        self.kg_embedding_dim = model_cfg.get('kg_embedding_dim', 64)

    def build_model(self, n_features, n_classes):
        self.model = KGEnhancedMLP_V2(
            in_channels=n_features,
            kg_embedding_dim=self.kg_embedding_dim,
            hidden_channels=self.hidden_dim,
            out_channels=n_classes,
            dropout=self.dropout
        ).to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate)
        return self.model


def load_kg_embeddings_mlp(X_train, X_val, X_test, k=20):
    """为MLP模型构建基于KNN的样本级嵌入（64维）"""
    n_features = X_train.shape[1]
    embedding_dim = 64

    X_train_norm = X_train / (np.linalg.norm(X_train, axis=1, keepdims=True) + 1e-8)
    k_actual = min(k + 1, len(X_train))

    nn_model = NearestNeighbors(n_neighbors=k_actual, metric='euclidean')
    nn_model.fit(X_train_norm)

    def _build_embedding(X_query, exclude_self=False):
        X_query_norm = X_query / (np.linalg.norm(X_query, axis=1, keepdims=True) + 1e-8)
        distances, indices = nn_model.kneighbors(X_query_norm)

        n_query = len(X_query)
        kg_emb = np.zeros((n_query, embedding_dim), dtype=np.float32)

        for i in range(n_query):
            if exclude_self:
                neighbor_indices = indices[i, 1:]
                neighbor_distances = distances[i, 1:]
            else:
                neighbor_indices = indices[i]
                neighbor_distances = distances[i]

            neighbor_features = X_train_norm[neighbor_indices]
            weights = 1.0 / np.maximum(neighbor_distances, 1e-8)
            weights = weights / weights.sum()
            neighbor_weighted = np.dot(weights, neighbor_features)
            diff = X_query_norm[i] - neighbor_weighted

            kg_emb[i, :n_features] = X_query_norm[i]
            kg_emb[i, n_features:2*n_features] = neighbor_weighted
            kg_emb[i, 2*n_features:3*n_features] = diff

        return kg_emb

    return (
        _build_embedding(X_train, exclude_self=True),
        _build_embedding(X_val, exclude_self=False),
        _build_embedding(X_test, exclude_self=False),
    )


def load_kg_embeddings_v4(fault_emb_path, fault_labels, kg_embed_path=None):
    """基于故障类型的知识图谱嵌入（33维）"""
    with open(fault_emb_path, 'r') as f:
        fault_emb_data = json.load(f)

    fault_types = fault_emb_data['fault_types']
    fault_similarity = np.array(fault_emb_data['fault_similarity'])
    fault_component_matrix = np.array(fault_emb_data['fault_component_matrix'])
    fault_feature_matrix = np.array(fault_emb_data['fault_feature_matrix'])

    global_features = None
    if kg_embed_path:
        try:
            with open(kg_embed_path, 'r') as f:
                kg_data = json.load(f)
            sf = kg_data.get('structural_features', {})
            node_counts = sf.get('node_counts', {})
            edge_counts = sf.get('edge_counts', {})
            global_features = np.array([
                np.log1p(node_counts.get('Fault', 1)) / 20,
                np.log1p(node_counts.get('Component', 1)) / 5,
                np.log1p(node_counts.get('Feature', 1)) / 20,
                edge_counts.get('CAUSED_BY', 0) / 100,
                edge_counts.get('LOCATED_AT', 0) / 100,
                edge_counts.get('HAS_FEATURE', 0) / 500,
            ], dtype=np.float32)
        except (FileNotFoundError, KeyError, json.JSONDecodeError):
            global_features = None

    embedding_dim = 9 + fault_component_matrix.shape[1] + fault_feature_matrix.shape[1]
    if global_features is not None:
        embedding_dim += len(global_features)

    n_samples = len(fault_labels)
    kg_embeddings = np.zeros((n_samples, embedding_dim), dtype=np.float32)

    for i in range(n_samples):
        label = fault_labels[i]

        if isinstance(label, str) or (isinstance(label, (int, np.integer)) and label >= len(fault_types)):
            kg_idx = None
            for ft_idx, ft in enumerate(fault_types):
                if ft.lower() in label.lower() or label.lower() in ft.lower():
                    kg_idx = ft_idx
                    break
            if kg_idx is None:
                kg_idx = 0
        else:
            kg_idx = int(label) if isinstance(label, (int, np.integer)) else 0

        parts = [
            fault_similarity[kg_idx],
            fault_component_matrix[kg_idx],
            fault_feature_matrix[kg_idx],
        ]
        if global_features is not None:
            parts.append(global_features)
        kg_embeddings[i] = np.concatenate(parts)

    return kg_embeddings


def load_kg_embeddings_v3(kg_embed_path, n_samples, fault_labels, X_train, X_test=None, fault_types=None, fault_to_idx=None):
    """纯知识图谱嵌入 - 只使用KG的全局结构信息（已弃用，使用v4）"""
    with open(kg_embed_path, 'r') as f:
        kg_data = json.load(f)

    structural_features = kg_data.get('structural_features', {})
    node_counts = structural_features.get('node_counts', {})
    edge_counts = structural_features.get('edge_counts', {})

    global_features = np.array([
        np.log1p(node_counts.get('Fault', 1)) / 20,
        np.log1p(node_counts.get('Component', 1)) / 5,
        np.log1p(node_counts.get('Feature', 1)) / 20,
        edge_counts.get('CAUSED_BY', 0) / 100,
        edge_counts.get('LOCATED_AT', 0) / 100,
        edge_counts.get('HAS_FEATURE', 0) / 500,
    ], dtype=np.float32)

    np.random.seed(42)
    random_projection = np.random.randn(6, 58).astype(np.float32) * 0.01

    kg_final = np.zeros((n_samples, 64), dtype=np.float32)
    kg_final[:, :6] = global_features
    for i in range(n_samples):
        kg_final[i, 6:] = np.dot(global_features, random_projection)

    return kg_final
