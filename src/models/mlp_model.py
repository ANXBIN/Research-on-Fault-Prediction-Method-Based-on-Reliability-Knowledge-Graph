#!/usr/bin/env python3
"""
MLP模型：普通MLP和知识图谱增强MLP
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.neighbors import NearestNeighbors
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

    def save_model(self, path):
        """保存模型"""
        config = {
            'hidden_dim': self.hidden_dim,
            'dropout': self.dropout,
            'learning_rate': self.learning_rate
        }
        if hasattr(self, 'kg_embedding_dim'):
            config['kg_embedding_dim'] = self.kg_embedding_dim
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'fault_to_idx': self.fault_to_idx,
            'config': config
        }, path)
        print(f"[INFO] 模型已保存至: {path}")

    def load_model(self, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.fault_to_idx = checkpoint['fault_to_idx']
        print(f"[INFO] 模型已从: {path} 加载")


class PlainMLP(nn.Module):
    """普通MLP模型"""

    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.3):
        super(PlainMLP, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, out_channels)
        )

    def forward(self, x):
        return self.net(x)


class KGEnhancedMLP(nn.Module):
    """知识图谱增强MLP模型"""

    def __init__(self, in_channels, kg_embedding_dim, hidden_channels,
                 out_channels, dropout=0.3):
        super(KGEnhancedMLP, self).__init__()

        # 原始特征投影 - 与PlainMLP相同结构
        self.feature_net = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # KG嵌入投影
        self.kg_net = nn.Sequential(
            nn.Linear(kg_embedding_dim, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 融合层
        fusion_in_dim = hidden_channels // 2
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in_dim * 2, fusion_in_dim),
            nn.LayerNorm(fusion_in_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(fusion_in_dim, hidden_channels // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 4, out_channels)
        )

    def forward(self, x, kg_embedding):
        x_feat = self.feature_net(x)
        x_kg = self.kg_net(kg_embedding)

        # 拼接融合
        combined = torch.cat([x_feat, x_kg], dim=1)
        x_fused = self.fusion(combined)

        return self.classifier(x_fused)


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

    def train_epoch(self, X, y):
        self.model.train()
        X_tensor = torch.tensor(X, dtype=torch.float).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.long).to(self.device)
        self.optimizer.zero_grad()
        out = self.model(X_tensor)
        loss = F.nll_loss(F.log_softmax(out, dim=1), y_tensor)
        loss.backward()
        self.optimizer.step()
        pred = out.argmax(dim=1)
        acc = (pred == y_tensor).sum().item() / len(y_tensor)
        return loss.item(), acc

    @torch.no_grad()
    def evaluate(self, X, y):
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.long).to(self.device)
        out = self.model(X_tensor)
        loss = F.nll_loss(F.log_softmax(out, dim=1), y_tensor)
        pred = out.argmax(dim=1)
        acc = (pred == y_tensor).sum().item() / len(y_tensor)
        return {'loss': loss.item(), 'accuracy': acc}, pred.cpu().numpy()

    def predict(self, X):
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float).to(self.device)
        out = self.model(X_tensor)
        return out.argmax(dim=1).cpu().numpy()


class KGEnhancedMLPModel(BaseModelWrapper):
    """知识图谱增强MLP模型包装类"""

    def __init__(self, config_path='config.yaml'):
        super().__init__(config_path, 'kg_enhanced_mlp')
        model_cfg = self.config['models'].get('kg_enhanced_mlp', {})
        self.kg_embedding_dim = model_cfg.get('kg_embedding_dim', 64)

    def build_model(self, n_features, n_classes):
        self.model = KGEnhancedMLP(
            in_channels=n_features,
            kg_embedding_dim=self.kg_embedding_dim,
            hidden_channels=self.hidden_dim,
            out_channels=n_classes,
            dropout=self.dropout
        ).to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate)
        return self.model

    def train_epoch(self, X, y, kg_embeddings):
        self.model.train()
        X_tensor = torch.tensor(X, dtype=torch.float).to(self.device)
        kg_tensor = torch.tensor(kg_embeddings, dtype=torch.float).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.long).to(self.device)
        self.optimizer.zero_grad()
        out = self.model(X_tensor, kg_tensor)
        loss = F.nll_loss(F.log_softmax(out, dim=1), y_tensor)
        loss.backward()
        self.optimizer.step()
        pred = out.argmax(dim=1)
        acc = (pred == y_tensor).sum().item() / len(y_tensor)
        return loss.item(), acc

    @torch.no_grad()
    def evaluate(self, X, y, kg_embeddings):
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float).to(self.device)
        kg_tensor = torch.tensor(kg_embeddings, dtype=torch.float).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.long).to(self.device)
        out = self.model(X_tensor, kg_tensor)
        loss = F.nll_loss(F.log_softmax(out, dim=1), y_tensor)
        pred = out.argmax(dim=1)
        acc = (pred == y_tensor).sum().item() / len(y_tensor)
        return {'loss': loss.item(), 'accuracy': acc}, pred.cpu().numpy()

    def predict(self, X, kg_embeddings):
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float).to(self.device)
        kg_tensor = torch.tensor(kg_embeddings, dtype=torch.float).to(self.device)
        out = self.model(X_tensor, kg_tensor)
        return out.argmax(dim=1).cpu().numpy()


class KGEnhancedMLP_V2(nn.Module):
    """知识图谱增强MLP V2 - 门控融合架构

    改进点（相比V1的简单拼接）：
    - 门控机制自适应调整特征和KG的权重
    - 更深的融合层 + 残差连接
    - LayerNorm稳定训练
    """

    def __init__(self, in_channels, kg_embedding_dim, hidden_channels,
                 out_channels, dropout=0.3):
        super().__init__()

        # 特征编码器
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

        # KG编码器
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

        # 门控机制 - 自适应权重
        self.gate = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.Sigmoid()
        )

        # 深层融合 (3*hidden: cnn, kg, gated)
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

        # 分类器
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

        # 门控融合
        concat = torch.cat([feat, kg], dim=1)
        gate = self.gate(concat)
        gated = gate * feat + (1 - gate) * kg

        # 拼接 + 深层融合
        fused = torch.cat([feat, kg, gated], dim=1)
        fused = self.fusion(fused)

        return self.classifier(fused)


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

    def train_epoch(self, X, y, kg_embeddings):
        self.model.train()
        X_tensor = torch.tensor(X, dtype=torch.float).to(self.device)
        kg_tensor = torch.tensor(kg_embeddings, dtype=torch.float).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.long).to(self.device)
        self.optimizer.zero_grad()
        out = self.model(X_tensor, kg_tensor)
        loss = F.nll_loss(F.log_softmax(out, dim=1), y_tensor)
        loss.backward()
        self.optimizer.step()
        pred = out.argmax(dim=1)
        acc = (pred == y_tensor).sum().item() / len(y_tensor)
        return loss.item(), acc

    @torch.no_grad()
    def evaluate(self, X, y, kg_embeddings):
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float).to(self.device)
        kg_tensor = torch.tensor(kg_embeddings, dtype=torch.float).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.long).to(self.device)
        out = self.model(X_tensor, kg_tensor)
        loss = F.nll_loss(F.log_softmax(out, dim=1), y_tensor)
        pred = out.argmax(dim=1)
        acc = (pred == y_tensor).sum().item() / len(y_tensor)
        return {'loss': loss.item(), 'accuracy': acc}, pred.cpu().numpy()

    def predict(self, X, kg_embeddings):
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float).to(self.device)
        kg_tensor = torch.tensor(kg_embeddings, dtype=torch.float).to(self.device)
        out = self.model(X_tensor, kg_tensor)
        return out.argmax(dim=1).cpu().numpy()


def build_knn_graph_features(X, k=10):
    """
    基于特征的K近邻图构建样本嵌入（已弃用，保留用于参考）
    """
    n_samples = len(X)
    n_features = X.shape[1]
    k = min(k, max(1, n_samples - 1))

    X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    row_norms = np.linalg.norm(X_norm, axis=1)
    zero_rows = row_norms < 1e-8
    if np.any(zero_rows):
        X_norm[zero_rows] = 1e-8

    nn_model = NearestNeighbors(n_neighbors=k + 1, metric='euclidean')
    nn_model.fit(X_norm)
    distances, indices = nn_model.kneighbors(X_norm)

    kg_embeddings = np.zeros((n_samples, 64), dtype=np.float32)

    for i in range(n_samples):
        neighbor_indices = indices[i, 1:]
        neighbor_features = X_norm[neighbor_indices]

        dists = np.maximum(distances[i, 1:], 1e-8)
        weights = 1.0 / dists
        weights = weights / weights.sum()

        neighbor_weighted = np.dot(weights, neighbor_features)

        kg_embeddings[i, :n_features] = X_norm[i]
        kg_embeddings[i, n_features:2*n_features] = neighbor_weighted

    return kg_embeddings


def load_kg_embeddings_mlp(X_train, X_val, X_test, k=20):
    """
    为MLP模型构建基于KNN的样本级嵌入

    每个样本的嵌入 = [归一化原始特征, 邻居特征加权和, 差异特征, 填充]
    KNN索引仅在训练集上构建，避免数据泄露。

    Args:
        X_train: 训练集特征
        X_val: 验证集特征
        X_test: 测试集特征
        k: 近邻数

    Returns:
        kg_train_emb, kg_val_emb, kg_test_emb: KNN嵌入 (64维)
    """
    n_features = X_train.shape[1]
    embedding_dim = 64

    # 仅在训练集上归一化并构建KNN索引
    X_train_norm = X_train / (np.linalg.norm(X_train, axis=1, keepdims=True) + 1e-8)
    k_actual = min(k + 1, len(X_train))  # +1 因为训练集查询会包含自己

    nn_model = NearestNeighbors(n_neighbors=k_actual, metric='euclidean')
    nn_model.fit(X_train_norm)

    def _build_embedding(X_query, exclude_self=False):
        """为查询样本构建KNN嵌入，邻居全部来自训练集"""
        X_query_norm = X_query / (np.linalg.norm(X_query, axis=1, keepdims=True) + 1e-8)
        distances, indices = nn_model.kneighbors(X_query_norm)

        n_query = len(X_query)
        kg_emb = np.zeros((n_query, embedding_dim), dtype=np.float32)

        for i in range(n_query):
            if exclude_self:
                # 训练集查询：第一个邻居可能是自己，跳过
                neighbor_indices = indices[i, 1:]
                neighbor_distances = distances[i, 1:]
            else:
                # 验证/测试集查询：直接使用所有k个邻居
                neighbor_indices = indices[i]
                neighbor_distances = distances[i]

            neighbor_features = X_train_norm[neighbor_indices]

            # 距离倒数加权
            weights = 1.0 / np.maximum(neighbor_distances, 1e-8)
            weights = weights / weights.sum()

            # 邻居特征加权和
            neighbor_weighted = np.dot(weights, neighbor_features)

            # 差异
            diff = X_query_norm[i] - neighbor_weighted

            # 拼接: [归一化特征(20), 邻居加权(20), 差异(20), 填充(4)]
            kg_emb[i, :n_features] = X_query_norm[i]
            kg_emb[i, n_features:2*n_features] = neighbor_weighted
            kg_emb[i, 2*n_features:3*n_features] = diff

        return kg_emb

    kg_train_emb = _build_embedding(X_train, exclude_self=True)
    kg_val_emb = _build_embedding(X_val, exclude_self=False)
    kg_test_emb = _build_embedding(X_test, exclude_self=False)

    return kg_train_emb, kg_val_emb, kg_test_emb


def load_kg_embeddings_v4(fault_emb_path, fault_labels, kg_embed_path=None):
    """
    基于故障类型的知识图谱嵌入 - V4版本

    根据每个样本的故障类型，从知识图谱中提取:
    1. 故障相似度向量 (该故障与其他所有故障的相似度)
    2. 故障-部件关联向量
    3. 故障-特征权重向量

    Args:
        fault_emb_path: fault_embeddings.json路径
        fault_labels: array of fault type labels (original string like "1ndBearing_ball")
        kg_embed_path: 可选，旧的kg_embeddings.json路径（用于获取全局统计）

    Returns:
        kg_embeddings: (n_samples, embedding_dim) 其中embedding_dim = 故障类型数 + 部件数 + 特征数 + 全局特征数
    """
    with open(fault_emb_path, 'r') as f:
        fault_emb_data = json.load(f)

    fault_types = fault_emb_data['fault_types']  # ['Ball_Fault', 'Broken_Tooth', ...]
    fault_to_idx = {f: i for i, f in enumerate(fault_types)}

    fault_similarity = np.array(fault_emb_data['fault_similarity'])  # (9, 9)
    fault_component_matrix = np.array(fault_emb_data['fault_component_matrix'])  # (9, n_comp)
    fault_feature_matrix = np.array(fault_emb_data['fault_feature_matrix'])  # (9, n_feat)

    # 合并全局KG统计信息
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
        except:
            global_features = None

    # 合并所有向量：故障相似度(9) + 部件(4) + 特征(14) + 全局(6) = 33维
    embedding_dim = 9 + fault_component_matrix.shape[1] + fault_feature_matrix.shape[1]
    if global_features is not None:
        embedding_dim += len(global_features)

    n_samples = len(fault_labels)
    kg_embeddings = np.zeros((n_samples, embedding_dim), dtype=np.float32)

    for i in range(n_samples):
        label = fault_labels[i]

        # 判断label是字符串还是整数
        if isinstance(label, str) or (isinstance(label, (int, np.integer)) and label >= len(fault_types)):
            # 可能是字符串故障名称如 "1ndBearing_ball"
            # 需要映射到标准故障类型
            kg_idx = None
            for ft_idx, ft in enumerate(fault_types):
                # 检查是否匹配
                if ft.lower() in label.lower() or label.lower() in ft.lower():
                    kg_idx = ft_idx
                    break
            if kg_idx is None:
                kg_idx = 0  # 默认
        else:
            kg_idx = int(label) if isinstance(label, (int, np.integer)) else 0

        # 提取该故障类型的嵌入
        sim_vec = fault_similarity[kg_idx]  # (9,)
        comp_vec = fault_component_matrix[kg_idx]  # (n_comp,)
        feat_vec = fault_feature_matrix[kg_idx]  # (n_feat,)

        # 拼接
        parts = [sim_vec, comp_vec, feat_vec]
        if global_features is not None:
            parts.append(global_features)

        kg_embeddings[i] = np.concatenate(parts)

    return kg_embeddings


def load_kg_embeddings_v3(kg_embed_path, n_samples, fault_labels, X_train, X_test=None, fault_types=None, fault_to_idx=None):
    """
    纯知识图谱嵌入 - 只使用KG的全局结构信息，不基于样本特征（已弃用，使用v4）
    """
    with open(kg_embed_path, 'r') as f:
        kg_data = json.load(f)

    structural_features = kg_data.get('structural_features', {})

    # 从KG获取全局结构特征
    node_counts = structural_features.get('node_counts', {})
    edge_counts = structural_features.get('edge_counts', {})

    # 全局KG特征 (6维)
    global_features = np.array([
        np.log1p(node_counts.get('Fault', 1)) / 20,
        np.log1p(node_counts.get('Component', 1)) / 5,
        np.log1p(node_counts.get('Feature', 1)) / 20,
        edge_counts.get('CAUSED_BY', 0) / 100,
        edge_counts.get('LOCATED_AT', 0) / 100,
        edge_counts.get('HAS_FEATURE', 0) / 500,
    ], dtype=np.float32)

    # 填充到64维: 前6维是KG全局特征，后58维用不同的随机投影填充
    np.random.seed(42)  # 固定种子保证可重复性
    random_projection = np.random.randn(6, 58).astype(np.float32) * 0.01

    # 最终嵌入 = 全局特征 @ 随机投影
    kg_final = np.zeros((n_samples, 64), dtype=np.float32)
    kg_final[:, :6] = global_features

    # 广播global_features到所有样本，然后通过随机投影
    for i in range(n_samples):
        kg_final[i, 6:] = np.dot(global_features, random_projection)

    return kg_final