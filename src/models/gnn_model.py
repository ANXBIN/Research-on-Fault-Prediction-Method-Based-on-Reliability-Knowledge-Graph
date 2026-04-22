#!/usr/bin/env python3
"""
基于图神经网络(GNN)的故障预测模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Dataset
import numpy as np
import yaml


class FaultGNN(nn.Module):
    """故障预测GNN模型"""

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, dropout=0.3):
        super(FaultGNN, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout

        # 图卷积层
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # 第一层
        self.convs.append(GATConv(in_channels, hidden_channels, heads=4, concat=True))
        self.bns.append(nn.BatchNorm1d(hidden_channels * 4))

        # 中间层
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * 4, hidden_channels, heads=4, concat=True))
            self.bns.append(nn.BatchNorm1d(hidden_channels * 4))

        # 最后一层
        self.convs.append(GATConv(hidden_channels * 4, hidden_channels, heads=4, concat=True))
        self.bns.append(nn.BatchNorm1d(hidden_channels * 4))

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels * 4 * 2, hidden_channels * 2),  # 2 = mean + max pooling
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, x, edge_index, batch):
        """
        前向传播
        x: 节点特征 [num_nodes, in_channels]
        edge_index: 边索引 [2, num_edges]
        batch: 批次索引 [num_nodes]
        """
        # 图卷积层
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # 图级别池化
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)

        # 分类
        x = self.classifier(x)

        return F.log_softmax(x, dim=1)


class GNNModel:
    """GNN模型包装类"""

    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.model_config = self.config['models']['gnn']
        self.hidden_dim = self.model_config['hidden_dim']
        self.num_layers = self.model_config['num_layers']
        self.dropout = self.model_config['dropout']
        self.learning_rate = self.model_config['learning_rate']
        self.epochs = self.model_config['epochs']

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        self.model = None
        self.optimizer = None

    def build_model(self, in_channels, out_channels):
        """构建模型"""
        self.model = FaultGNN(
            in_channels=in_channels,
            hidden_channels=self.hidden_dim,
            out_channels=out_channels,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-4
        )

        return self.model

    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for data in train_loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()

            out = self.model(data.x, data.edge_index, data.batch)
            loss = F.nll_loss(out, data.y)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * data.num_graphs
            pred = out.argmax(dim=1)
            correct += (pred == data.y).sum().item()
            total += len(data.y)

        return total_loss / total, correct / total

    @torch.no_grad()
    def evaluate(self, loader):
        """评估模型"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        for data in loader:
            data = data.to(self.device)
            out = self.model(data.x, data.edge_index, data.batch)
            loss = F.nll_loss(out, data.y)

            total_loss += loss.item() * data.num_graphs
            pred = out.argmax(dim=1)
            correct += (pred == data.y).sum().item()
            total += len(data.y)

            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())

        return total_loss / total, correct / total, all_preds, all_labels

    def save_model(self, path):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.model_config
        }, path)
        print(f"[INFO] 模型已保存至: {path}")

    def load_model(self, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"[INFO] 模型已从: {path} 加载")


class FaultGraphDataset(Dataset):
    """故障图数据集"""

    def __init__(self, features_list, labels, fault_to_idx, adj_matrix=None, idx_to_fault=None):
        """
        features_list: 特征列表
        labels: 标签列表
        fault_to_idx: 故障类型到索引的映射
        adj_matrix: 邻接矩阵（可选）
        idx_to_fault: 索引到故障类型的映射（用于将整数标签转换为字符串）
        """
        self.features_list = features_list
        # 如果labels是整数，需要先转换为故障类型名称再转换为索引
        if idx_to_fault is not None:
            self.labels = [fault_to_idx[idx_to_fault[l]] for l in labels]
        else:
            self.labels = [fault_to_idx[l] if isinstance(l, str) else l for l in labels]
        self.fault_to_idx = fault_to_idx
        self.idx_to_fault = idx_to_fault
        self.adj_matrix = adj_matrix

    def __len__(self):
        return len(self.features_list)

    def __getitem__(self, idx):
        features = self.features_list[idx]
        label = self.labels[idx]

        # 构建图数据
        if self.adj_matrix is not None:
            # 使用知识图谱的邻接矩阵
            edge_index = self._build_edge_index_from_adj()
        else:
            # 构建简单KNN图
            edge_index = self._build_knn_graph(features)

        # 创建节点特征（复制n_nodes次以模拟图结构）
        n_nodes = max(edge_index.max().item() + 1, len(features))
        x = torch.tensor(features, dtype=torch.float).unsqueeze(0).repeat(n_nodes, 1)

        data = Data(
            x=x,
            edge_index=edge_index,
            y=torch.tensor([label], dtype=torch.long)
        )

        return data

    def _build_edge_index_from_adj(self):
        """从邻接矩阵构建边索引"""
        adj = self.adj_matrix
        edges = []

        for i in range(len(adj)):
            for j in range(i + 1, len(adj)):
                if adj[i][j] > 0:
                    edges.append([i, j])
                    edges.append([j, i])

        if not edges:
            # 如果没有边，创建自环
            edges = [[0, 0]]

        return torch.tensor(edges, dtype=torch.long).t().contiguous()

    def _build_knn_graph(self, features, k=5):
        """构建KNN图"""
        n = len(features)
        if n <= 1:
            return torch.tensor([[0, 0]], dtype=torch.long).t().contiguous()

        # 简化：只创建顺序边
        edges = []
        for i in range(n - 1):
            edges.append([i, i + 1])
            edges.append([i + 1, i])

        return torch.tensor(edges, dtype=torch.long).t().contiguous()


def create_graph_data_from_samples(features, labels, fault_to_idx, kg_adj_matrix=None):
    """从样本创建图数据"""
    dataset = FaultGraphDataset(features, labels, fault_to_idx, kg_adj_matrix)

    graph_list = []
    for i in range(len(dataset)):
        graph_list.append(dataset[i])

    return graph_list
