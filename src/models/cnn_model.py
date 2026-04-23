#!/usr/bin/env python3
"""
CNN模型：普通CNN和CNN+知识图谱融合
用于时序故障预测
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml


class CNN1D(nn.Module):
    """1维CNN模型 - 用于时序特征提取"""

    def __init__(self, in_channels, num_classes, hidden_channels=64, dropout=0.3):
        super().__init__()

        # 1D CNN特征提取 - 简化版，适合小输入维度
        # 输入: (batch, in_channels, 1) -> 视为1D信号
        self.conv_layers = nn.Sequential(
            # Conv1: in_channels -> 32
            nn.Conv1d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Conv2: 64 -> 128
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Conv3: 128 -> hidden_channels
            nn.Conv1d(128, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # 全局平均池化
        )

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, num_classes)
        )

    def forward(self, x):
        # x: (batch, in_channels)
        x = x.unsqueeze(2)  # -> (batch, in_channels, 1)
        x = self.conv_layers(x)  # -> (batch, hidden_channels, 1)
        x = x.squeeze(2)  # -> (batch, hidden_channels)
        return self.classifier(x)


class CNNKG1D(nn.Module):
    """CNN + 知识图谱融合模型"""

    def __init__(self, in_channels, kg_embedding_dim, num_classes,
                 hidden_channels=64, dropout=0.3):
        super().__init__()

        # CNN特征提取 - 简化版
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv1d(128, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        # KG嵌入编码器
        self.kg_net = nn.Sequential(
            nn.Linear(kg_embedding_dim, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, num_classes)
        )

    def forward(self, x, kg_embedding):
        # CNN特征
        x = x.unsqueeze(2)  # (batch, in_channels, 1)
        cnn_feat = self.conv_layers(x)  # (batch, hidden_channels, 1)
        cnn_feat = cnn_feat.squeeze(2)  # (batch, hidden_channels)

        # KG嵌入
        kg_feat = self.kg_net(kg_embedding)  # (batch, hidden_channels)

        # 融合
        fused = torch.cat([cnn_feat, kg_feat], dim=1)  # (batch, 2*hidden)
        fused = self.fusion(fused)  # (batch, hidden)

        return self.classifier(fused)


class CNNKG1D_V2(nn.Module):
    """CNN + KG融合模型 V2 - 门控融合

    改进：
    - 门控机制自适应调整CNN和KG特征的权重
    - 加性融合替代简单拼接
    - 更深的融合层
    """

    def __init__(self, in_channels, kg_embedding_dim, num_classes,
                 hidden_channels=64, dropout=0.3):
        super().__init__()

        # CNN特征提取
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv1d(128, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        # KG嵌入编码器
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

        # 融合层 - 输入是3个hidden_channels的拼接(cnn, kg, gated)
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
            nn.Linear(hidden_channels // 2, num_classes)
        )

    def forward(self, x, kg_embedding):
        # CNN特征
        x = x.unsqueeze(2)  # (batch, in_channels, 1)
        cnn_feat = self.conv_layers(x)  # (batch, hidden_channels, 1)
        cnn_feat = cnn_feat.squeeze(2)  # (batch, hidden_channels)

        # KG嵌入
        kg_feat = self.kg_net(kg_embedding)  # (batch, hidden_channels)

        # 门控融合
        concat = torch.cat([cnn_feat, kg_feat], dim=1)
        gate = self.gate(concat)  # (batch, hidden_channels)
        gated = gate * cnn_feat + (1 - gate) * kg_feat  # 自适应加权

        # 拼接+深层融合
        fused = torch.cat([cnn_feat, kg_feat, gated], dim=1)  # 3*hidden
        fused = self.fusion(fused)  # hidden

        return self.classifier(fused)


class ResidualBlock(nn.Module):
    """残差块 - 带跳跃连接的残差模块"""

    def __init__(self, in_dim, hidden_dim, dropout=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, in_dim),
            nn.LayerNorm(in_dim),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return x + self.block(x)  # 残差连接


class CNNKG1D_V3(nn.Module):
    """CNN + KG融合模型 V3 - 残差连接融合

    改进：
    - 双塔编码器：CNN和KG各自提取特征
    - 残差融合块：多层残差连接，保留原始信息
    - 特征交互：交互式学习CNN和KG的关系
    """

    def __init__(self, in_channels, kg_embedding_dim, num_classes,
                 hidden_channels=64, dropout=0.2):
        super().__init__()

        # CNN特征提取 - 更深的网络
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv1d(128, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        # KG嵌入编码器 - 更深的网络
        self.kg_net = nn.Sequential(
            nn.Linear(kg_embedding_dim, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
        )

        # 交互层 - 学习CNN和KG的交互
        self.interact = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.Sigmoid()
        )

        # 残差融合块
        self.residual_fusion = nn.Sequential(
            ResidualBlock(hidden_channels * 2, hidden_channels, dropout),
            ResidualBlock(hidden_channels * 2, hidden_channels, dropout),
        )

        # 最终融合 + 分类器
        self.final_fusion = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.LayerNorm(hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, num_classes)
        )

    def forward(self, x, kg_embedding):
        # CNN特征
        x = x.unsqueeze(2)  # (batch, in_channels, 1)
        cnn_feat = self.conv_layers(x)  # (batch, hidden_channels, 1)
        cnn_feat = cnn_feat.squeeze(2)  # (batch, hidden_channels)

        # KG嵌入
        kg_feat = self.kg_net(kg_embedding)  # (batch, hidden_channels)

        # 交互：学习CNN和KG的关系
        concat = torch.cat([cnn_feat, kg_feat], dim=1)
        interaction = self.interact(concat)  # (batch, hidden_channels)

        # 交互后的特征
        cnn_inter = cnn_feat * interaction
        kg_inter = kg_feat * interaction

        # 残差融合
        fused = torch.cat([cnn_inter, kg_inter], dim=1)  # 2*hidden
        fused = self.residual_fusion(fused)  # 2*hidden, 保留残差

        # 最终融合
        fused = self.final_fusion(fused)  # hidden

        return self.classifier(fused)


class CNNKGModelV2:
    """CNN + KG融合模型 V2 包装类"""

    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        model_cfg = self.config['models'].get('cnn_kg_v2', {})
        self.hidden_dim = model_cfg.get('hidden_channels', 64)
        self.kg_embedding_dim = model_cfg.get('kg_embedding_dim', 33)
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

    def build_model(self, n_features, n_classes):
        self.model = CNNKG1D_V2(
            in_channels=n_features,
            kg_embedding_dim=self.kg_embedding_dim,
            num_classes=n_classes,
            hidden_channels=self.hidden_dim,
            dropout=self.dropout
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
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

    def save_model(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'fault_to_idx': self.fault_to_idx,
            'config': {
                'hidden_dim': self.hidden_dim,
                'kg_embedding_dim': self.kg_embedding_dim,
                'dropout': self.dropout,
                'learning_rate': self.learning_rate
            }
        }, path)
        print(f"[INFO] 模型已保存至: {path}")

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.fault_to_idx = checkpoint['fault_to_idx']
        print(f"[INFO] 模型已从: {path} 加载")


class CNNKGModelV3:
    """CNN + KG融合模型 V3 包装类 - 残差连接"""

    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        model_cfg = self.config['models'].get('cnn_kg_v3', {})
        self.hidden_dim = model_cfg.get('hidden_channels', 64)
        self.kg_embedding_dim = model_cfg.get('kg_embedding_dim', 33)
        self.dropout = model_cfg.get('dropout', 0.2)
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

    def build_model(self, n_features, n_classes):
        self.model = CNNKG1D_V3(
            in_channels=n_features,
            kg_embedding_dim=self.kg_embedding_dim,
            num_classes=n_classes,
            hidden_channels=self.hidden_dim,
            dropout=self.dropout
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
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

    def save_model(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'fault_to_idx': self.fault_to_idx,
            'config': {
                'hidden_dim': self.hidden_dim,
                'kg_embedding_dim': self.kg_embedding_dim,
                'dropout': self.dropout,
                'learning_rate': self.learning_rate
            }
        }, path)
        print(f"[INFO] 模型已保存至: {path}")

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.fault_to_idx = checkpoint['fault_to_idx']
        print(f"[INFO] 模型已从: {path} 加载")


class CNNModel:
    """CNN模型包装类"""

    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        model_cfg = self.config['models'].get('cnn', {})
        self.hidden_dim = model_cfg.get('hidden_channels', 64)
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

    def build_model(self, n_features, n_classes):
        self.model = CNN1D(
            in_channels=n_features,
            num_classes=n_classes,
            hidden_channels=self.hidden_dim,
            dropout=self.dropout
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
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

    def save_model(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'fault_to_idx': self.fault_to_idx,
            'config': {
                'hidden_dim': self.hidden_dim,
                'dropout': self.dropout,
                'learning_rate': self.learning_rate
            }
        }, path)
        print(f"[INFO] 模型已保存至: {path}")

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.fault_to_idx = checkpoint['fault_to_idx']
        print(f"[INFO] 模型已从: {path} 加载")


class CNNKGModel:
    """CNN + KG融合模型包装类"""

    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        model_cfg = self.config['models'].get('cnn_kg', {})
        self.hidden_dim = model_cfg.get('hidden_channels', 64)
        self.kg_embedding_dim = model_cfg.get('kg_embedding_dim', 33)
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

    def build_model(self, n_features, n_classes):
        self.model = CNNKG1D(
            in_channels=n_features,
            kg_embedding_dim=self.kg_embedding_dim,
            num_classes=n_classes,
            hidden_channels=self.hidden_dim,
            dropout=self.dropout
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
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

    def save_model(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'fault_to_idx': self.fault_to_idx,
            'config': {
                'hidden_dim': self.hidden_dim,
                'kg_embedding_dim': self.kg_embedding_dim,
                'dropout': self.dropout,
                'learning_rate': self.learning_rate
            }
        }, path)
        print(f"[INFO] 模型已保存至: {path}")

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.fault_to_idx = checkpoint['fault_to_idx']
        print(f"[INFO] 模型已从: {path} 加载")
