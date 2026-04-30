#!/usr/bin/env python3
"""
CNN模型：普通CNN和CNN+知识图谱融合
"""

import torch
import torch.nn as nn

from src.models.mlp_model import BaseModelWrapper


class CNN1D(nn.Module):
    """1维CNN模型 - 用于时序特征提取"""

    def __init__(self, in_channels, num_classes, hidden_channels=64, dropout=0.3):
        super().__init__()

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

        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(2)
        x = self.conv_layers(x)
        x = x.squeeze(2)
        return self.classifier(x)


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

    def forward(self, x):
        return x + self.block(x)


class CNNKG1D_V3(nn.Module):
    """CNN + KG融合模型 V3 - 残差连接融合"""

    def __init__(self, in_channels, kg_embedding_dim, num_classes,
                 hidden_channels=64, dropout=0.2):
        super().__init__()

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

        self.interact = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.Sigmoid()
        )

        self.residual_fusion = nn.Sequential(
            ResidualBlock(hidden_channels * 2, hidden_channels, dropout),
            ResidualBlock(hidden_channels * 2, hidden_channels, dropout),
        )

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
        x = x.unsqueeze(2)
        cnn_feat = self.conv_layers(x).squeeze(2)

        kg_feat = self.kg_net(kg_embedding)

        concat = torch.cat([cnn_feat, kg_feat], dim=1)
        interaction = self.interact(concat)

        cnn_inter = cnn_feat * interaction
        kg_inter = kg_feat * interaction

        fused = torch.cat([cnn_inter, kg_inter], dim=1)
        fused = self.residual_fusion(fused)
        fused = self.final_fusion(fused)

        return self.classifier(fused)


class CNNModel(BaseModelWrapper):
    """CNN模型包装类"""

    def __init__(self, config_path='config.yaml'):
        super().__init__(config_path, 'cnn')
        model_cfg = self.config['models'].get('cnn', {})
        self.hidden_dim = model_cfg.get('hidden_channels', 64)

    def build_model(self, n_features, n_classes):
        self.model = CNN1D(
            in_channels=n_features,
            num_classes=n_classes,
            hidden_channels=self.hidden_dim,
            dropout=self.dropout
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return self.model


class CNNKGModelV3(BaseModelWrapper):
    """CNN + KG融合模型 V3 包装类 - 残差连接"""

    def __init__(self, config_path='config.yaml'):
        super().__init__(config_path, 'cnn_kg_v3')
        model_cfg = self.config['models'].get('cnn_kg_v3', {})
        self.hidden_dim = model_cfg.get('hidden_channels', 64)
        self.kg_embedding_dim = model_cfg.get('kg_embedding_dim', 33)

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
