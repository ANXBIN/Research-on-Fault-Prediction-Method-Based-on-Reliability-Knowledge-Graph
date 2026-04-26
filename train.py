#!/usr/bin/env python3
"""
统一训练脚本 - 训练MLP和KG增强MLP模型
支持选择性训练：--mlp, --kg-v1, --kg-v2
支持V2模型调优：--tune-v2
显示训练进度
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import torch
import json
import yaml
import argparse
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from src.models.mlp_model import MLPModel, KGEnhancedMLPModel, KGEnhancedMLPV2Model, load_kg_embeddings_v3, load_kg_embeddings_v4, load_kg_embeddings_mlp
from src.models.cnn_model import CNNModel, CNNKGModel, CNNKGModelV2, CNNKGModelV3
from src.models.gnn_model import GNNModel, GNNKGModel, GNNKGModelV2, GATModel, GATKGModel, GATKGModelV2


class EarlyStopping:
    """早停机制 - 验证集loss不再下降时停止训练"""

    def __init__(self, patience=30, min_delta=1e-4):
        """
        Args:
            patience: 容忍多少个epoch没有改善
            min_delta: 最小改善量
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.best_state = None
        self.should_stop = False

    def step(self, val_loss, model_state):
        """
        检查是否应该停止训练

        Args:
            val_loss: 当前验证集loss
            model_state: 当前模型state_dict

        Returns:
            True if should stop
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_state = {k: v.clone() for k, v in model_state.items()}
            return False

        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_state = {k: v.clone() for k, v in model_state.items()}
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                return True
        return False

    def restore_best(self, model):
        """恢复最佳模型状态"""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


class Trainer:
    def __init__(self):
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        print(f"[INFO] 使用设备: {self.device}")

        # 创建目录
        Path('models').mkdir(exist_ok=True)
        Path('results').mkdir(exist_ok=True)

        # 加载配置
        with open('config.yaml', 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        self.es_patience = self.config.get('training', {}).get('early_stopping_patience', 30)

        # 加载数据
        self.load_data()

        # 贝叶斯优化得出的最佳配置
        self.best_config = {
            'hidden_dim': 128,
            'dropout': 0.215,
            'lr': 0.0064,
            'weight_decay': 0.00021
        }

    def load_data(self):
        """加载和预处理数据"""
        print("\n" + "=" * 60)
        print("加载数据")
        print("=" * 60)

        df = pd.read_csv('data/processed/processed_features.csv')
        self.fault_types = df['fault_type'].unique()
        print(f"数据集大小: {len(df)} 样本")
        print(f"故障类型数: {len(self.fault_types)}")

        # 标签编码
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.fault_types)
        y = self.label_encoder.transform(df['fault_type'])
        self.fault_to_idx = dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))

        # 特征
        feature_cols = [col for col in df.columns if col not in ['fault_type', 'channel']]
        X = df[feature_cols].values

        # 标准化
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # 划分数据集
        indices = np.arange(len(df))
        train_idx, test_idx = train_test_split(
            indices, test_size=0.2, random_state=42, stratify=y
        )
        train_idx, val_idx = train_test_split(
            train_idx, test_size=0.2, random_state=42, stratify=y[train_idx]
        )

        self.X_train = X_scaled[train_idx]
        self.X_val = X_scaled[val_idx]
        self.X_test = X_scaled[test_idx]
        self.y_train = y[train_idx]
        self.y_val = y[val_idx]
        self.y_test = y[test_idx]

        # 获取划分后的故障类型标签（字符串）
        fault_labels_all = df['fault_type'].values
        fault_labels_train = fault_labels_all[train_idx]
        fault_labels_val = fault_labels_all[val_idx]
        fault_labels_test = fault_labels_all[test_idx]

        # V1嵌入：使用旧的全局KG嵌入（64维，所有样本共享相同嵌入）
        print("加载V1用全局KG嵌入...")
        self.kg_train_emb_v1 = load_kg_embeddings_v3(
            'data/processed/kg_embeddings.json',
            len(self.X_train), None, self.X_train
        )
        self.kg_val_emb_v1 = load_kg_embeddings_v3(
            'data/processed/kg_embeddings.json',
            len(self.X_val), None, self.X_train, self.X_val
        )
        self.kg_test_emb_v1 = load_kg_embeddings_v3(
            'data/processed/kg_embeddings.json',
            len(self.X_test), None, self.X_train, self.X_test
        )

        # V2嵌入：使用故障级别KG嵌入（33维，每个故障类型不同）
        print("加载V2用故障级别KG嵌入（33维）...")
        self.kg_train_emb = load_kg_embeddings_v4(
            'data/processed/fault_embeddings.json',
            fault_labels_train,
            'data/processed/kg_embeddings.json'
        )
        self.kg_val_emb = load_kg_embeddings_v4(
            'data/processed/fault_embeddings.json',
            fault_labels_val,
            'data/processed/kg_embeddings.json'
        )
        self.kg_test_emb = load_kg_embeddings_v4(
            'data/processed/fault_embeddings.json',
            fault_labels_test,
            'data/processed/kg_embeddings.json'
        )

        # MLP专用嵌入：基于KNN的样本级嵌入
        print("加载MLP专用KNN嵌入（64维）...")
        self.kg_train_emb_mlp, self.kg_val_emb_mlp, self.kg_test_emb_mlp = load_kg_embeddings_mlp(
            self.X_train, self.X_val, self.X_test, k=20
        )

        print(f"训练集: {len(self.X_train)} 样本")
        print(f"验证集: {len(self.X_val)} 样本")
        print(f"测试集: {len(self.X_test)} 样本")
        print(f"V1 KG嵌入维度: {self.kg_train_emb_v1.shape}")
        print(f"V2 KG嵌入维度: {self.kg_train_emb.shape}")
        print(f"MLP KNN嵌入维度: {self.kg_train_emb_mlp.shape}")

    def train_mlp(self, epochs=100, use_early_stopping=True):
        """训练普通MLP"""
        print("\n" + "=" * 60)
        print("训练 MLP 模型 (传统方法)")
        print("=" * 60)

        model = MLPModel(config_path='config.yaml')
        model.fault_to_idx = self.fault_to_idx
        model.build_model(self.X_train.shape[1], len(self.fault_types))

        early_stop = EarlyStopping(patience=self.es_patience) if use_early_stopping else None

        pbar = tqdm(range(epochs), desc="MLP Training")
        for epoch in pbar:
            train_loss, train_acc = model.train_epoch(self.X_train, self.y_train)
            val_metrics, _ = model.evaluate(self.X_val, self.y_val)
            pbar.set_postfix({'loss': f'{train_loss:.4f}', 'acc': f'{train_acc:.4f}', 'val': f'{val_metrics["accuracy"]:.4f}'})

            if early_stop and early_stop.step(val_metrics['loss'], model.model.state_dict()):
                print(f"  早停触发 (epoch {epoch+1})，恢复最佳模型")
                early_stop.restore_best(model.model)
                break

        val_metrics, _ = model.evaluate(self.X_val, self.y_val)
        print(f"MLP 验证集准确率: {val_metrics['accuracy']:.4f}")
        model.save_model('models/mlp_model.pt')

        return model, val_metrics

    def train_kg_mlp_v1(self, config=None, epochs=100, verbose=True, use_early_stopping=True):
        """训练MLP-KG模型 - 使用KNN嵌入"""
        if verbose:
            print("\n" + "=" * 60)
            print("训练 MLP-KG 模型 (KNN嵌入融合)")
            print("=" * 60)

        if config is None:
            config = self.best_config

        model = KGEnhancedMLPModel(config_path='config.yaml')
        model.hidden_dim = config['hidden_dim']
        model.dropout = config['dropout']
        model.learning_rate = config['lr']
        model.fault_to_idx = self.fault_to_idx
        model.build_model(self.X_train.shape[1], len(self.fault_types))

        model.optimizer = torch.optim.Adam(
            model.model.parameters(),
            lr=config['lr'],
            weight_decay=config['weight_decay']
        )

        early_stop = EarlyStopping(patience=self.es_patience) if use_early_stopping and verbose else None

        if verbose:
            pbar = tqdm(range(epochs), desc="MLP-KG Training")
        else:
            pbar = range(epochs)

        for epoch in pbar:
            train_loss, train_acc = model.train_epoch(self.X_train, self.y_train, self.kg_train_emb_mlp)
            if verbose:
                pbar.set_postfix({'loss': f'{train_loss:.4f}', 'acc': f'{train_acc:.4f}'})

            if early_stop:
                val_m, _ = model.evaluate(self.X_val, self.y_val, self.kg_val_emb_mlp)
                if early_stop.step(val_m['loss'], model.model.state_dict()):
                    print(f"  早停触发 (epoch {epoch+1})，恢复最佳模型")
                    early_stop.restore_best(model.model)
                    break

        val_metrics, _ = model.evaluate(self.X_val, self.y_val, self.kg_val_emb_mlp)

        if verbose:
            print(f"MLP-KG 验证集准确率: {val_metrics['accuracy']:.4f}")
            model.save_model('models/mlp_kg_model.pt')

        return model, val_metrics

    def train_kg_mlp_v2(self, config=None, epochs=100, verbose=True, use_early_stopping=True):
        """训练MLP-KG-V2模型 - 门控融合架构

        Args:
            config: 超参数字典，None则使用best_config
            epochs: 训练轮数
            verbose: 是否打印详细信息
            use_early_stopping: 是否使用早停
        """
        if config is None:
            config = self.best_config

        if verbose:
            print("\n" + "=" * 60)
            print("训练 MLP-KG-V2 模型 (门控融合)")
            print("=" * 60)
            print(f"配置: hidden_dim={config['hidden_dim']}, dropout={config['dropout']:.3f}, lr={config['lr']:.6f}")

        model = KGEnhancedMLPV2Model(config_path='config.yaml')
        model.hidden_dim = config['hidden_dim']
        model.dropout = config['dropout']
        model.learning_rate = config['lr']
        model.kg_embedding_dim = 64  # MLP专用64维KNN嵌入
        model.fault_to_idx = self.fault_to_idx
        model.build_model(self.X_train.shape[1], len(self.fault_types))

        model.optimizer = torch.optim.Adam(
            model.model.parameters(),
            lr=config['lr'],
            weight_decay=config['weight_decay']
        )

        early_stop = EarlyStopping(patience=self.es_patience) if use_early_stopping and verbose else None

        if verbose:
            pbar = tqdm(range(epochs), desc="MLP-KG-V2 Training")
        else:
            pbar = range(epochs)

        for epoch in pbar:
            train_loss, train_acc = model.train_epoch(self.X_train, self.y_train, self.kg_train_emb_mlp)
            if verbose:
                pbar.set_postfix({'loss': f'{train_loss:.4f}', 'acc': f'{train_acc:.4f}'})

            if early_stop:
                val_m, _ = model.evaluate(self.X_val, self.y_val, self.kg_val_emb_mlp)
                if early_stop.step(val_m['loss'], model.model.state_dict()):
                    print(f"  早停触发 (epoch {epoch+1})，恢复最佳模型")
                    early_stop.restore_best(model.model)
                    break

        val_metrics, _ = model.evaluate(self.X_val, self.y_val, self.kg_val_emb_mlp)

        if verbose:
            print(f"MLP-KG-V2 验证集准确率: {val_metrics['accuracy']:.4f}")
            model.save_model('models/mlp_kg_v2_model.pt')

        return model, val_metrics

    def train_cnn(self, epochs=100, verbose=True, use_early_stopping=True):
        """训练CNN模型"""
        if verbose:
            print("\n" + "=" * 60)
            print("训练 CNN 模型 (1D卷积神经网络)")
            print("=" * 60)

        model = CNNModel(config_path='config.yaml')
        model.fault_to_idx = self.fault_to_idx
        model.build_model(self.X_train.shape[1], len(self.fault_types))

        early_stop = EarlyStopping(patience=self.es_patience) if use_early_stopping and verbose else None

        if verbose:
            pbar = tqdm(range(epochs), desc="CNN Training")
        else:
            pbar = range(epochs)

        for epoch in pbar:
            train_loss, train_acc = model.train_epoch(self.X_train, self.y_train)
            if verbose:
                pbar.set_postfix({'loss': f'{train_loss:.4f}', 'acc': f'{train_acc:.4f}'})

            if early_stop:
                val_m, _ = model.evaluate(self.X_val, self.y_val)
                if early_stop.step(val_m['loss'], model.model.state_dict()):
                    print(f"  早停触发 (epoch {epoch+1})，恢复最佳模型")
                    early_stop.restore_best(model.model)
                    break

        val_metrics, _ = model.evaluate(self.X_val, self.y_val)

        if verbose:
            print(f"CNN 验证集准确率: {val_metrics['accuracy']:.4f}")
        model.save_model('models/cnn_model.pt')

        return model, val_metrics

    def train_cnn_kg(self, epochs=100, verbose=True, use_early_stopping=True):
        """训练CNN + KG融合模型"""
        if verbose:
            print("\n" + "=" * 60)
            print("训练 CNN-KG 融合模型")
            print("=" * 60)

        model = CNNKGModel(config_path='config.yaml')
        model.fault_to_idx = self.fault_to_idx
        model.build_model(self.X_train.shape[1], len(self.fault_types))

        early_stop = EarlyStopping(patience=self.es_patience) if use_early_stopping and verbose else None

        if verbose:
            pbar = tqdm(range(epochs), desc="CNN-KG Training")
        else:
            pbar = range(epochs)

        for epoch in pbar:
            train_loss, train_acc = model.train_epoch(self.X_train, self.y_train, self.kg_train_emb)
            if verbose:
                pbar.set_postfix({'loss': f'{train_loss:.4f}', 'acc': f'{train_acc:.4f}'})

            if early_stop:
                val_m, _ = model.evaluate(self.X_val, self.y_val, self.kg_val_emb)
                if early_stop.step(val_m['loss'], model.model.state_dict()):
                    print(f"  早停触发 (epoch {epoch+1})，恢复最佳模型")
                    early_stop.restore_best(model.model)
                    break

        val_metrics, _ = model.evaluate(self.X_val, self.y_val, self.kg_val_emb)

        if verbose:
            print(f"CNN-KG 验证集准确率: {val_metrics['accuracy']:.4f}")
        model.save_model('models/cnn_kg_model.pt')

        return model, val_metrics

    def train_cnn_kg_v2(self, epochs=100, verbose=True, use_early_stopping=True):
        """训练CNN + KG融合模型 V2 (门控融合)"""
        if verbose:
            print("\n" + "=" * 60)
            print("训练 CNN-KG V2 融合模型 (门控融合)")
            print("=" * 60)

        model = CNNKGModelV2(config_path='config.yaml')
        model.fault_to_idx = self.fault_to_idx
        model.build_model(self.X_train.shape[1], len(self.fault_types))

        early_stop = EarlyStopping(patience=self.es_patience) if use_early_stopping and verbose else None

        if verbose:
            pbar = tqdm(range(epochs), desc="CNN-KG V2 Training")
        else:
            pbar = range(epochs)

        for epoch in pbar:
            train_loss, train_acc = model.train_epoch(self.X_train, self.y_train, self.kg_train_emb)
            if verbose:
                pbar.set_postfix({'loss': f'{train_loss:.4f}', 'acc': f'{train_acc:.4f}'})

            if early_stop:
                val_m, _ = model.evaluate(self.X_val, self.y_val, self.kg_val_emb)
                if early_stop.step(val_m['loss'], model.model.state_dict()):
                    print(f"  早停触发 (epoch {epoch+1})，恢复最佳模型")
                    early_stop.restore_best(model.model)
                    break

        val_metrics, _ = model.evaluate(self.X_val, self.y_val, self.kg_val_emb)

        if verbose:
            print(f"CNN-KG V2 验证集准确率: {val_metrics['accuracy']:.4f}")
        model.save_model('models/cnn_kg_v2_model.pt')

        return model, val_metrics

    def train_cnn_kg_v3(self, epochs=100, verbose=True, use_early_stopping=True):
        """训练CNN + KG融合模型 V3 (残差连接)"""
        if verbose:
            print("\n" + "=" * 60)
            print("训练 CNN-KG V3 融合模型 (残差连接)")
            print("=" * 60)

        model = CNNKGModelV3(config_path='config.yaml')
        model.fault_to_idx = self.fault_to_idx
        model.build_model(self.X_train.shape[1], len(self.fault_types))

        early_stop = EarlyStopping(patience=self.es_patience) if use_early_stopping and verbose else None

        if verbose:
            pbar = tqdm(range(epochs), desc="CNN-KG V3 Training")
        else:
            pbar = range(epochs)

        for epoch in pbar:
            train_loss, train_acc = model.train_epoch(self.X_train, self.y_train, self.kg_train_emb)
            if verbose:
                pbar.set_postfix({'loss': f'{train_loss:.4f}', 'acc': f'{train_acc:.4f}'})

            if early_stop:
                val_m, _ = model.evaluate(self.X_val, self.y_val, self.kg_val_emb)
                if early_stop.step(val_m['loss'], model.model.state_dict()):
                    print(f"  早停触发 (epoch {epoch+1})，恢复最佳模型")
                    early_stop.restore_best(model.model)
                    break

        val_metrics, _ = model.evaluate(self.X_val, self.y_val, self.kg_val_emb)

        if verbose:
            print(f"CNN-KG V3 验证集准确率: {val_metrics['accuracy']:.4f}")
        model.save_model('models/cnn_kg_v3_model.pt')

        return model, val_metrics

    def train_gnn(self, epochs=100, verbose=True, use_early_stopping=True):
        """训练GNN模型 (使用V2改进版)"""
        if verbose:
            print("\n" + "=" * 60)
            print("训练 GNN 模型 (图神经网络)")
            print("=" * 60)

        model = GNNModel(config_path='config.yaml')
        model.fault_to_idx = self.fault_to_idx
        model.build_model(self.X_train.shape[1], len(self.fault_types))

        early_stop = EarlyStopping(patience=self.es_patience) if use_early_stopping and verbose else None

        if verbose:
            pbar = tqdm(range(epochs), desc="GNN Training")
        else:
            pbar = range(epochs)

        for epoch in pbar:
            train_loss, train_acc = model.train_epoch(self.X_train, self.y_train)
            if verbose:
                pbar.set_postfix({'loss': f'{train_loss:.4f}', 'acc': f'{train_acc:.4f}'})

            if early_stop:
                val_m, _ = model.evaluate(self.X_val, self.y_val)
                if early_stop.step(val_m['loss'], model.model.state_dict()):
                    print(f"  早停触发 (epoch {epoch+1})，恢复最佳模型")
                    early_stop.restore_best(model.model)
                    break

        val_metrics, _ = model.evaluate(self.X_val, self.y_val)

        if verbose:
            print(f"GNN 验证集准确率: {val_metrics['accuracy']:.4f}")
        model.save_model('models/gnn_model.pt')

        return model, val_metrics

    def train_gnn_kg(self, epochs=100, verbose=True, use_early_stopping=True):
        """训练GNN + KG融合模型 (使用V2改进版)"""
        if verbose:
            print("\n" + "=" * 60)
            print("训练 GNN-KG 融合模型")
            print("=" * 60)

        model = GNNKGModel(config_path='config.yaml')
        model.fault_to_idx = self.fault_to_idx
        model.build_model(self.X_train.shape[1], len(self.fault_types))

        early_stop = EarlyStopping(patience=self.es_patience) if use_early_stopping and verbose else None

        if verbose:
            pbar = tqdm(range(epochs), desc="GNN-KG Training")
        else:
            pbar = range(epochs)

        for epoch in pbar:
            train_loss, train_acc = model.train_epoch(self.X_train, self.y_train, self.kg_train_emb)
            if verbose:
                pbar.set_postfix({'loss': f'{train_loss:.4f}', 'acc': f'{train_acc:.4f}'})

            if early_stop:
                val_m, _ = model.evaluate(self.X_val, self.y_val, self.kg_val_emb)
                if early_stop.step(val_m['loss'], model.model.state_dict()):
                    print(f"  早停触发 (epoch {epoch+1})，恢复最佳模型")
                    early_stop.restore_best(model.model)
                    break

        val_metrics, _ = model.evaluate(self.X_val, self.y_val, self.kg_val_emb)

        if verbose:
            print(f"GNN-KG 验证集准确率: {val_metrics['accuracy']:.4f}")
        model.save_model('models/gnn_kg_model.pt')

        return model, val_metrics

    def train_gnn_kg_v2(self, epochs=100, verbose=True, use_early_stopping=True):
        """训练GNN-KG V2：交叉注意力 + 端到端可学习KG嵌入"""
        if verbose:
            print("\n" + "=" * 60)
            print("训练 GNN-KG V2 模型 (交叉注意力+可学习嵌入)")
            print("=" * 60)

        model = GNNKGModelV2(config_path='config.yaml')
        model.fault_to_idx = self.fault_to_idx
        model.build_model(self.X_train.shape[1], len(self.fault_types))

        early_stop = EarlyStopping(patience=self.es_patience) if use_early_stopping and verbose else None

        if verbose:
            pbar = tqdm(range(epochs), desc="GNN-KG-V2 Training")
        else:
            pbar = range(epochs)

        for epoch in pbar:
            train_loss, train_acc = model.train_epoch(self.X_train, self.y_train, self.kg_train_emb)
            if verbose:
                pbar.set_postfix({'loss': f'{train_loss:.4f}', 'acc': f'{train_acc:.4f}'})

            if early_stop:
                val_m, _ = model.evaluate(self.X_val, self.y_val, self.kg_val_emb)
                if early_stop.step(val_m['loss'], model.model.state_dict()):
                    print(f"  早停触发 (epoch {epoch+1})，恢复最佳模型")
                    early_stop.restore_best(model.model)
                    break

        val_metrics, _ = model.evaluate(self.X_val, self.y_val, self.kg_val_emb)

        if verbose:
            print(f"GNN-KG-V2 验证集准确率: {val_metrics['accuracy']:.4f}")
        model.save_model('models/gnn_kg_v2_model.pt')

        return model, val_metrics

    def train_gat(self, epochs=100, verbose=True, use_early_stopping=True):
        """训练GAT模型 (图注意力网络)"""
        if verbose:
            print("\n" + "=" * 60)
            print("训练 GAT 模型 (图注意力网络)")
            print("=" * 60)

        model = GATModel(config_path='config.yaml')
        model.fault_to_idx = self.fault_to_idx
        model.build_model(self.X_train.shape[1], len(self.fault_types))

        early_stop = EarlyStopping(patience=self.es_patience) if use_early_stopping and verbose else None

        if verbose:
            pbar = tqdm(range(epochs), desc="GAT Training")
        else:
            pbar = range(epochs)

        for epoch in pbar:
            train_loss, train_acc = model.train_epoch(self.X_train, self.y_train)
            if verbose:
                pbar.set_postfix({'loss': f'{train_loss:.4f}', 'acc': f'{train_acc:.4f}'})

            if early_stop:
                val_m, _ = model.evaluate(self.X_val, self.y_val)
                if early_stop.step(val_m['loss'], model.model.state_dict()):
                    print(f"  早停触发 (epoch {epoch+1})，恢复最佳模型")
                    early_stop.restore_best(model.model)
                    break

        val_metrics, _ = model.evaluate(self.X_val, self.y_val)

        if verbose:
            print(f"GAT 验证集准确率: {val_metrics['accuracy']:.4f}")
        model.save_model('models/gat_model.pt')

        return model, val_metrics

    def train_gat_kg(self, epochs=100, verbose=True, use_early_stopping=True):
        """训练GAT + KG融合模型"""
        if verbose:
            print("\n" + "=" * 60)
            print("训练 GAT-KG 融合模型")
            print("=" * 60)

        model = GATKGModel(config_path='config.yaml')
        model.fault_to_idx = self.fault_to_idx
        model.build_model(self.X_train.shape[1], len(self.fault_types))

        early_stop = EarlyStopping(patience=self.es_patience) if use_early_stopping and verbose else None

        if verbose:
            pbar = tqdm(range(epochs), desc="GAT-KG Training")
        else:
            pbar = range(epochs)

        for epoch in pbar:
            train_loss, train_acc = model.train_epoch(self.X_train, self.y_train, self.kg_train_emb)
            if verbose:
                pbar.set_postfix({'loss': f'{train_loss:.4f}', 'acc': f'{train_acc:.4f}'})

            if early_stop:
                val_m, _ = model.evaluate(self.X_val, self.y_val, self.kg_val_emb)
                if early_stop.step(val_m['loss'], model.model.state_dict()):
                    print(f"  早停触发 (epoch {epoch+1})，恢复最佳模型")
                    early_stop.restore_best(model.model)
                    break

        val_metrics, _ = model.evaluate(self.X_val, self.y_val, self.kg_val_emb)

        if verbose:
            print(f"GAT-KG 验证集准确率: {val_metrics['accuracy']:.4f}")
        model.save_model('models/gat_kg_model.pt')

        return model, val_metrics

    def train_gat_kg_v2(self, epochs=100, verbose=True, use_early_stopping=True):
        """训练GAT-KG V2：KG投影增强 + 深层融合"""
        if verbose:
            print("\n" + "=" * 60)
            print("训练 GAT-KG V2 模型 (KG投影增强)")
            print("=" * 60)

        model = GATKGModelV2(config_path='config.yaml')
        model.fault_to_idx = self.fault_to_idx
        model.build_model(self.X_train.shape[1], len(self.fault_types))

        early_stop = EarlyStopping(patience=self.es_patience) if use_early_stopping and verbose else None

        if verbose:
            pbar = tqdm(range(epochs), desc="GAT-KG-V2 Training")
        else:
            pbar = range(epochs)

        for epoch in pbar:
            train_loss, train_acc = model.train_epoch(self.X_train, self.y_train, self.kg_train_emb)
            if verbose:
                pbar.set_postfix({'loss': f'{train_loss:.4f}', 'acc': f'{train_acc:.4f}'})

            if early_stop:
                val_m, _ = model.evaluate(self.X_val, self.y_val, self.kg_val_emb)
                if early_stop.step(val_m['loss'], model.model.state_dict()):
                    print(f"  早停触发 (epoch {epoch+1})，恢复最佳模型")
                    early_stop.restore_best(model.model)
                    break

        val_metrics, _ = model.evaluate(self.X_val, self.y_val, self.kg_val_emb)

        if verbose:
            print(f"GAT-KG-V2 验证集准确率: {val_metrics['accuracy']:.4f}")
        model.save_model('models/gat_kg_v2_model.pt')

        return model, val_metrics

    def save_results(self, results):
        """保存训练结果"""
        with open('results/training_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存至: results/training_results.json")

    def run(self, train_mlp=True, train_kg_v1=True, train_kg_v2=True, train_cnn=True, train_cnn_kg=True, train_cnn_kg_v2=True, train_cnn_kg_v3=True, train_gnn=True, train_gnn_kg=True, train_gnn_kg_v2=False, train_gat=False, train_gat_kg=False, train_gat_kg_v2=False, epochs=100):
        """运行完整训练流程"""
        print("\n" + "=" * 60)
        print("开始训练流程")
        print("=" * 60)
        print(f"训练选项: MLP={train_mlp}, MLP-KG={train_kg_v1}, MLP-KG-V2={train_kg_v2}, CNN={train_cnn}, CNN_KG={train_cnn_kg}, CNN_KG_V2={train_cnn_kg_v2}, CNN_KG_V3={train_cnn_kg_v3}, GNN={train_gnn}, GNN_KG={train_gnn_kg}, GNN_KG_V2={train_gnn_kg_v2}, GAT={train_gat}, GAT_KG={train_gat_kg}, GAT_KG_V2={train_gat_kg_v2}, epochs={epochs}")

        results = {
            'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'best_config': self.best_config,
            'models': {}
        }

        if train_mlp:
            mlp_model, mlp_val = self.train_mlp(epochs=epochs)
            results['models']['MLP'] = {'val_accuracy': float(mlp_val['accuracy'])}

        if train_kg_v1:
            kg_mlp_v1, kg_mlp_v1_val = self.train_kg_mlp_v1(epochs=epochs)
            results['models']['MLP_KG'] = {'val_accuracy': float(kg_mlp_v1_val['accuracy'])}

        if train_kg_v2:
            kg_mlp_v2, kg_mlp_v2_val = self.train_kg_mlp_v2(epochs=epochs)
            results['models']['MLP_KG_V2'] = {'val_accuracy': float(kg_mlp_v2_val['accuracy'])}

        if train_cnn:
            cnn_model, cnn_val = self.train_cnn(epochs=epochs)
            results['models']['CNN'] = {'val_accuracy': float(cnn_val['accuracy'])}

        if train_cnn_kg:
            cnn_kg_model, cnn_kg_val = self.train_cnn_kg(epochs=epochs)
            results['models']['CNN_KG'] = {'val_accuracy': float(cnn_kg_val['accuracy'])}

        if train_cnn_kg_v2:
            cnn_kg_v2_model, cnn_kg_v2_val = self.train_cnn_kg_v2(epochs=epochs)
            results['models']['CNN_KG_V2'] = {'val_accuracy': float(cnn_kg_v2_val['accuracy'])}

        if train_cnn_kg_v3:
            cnn_kg_v3_model, cnn_kg_v3_val = self.train_cnn_kg_v3(epochs=epochs)
            results['models']['CNN_KG_V3'] = {'val_accuracy': float(cnn_kg_v3_val['accuracy'])}

        if train_gnn:
            gnn_model, gnn_val = self.train_gnn(epochs=epochs)
            results['models']['GNN'] = {'val_accuracy': float(gnn_val['accuracy'])}

        if train_gnn_kg:
            gnn_kg_model, gnn_kg_val = self.train_gnn_kg(epochs=epochs)
            results['models']['GNN_KG'] = {'val_accuracy': float(gnn_kg_val['accuracy'])}

        if train_gnn_kg_v2:
            gnn_kg_v2_model, gnn_kg_v2_val = self.train_gnn_kg_v2(epochs=epochs)
            results['models']['GNN_KG_V2'] = {'val_accuracy': float(gnn_kg_v2_val['accuracy'])}

        if train_gat:
            gat_model, gat_val = self.train_gat(epochs=epochs)
            results['models']['GAT'] = {'val_accuracy': float(gat_val['accuracy'])}

        if train_gat_kg:
            gat_kg_model, gat_kg_val = self.train_gat_kg(epochs=epochs)
            results['models']['GAT_KG'] = {'val_accuracy': float(gat_kg_val['accuracy'])}

        if train_gat_kg_v2:
            gat_kg_v2_model, gat_kg_v2_val = self.train_gat_kg_v2(epochs=epochs)
            results['models']['GAT_KG_V2'] = {'val_accuracy': float(gat_kg_v2_val['accuracy'])}

        # 保存结果
        self.save_results(results)

        # 打印汇总
        if results['models']:
            print("\n" + "=" * 60)
            print("训练完成 - 模型汇总")
            print("=" * 60)
            print(f"{'模型':<25} | {'验证集准确率':<15}")
            print("-" * 45)
            for name, data in results['models'].items():
                print(f"{name:<25} | {data['val_accuracy']:.4f}")

        return results


class V2Tuner:
    """V2模型贝叶斯调优器"""

    def __init__(self, trainer, n_trials=30):
        """
        Args:
            trainer: Trainer实例
            n_trials: 试验次数
        """
        self.trainer = trainer
        self.n_trials = n_trials
        self.study = None
        self.best_params = None

    def objective(self, trial):
        """优化目标函数"""
        # 定义搜索空间
        hidden_dim = trial.suggest_categorical('hidden_dim', [64, 96, 128, 160, 192])
        dropout = trial.suggest_float('dropout', 0.1, 0.5, step=0.05)
        lr = trial.suggest_float('lr', 1e-4, 1e-1, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)

        config = {
            'hidden_dim': hidden_dim,
            'dropout': dropout,
            'lr': lr,
            'weight_decay': weight_decay
        }

        # 训练模型（使用较少epoch加快调优）
        model, val_metrics = self.trainer.train_kg_mlp_v2(
            config=config,
            epochs=50,  # 调优时用较少epoch
            verbose=False
        )

        return val_metrics['accuracy']

    def tune(self):
        """执行贝叶斯优化调优"""
        print("\n" + "=" * 60)
        print("开始V2模型贝叶斯优化调优")
        print("=" * 60)
        print(f"试验次数: {self.n_trials}")
        print(f"验证集样本数: {len(self.trainer.X_val)}")

        # 创建研究
        self.study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42)
        )

        # 运行优化
        self.study.optimize(
            self.objective,
            n_trials=self.n_trials,
            show_progress_bar=True
        )

        # 获取最佳参数
        self.best_params = self.study.best_params
        best_value = self.study.best_value

        print("\n" + "=" * 60)
        print("调优完成!")
        print("=" * 60)
        print(f"最佳验证集准确率: {best_value:.4f}")
        print("最佳参数:")
        for k, v in self.best_params.items():
            print(f"  {k}: {v}")

        return self.best_params, best_value

    def train_best(self):
        """使用最佳参数训练最终模型"""
        if self.best_params is None:
            print("错误: 请先运行tune()进行调优")
            return None, None

        print("\n" + "=" * 60)
        print("使用最佳参数训练最终V2模型")
        print("=" * 60)

        config = self.best_params.copy()
        model, val_metrics = self.trainer.train_kg_mlp_v2(
            config=config,
            epochs=100,  # 最终训练用完整epoch
            verbose=True
        )

        return model, val_metrics

    def save_best_params(self, path='results/v2_tuning_results.json'):
        """保存调优结果"""
        if self.study is None:
            return

        results = {
            'best_params': self.best_params,
            'best_value': self.study.best_value,
            'n_trials': self.n_trials,
            'trials': [
                {
                    'params': trial.params,
                    'value': trial.value
                }
                for trial in self.study.trials
                if trial.value is not None
            ]
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\n调优结果已保存至: {path}")


class BatchTuner:
    """批量模型调优器 - 同时调优所有模型"""

    def __init__(self, trainer, n_trials=20):
        """
        Args:
            trainer: Trainer实例
            n_trials: 每个模型的试验次数
        """
        self.trainer = trainer
        self.n_trials = n_trials
        self.results = {}

    def tune_mlp(self, trial):
        """MLP调优目标"""
        hidden_dim = trial.suggest_categorical('hidden_dim', [64, 96, 128, 160, 192, 256])
        dropout = trial.suggest_float('dropout', 0.1, 0.5, step=0.05)
        lr = trial.suggest_float('lr', 1e-4, 1e-1, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)

        # 临时修改配置
        model = MLPModel(config_path='config.yaml')
        original_hidden = model.hidden_dim
        original_dropout = model.dropout
        original_lr = model.learning_rate

        model.hidden_dim = hidden_dim
        model.dropout = dropout
        model.learning_rate = lr
        model.fault_to_idx = self.trainer.fault_to_idx
        model.build_model(self.trainer.X_train.shape[1], len(self.trainer.fault_types))
        model.optimizer = torch.optim.Adam(model.model.parameters(), lr=lr, weight_decay=weight_decay)

        # 训练
        for epoch in range(30):
            model.train_epoch(self.trainer.X_train, self.trainer.y_train)

        # 评估
        val_metrics, _ = model.evaluate(self.trainer.X_val, self.trainer.y_val)

        # 恢复配置
        model.hidden_dim = original_hidden
        model.dropout = original_dropout
        model.learning_rate = original_lr

        return val_metrics['accuracy']

    def tune_kg_mlp_v1(self, trial):
        """MLP-KG调优目标"""
        hidden_dim = trial.suggest_categorical('hidden_dim', [64, 96, 128, 160, 192])
        dropout = trial.suggest_float('dropout', 0.1, 0.5, step=0.05)
        lr = trial.suggest_float('lr', 1e-4, 1e-1, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)

        config = {
            'hidden_dim': hidden_dim,
            'dropout': dropout,
            'lr': lr,
            'weight_decay': weight_decay
        }

        model, val_metrics = self.trainer.train_kg_mlp_v1(config=config, epochs=30, verbose=False)
        return val_metrics['accuracy']

    def tune_kg_mlp_v2(self, trial):
        """MLP-KG-V2调优目标"""
        hidden_dim = trial.suggest_categorical('hidden_dim', [64, 96, 128, 160, 192])
        dropout = trial.suggest_float('dropout', 0.1, 0.5, step=0.05)
        lr = trial.suggest_float('lr', 1e-4, 1e-1, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)

        config = {
            'hidden_dim': hidden_dim,
            'dropout': dropout,
            'lr': lr,
            'weight_decay': weight_decay
        }

        model, val_metrics = self.trainer.train_kg_mlp_v2(config=config, epochs=30, verbose=False)
        return val_metrics['accuracy']

    def tune_cnn(self, trial):
        """CNN调优目标"""
        hidden_dim = trial.suggest_categorical('hidden_channels', [32, 64, 96, 128])
        dropout = trial.suggest_float('dropout', 0.1, 0.5, step=0.05)
        lr = trial.suggest_float('lr', 1e-4, 1e-1, log=True)

        model = CNNModel(config_path='config.yaml')
        model.hidden_dim = hidden_dim
        model.dropout = dropout
        model.learning_rate = lr
        model.fault_to_idx = self.trainer.fault_to_idx
        model.build_model(self.trainer.X_train.shape[1], len(self.trainer.fault_types))

        for epoch in range(30):
            model.train_epoch(self.trainer.X_train, self.trainer.y_train)

        val_metrics, _ = model.evaluate(self.trainer.X_val, self.trainer.y_val)
        return val_metrics['accuracy']

    def tune_cnn_kg(self, trial):
        """CNN-KG调优目标"""
        hidden_dim = trial.suggest_categorical('hidden_channels', [32, 64, 96, 128])
        dropout = trial.suggest_float('dropout', 0.1, 0.5, step=0.05)
        lr = trial.suggest_float('lr', 1e-4, 1e-1, log=True)

        model = CNNKGModel(config_path='config.yaml')
        model.hidden_dim = hidden_dim
        model.dropout = dropout
        model.learning_rate = lr
        model.fault_to_idx = self.trainer.fault_to_idx
        model.build_model(self.trainer.X_train.shape[1], len(self.trainer.fault_types))

        for epoch in range(30):
            model.train_epoch(self.trainer.X_train, self.trainer.y_train, self.trainer.kg_train_emb)

        val_metrics, _ = model.evaluate(self.trainer.X_val, self.trainer.y_val, self.trainer.kg_val_emb)
        return val_metrics['accuracy']

    def tune_cnn_kg_v2(self, trial):
        """CNN-KG V2调优目标"""
        hidden_dim = trial.suggest_categorical('hidden_channels', [32, 64, 96, 128])
        dropout = trial.suggest_float('dropout', 0.1, 0.5, step=0.05)
        lr = trial.suggest_float('lr', 1e-4, 1e-1, log=True)

        model = CNNKGModelV2(config_path='config.yaml')
        model.hidden_dim = hidden_dim
        model.dropout = dropout
        model.learning_rate = lr
        model.fault_to_idx = self.trainer.fault_to_idx
        model.build_model(self.trainer.X_train.shape[1], len(self.trainer.fault_types))

        for epoch in range(30):
            model.train_epoch(self.trainer.X_train, self.trainer.y_train, self.trainer.kg_train_emb)

        val_metrics, _ = model.evaluate(self.trainer.X_val, self.trainer.y_val, self.trainer.kg_val_emb)
        return val_metrics['accuracy']

    def tune_cnn_kg_v3(self, trial):
        """CNN-KG V3调优目标"""
        hidden_dim = trial.suggest_categorical('hidden_channels', [32, 64, 96, 128])
        dropout = trial.suggest_float('dropout', 0.1, 0.4, step=0.05)
        lr = trial.suggest_float('lr', 1e-4, 1e-1, log=True)

        model = CNNKGModelV3(config_path='config.yaml')
        model.hidden_dim = hidden_dim
        model.dropout = dropout
        model.learning_rate = lr
        model.fault_to_idx = self.trainer.fault_to_idx
        model.build_model(self.trainer.X_train.shape[1], len(self.trainer.fault_types))

        for epoch in range(30):
            model.train_epoch(self.trainer.X_train, self.trainer.y_train, self.trainer.kg_train_emb)

        val_metrics, _ = model.evaluate(self.trainer.X_val, self.trainer.y_val, self.trainer.kg_val_emb)
        return val_metrics['accuracy']

    def tune_gnn(self, trial):
        """GNN调优目标"""
        hidden_dim = trial.suggest_categorical('hidden_dim', [128, 192, 256, 320])
        num_layers = trial.suggest_categorical('num_layers', [2, 3, 4])
        dropout = trial.suggest_float('dropout', 0.1, 0.5, step=0.05)
        lr = trial.suggest_float('lr', 1e-4, 1e-1, log=True)

        model = GNNModel(config_path='config.yaml')
        model.hidden_dim = hidden_dim
        model.num_layers = num_layers
        model.dropout = dropout
        model.learning_rate = lr
        model.fault_to_idx = self.trainer.fault_to_idx
        model.build_model(self.trainer.X_train.shape[1], len(self.trainer.fault_types))

        for epoch in range(30):
            model.train_epoch(self.trainer.X_train, self.trainer.y_train)

        val_metrics, _ = model.evaluate(self.trainer.X_val, self.trainer.y_val)
        return val_metrics['accuracy']

    def tune_gnn_kg(self, trial):
        """GNN-KG调优目标"""
        hidden_dim = trial.suggest_categorical('hidden_dim', [128, 192, 256, 320])
        num_layers = trial.suggest_categorical('num_layers', [2, 3, 4])
        dropout = trial.suggest_float('dropout', 0.1, 0.5, step=0.05)
        lr = trial.suggest_float('lr', 1e-4, 1e-1, log=True)

        model = GNNKGModel(config_path='config.yaml')
        model.hidden_dim = hidden_dim
        model.num_layers = num_layers
        model.dropout = dropout
        model.learning_rate = lr
        model.kg_embedding_dim = 33
        model.fault_to_idx = self.trainer.fault_to_idx
        model.build_model(self.trainer.X_train.shape[1], len(self.trainer.fault_types))

        for epoch in range(30):
            model.train_epoch(self.trainer.X_train, self.trainer.y_train, self.trainer.kg_train_emb)

        val_metrics, _ = model.evaluate(self.trainer.X_val, self.trainer.y_val, self.trainer.kg_val_emb)
        return val_metrics['accuracy']

    def tune_gnn_kg_v2(self, trial):
        """GNN-KG V2调优目标"""
        hidden_dim = trial.suggest_categorical('hidden_dim', [128, 192, 256, 320])
        kg_embedding_dim = trial.suggest_categorical('kg_embedding_dim', [16, 32, 48, 64])
        dropout = trial.suggest_float('dropout', 0.1, 0.5, step=0.05)
        lr = trial.suggest_float('lr', 1e-4, 1e-1, log=True)

        model = GNNKGModelV2(config_path='config.yaml')
        model.hidden_dim = hidden_dim
        model.kg_embedding_dim = kg_embedding_dim
        model.dropout = dropout
        model.learning_rate = lr
        model.fault_to_idx = self.trainer.fault_to_idx
        model.build_model(self.trainer.X_train.shape[1], len(self.trainer.fault_types))

        for epoch in range(30):
            model.train_epoch(self.trainer.X_train, self.trainer.y_train)

        val_metrics, _ = model.evaluate(self.trainer.X_val, self.trainer.y_val)
        return val_metrics['accuracy']

    def tune_gat(self, trial):
        """GAT调优目标"""
        hidden_dim = trial.suggest_categorical('hidden_dim', [128, 192, 256, 320])
        heads = trial.suggest_categorical('heads', [2, 4, 8])
        dropout = trial.suggest_float('dropout', 0.1, 0.5, step=0.05)
        lr = trial.suggest_float('lr', 1e-4, 1e-1, log=True)

        model = GATModel(config_path='config.yaml')
        model.hidden_dim = hidden_dim
        model.heads = heads
        model.dropout = dropout
        model.learning_rate = lr
        model.fault_to_idx = self.trainer.fault_to_idx
        model.build_model(self.trainer.X_train.shape[1], len(self.trainer.fault_types))

        for epoch in range(30):
            model.train_epoch(self.trainer.X_train, self.trainer.y_train)

        val_metrics, _ = model.evaluate(self.trainer.X_val, self.trainer.y_val)
        return val_metrics['accuracy']

    def tune_gat_kg(self, trial):
        """GAT-KG调优目标"""
        hidden_dim = trial.suggest_categorical('hidden_dim', [128, 192, 256, 320])
        heads = trial.suggest_categorical('heads', [2, 4, 8])
        dropout = trial.suggest_float('dropout', 0.1, 0.5, step=0.05)
        lr = trial.suggest_float('lr', 1e-4, 1e-1, log=True)

        model = GATKGModel(config_path='config.yaml')
        model.hidden_dim = hidden_dim
        model.heads = heads
        model.dropout = dropout
        model.learning_rate = lr
        model.fault_to_idx = self.trainer.fault_to_idx
        model.build_model(self.trainer.X_train.shape[1], len(self.trainer.fault_types))

        for epoch in range(30):
            model.train_epoch(self.trainer.X_train, self.trainer.y_train, self.trainer.kg_train_emb)

        val_metrics, _ = model.evaluate(self.trainer.X_val, self.trainer.y_val, self.trainer.kg_val_emb)
        return val_metrics['accuracy']

    def tune_gat_kg_v2(self, trial):
        """GAT-KG V2调优目标"""
        hidden_dim = trial.suggest_categorical('hidden_dim', [128, 192, 256, 320])
        heads = trial.suggest_categorical('heads', [2, 4, 8])
        dropout = trial.suggest_float('dropout', 0.1, 0.5, step=0.05)
        lr = trial.suggest_float('lr', 1e-4, 1e-1, log=True)

        model = GATKGModelV2(config_path='config.yaml')
        model.hidden_dim = hidden_dim
        model.heads = heads
        model.dropout = dropout
        model.learning_rate = lr
        model.fault_to_idx = self.trainer.fault_to_idx
        model.build_model(self.trainer.X_train.shape[1], len(self.trainer.fault_types))

        for epoch in range(30):
            model.train_epoch(self.trainer.X_train, self.trainer.y_train, self.trainer.kg_train_emb)

        val_metrics, _ = model.evaluate(self.trainer.X_val, self.trainer.y_val, self.trainer.kg_val_emb)
        return val_metrics['accuracy']

    def tune_all(self):
        """批量调优所有模型"""
        models_config = [
            ('MLP', self.tune_mlp),
            ('MLP-KG', self.tune_kg_mlp_v1),
            ('MLP-KG-V2', self.tune_kg_mlp_v2),
            ('CNN', self.tune_cnn),
            ('CNN-KG', self.tune_cnn_kg),
            ('CNN-KG V2', self.tune_cnn_kg_v2),
            ('CNN-KG V3', self.tune_cnn_kg_v3),
            ('GNN', self.tune_gnn),
            ('GNN-KG', self.tune_gnn_kg),
            ('GNN-KG V2', self.tune_gnn_kg_v2),
            ('GAT', self.tune_gat),
            ('GAT-KG', self.tune_gat_kg),
            ('GAT-KG V2', self.tune_gat_kg_v2),
        ]

        print("\n" + "=" * 60)
        print("批量模型调优")
        print("=" * 60)
        print(f"每个模型试验次数: {self.n_trials}")
        print(f"验证集样本数: {len(self.trainer.X_val)}")
        print("=" * 60)

        for model_name, objective_fn in models_config:
            print(f"\n{'='*60}")
            print(f"正在调优: {model_name}")
            print(f"{'='*60}")

            study = optuna.create_study(
                direction='maximize',
                sampler=optuna.samplers.TPESampler(seed=42)
            )

            study.optimize(
                objective_fn,
                n_trials=self.n_trials,
                show_progress_bar=True
            )

            self.results[model_name] = {
                'best_params': study.best_params,
                'best_value': study.best_value,
                'n_trials': len(study.trials)
            }

            print(f"  最佳准确率: {study.best_value:.4f}")
            print(f"  最佳参数: {study.best_params}")

        return self.results

    def train_all_best(self, epochs=100):
        """使用各模型最佳参数训练"""
        print("\n" + "=" * 60)
        print("使用最佳参数训练所有模型")
        print("=" * 60)

        trained_models = {}

        # MLP
        if 'MLP' in self.results:
            model = MLPModel(config_path='config.yaml')
            for k, v in self.results['MLP']['best_params'].items():
                setattr(model, k, v)
            model.fault_to_idx = self.trainer.fault_to_idx
            model.build_model(self.trainer.X_train.shape[1], len(self.trainer.fault_types))
            for epoch in range(epochs):
                model.train_epoch(self.trainer.X_train, self.trainer.y_train)
            model.save_model('models/mlp_model.pt')
            val_metrics, _ = model.evaluate(self.trainer.X_val, self.trainer.y_val)
            trained_models['MLP'] = val_metrics
            print(f"MLP - 验证准确率: {val_metrics['accuracy']:.4f}")

        # MLP-KG V1
        if 'MLP_KG' in self.results:
            model, val_metrics = self.trainer.train_kg_mlp_v1(
                config=self.results['MLP_KG']['best_params'],
                epochs=epochs
            )
            model.save_model('models/mlp_kg_model.pt')
            trained_models['MLP_KG'] = val_metrics
            print(f"MLP-KG - 验证准确率: {val_metrics['accuracy']:.4f}")

        # MLP-KG V2
        if 'MLP_KG_V2' in self.results:
            model, val_metrics = self.trainer.train_kg_mlp_v2(
                config=self.results['MLP_KG_V2']['best_params'],
                epochs=epochs, verbose=False
            )
            model.save_model('models/mlp_kg_v2_model.pt')
            trained_models['MLP_KG_V2'] = val_metrics
            print(f"MLP-KG-V2 - 验证准确率: {val_metrics['accuracy']:.4f}")

        # CNN
        if 'CNN' in self.results:
            model = CNNModel(config_path='config.yaml')
            for k, v in self.results['CNN']['best_params'].items():
                setattr(model, k, v)
            model.fault_to_idx = self.trainer.fault_to_idx
            model.build_model(self.trainer.X_train.shape[1], len(self.trainer.fault_types))
            for epoch in range(epochs):
                model.train_epoch(self.trainer.X_train, self.trainer.y_train)
            model.save_model('models/cnn_model.pt')
            val_metrics, _ = model.evaluate(self.trainer.X_val, self.trainer.y_val)
            trained_models['CNN'] = val_metrics
            print(f"CNN - 验证准确率: {val_metrics['accuracy']:.4f}")

        # CNN-KG
        if 'CNN-KG' in self.results:
            model = CNNKGModel(config_path='config.yaml')
            for k, v in self.results['CNN-KG']['best_params'].items():
                setattr(model, k, v)
            model.fault_to_idx = self.trainer.fault_to_idx
            model.build_model(self.trainer.X_train.shape[1], len(self.trainer.fault_types))
            for epoch in range(epochs):
                model.train_epoch(self.trainer.X_train, self.trainer.y_train, self.trainer.kg_train_emb)
            model.save_model('models/cnn_kg_model.pt')
            val_metrics, _ = model.evaluate(self.trainer.X_val, self.trainer.y_val, self.trainer.kg_val_emb)
            trained_models['CNN_KG'] = val_metrics
            print(f"CNN-KG - 验证准确率: {val_metrics['accuracy']:.4f}")

        # CNN-KG V2
        if 'CNN-KG V2' in self.results:
            model = CNNKGModelV2(config_path='config.yaml')
            for k, v in self.results['CNN-KG V2']['best_params'].items():
                setattr(model, k, v)
            model.fault_to_idx = self.trainer.fault_to_idx
            model.build_model(self.trainer.X_train.shape[1], len(self.trainer.fault_types))
            for epoch in range(epochs):
                model.train_epoch(self.trainer.X_train, self.trainer.y_train, self.trainer.kg_train_emb)
            model.save_model('models/cnn_kg_v2_model.pt')
            val_metrics, _ = model.evaluate(self.trainer.X_val, self.trainer.y_val, self.trainer.kg_val_emb)
            trained_models['CNN_KG_V2'] = val_metrics
            print(f"CNN-KG V2 - 验证准确率: {val_metrics['accuracy']:.4f}")

        # CNN-KG V3
        if 'CNN-KG V3' in self.results:
            model = CNNKGModelV3(config_path='config.yaml')
            for k, v in self.results['CNN-KG V3']['best_params'].items():
                setattr(model, k, v)
            model.fault_to_idx = self.trainer.fault_to_idx
            model.build_model(self.trainer.X_train.shape[1], len(self.trainer.fault_types))
            for epoch in range(epochs):
                model.train_epoch(self.trainer.X_train, self.trainer.y_train, self.trainer.kg_train_emb)
            model.save_model('models/cnn_kg_v3_model.pt')
            val_metrics, _ = model.evaluate(self.trainer.X_val, self.trainer.y_val, self.trainer.kg_val_emb)
            trained_models['CNN_KG_V3'] = val_metrics
            print(f"CNN-KG V3 - 验证准确率: {val_metrics['accuracy']:.4f}")

        # GNN
        if 'GNN' in self.results:
            model = GNNModel(config_path='config.yaml')
            for k, v in self.results['GNN']['best_params'].items():
                setattr(model, k, v)
            model.fault_to_idx = self.trainer.fault_to_idx
            model.build_model(self.trainer.X_train.shape[1], len(self.trainer.fault_types))
            pbar = tqdm(range(epochs), desc="GNN Training")
            for epoch in pbar:
                train_loss, train_acc = model.train_epoch(self.trainer.X_train, self.trainer.y_train)
                pbar.set_postfix({'loss': f'{train_loss:.4f}', 'acc': f'{train_acc:.4f}'})
            model.save_model('models/gnn_model.pt')
            val_metrics, _ = model.evaluate(self.trainer.X_val, self.trainer.y_val)
            trained_models['GNN'] = val_metrics
            print(f"GNN - 验证准确率: {val_metrics['accuracy']:.4f}")

        # GNN-KG
        if 'GNN-KG' in self.results:
            model = GNNKGModel(config_path='config.yaml')
            for k, v in self.results['GNN-KG']['best_params'].items():
                setattr(model, k, v)
            model.kg_embedding_dim = 33
            model.fault_to_idx = self.trainer.fault_to_idx
            model.build_model(self.trainer.X_train.shape[1], len(self.trainer.fault_types))
            pbar = tqdm(range(epochs), desc="GNN-KG Training")
            for epoch in pbar:
                train_loss, train_acc = model.train_epoch(self.trainer.X_train, self.trainer.y_train, self.trainer.kg_train_emb)
                pbar.set_postfix({'loss': f'{train_loss:.4f}', 'acc': f'{train_acc:.4f}'})
            model.save_model('models/gnn_kg_model.pt')
            val_metrics, _ = model.evaluate(self.trainer.X_val, self.trainer.y_val, self.trainer.kg_val_emb)
            trained_models['GNN_KG'] = val_metrics
            print(f"GNN-KG - 验证准确率: {val_metrics['accuracy']:.4f}")

        # GNN-KG V2
        if 'GNN-KG V2' in self.results:
            model = GNNKGModelV2(config_path='config.yaml')
            for k, v in self.results['GNN-KG V2']['best_params'].items():
                setattr(model, k, v)
            model.fault_to_idx = self.trainer.fault_to_idx
            model.build_model(self.trainer.X_train.shape[1], len(self.trainer.fault_types))
            pbar = tqdm(range(epochs), desc="GNN-KG-V2 Training")
            for epoch in pbar:
                train_loss, train_acc = model.train_epoch(self.trainer.X_train, self.trainer.y_train)
                pbar.set_postfix({'loss': f'{train_loss:.4f}', 'acc': f'{train_acc:.4f}'})
            model.save_model('models/gnn_kg_v2_model.pt')
            val_metrics, _ = model.evaluate(self.trainer.X_val, self.trainer.y_val)
            trained_models['GNN_KG_V2'] = val_metrics
            print(f"GNN-KG V2 - 验证准确率: {val_metrics['accuracy']:.4f}")

        # GAT
        if 'GAT' in self.results:
            model = GATModel(config_path='config.yaml')
            for k, v in self.results['GAT']['best_params'].items():
                setattr(model, k, v)
            model.fault_to_idx = self.trainer.fault_to_idx
            model.build_model(self.trainer.X_train.shape[1], len(self.trainer.fault_types))
            pbar = tqdm(range(epochs), desc="GAT Training")
            for epoch in pbar:
                train_loss, train_acc = model.train_epoch(self.trainer.X_train, self.trainer.y_train)
                pbar.set_postfix({'loss': f'{train_loss:.4f}', 'acc': f'{train_acc:.4f}'})
            model.save_model('models/gat_model.pt')
            val_metrics, _ = model.evaluate(self.trainer.X_val, self.trainer.y_val)
            trained_models['GAT'] = val_metrics
            print(f"GAT - 验证准确率: {val_metrics['accuracy']:.4f}")

        # GAT-KG
        if 'GAT-KG' in self.results:
            model = GATKGModel(config_path='config.yaml')
            for k, v in self.results['GAT-KG']['best_params'].items():
                setattr(model, k, v)
            model.fault_to_idx = self.trainer.fault_to_idx
            model.build_model(self.trainer.X_train.shape[1], len(self.trainer.fault_types))
            pbar = tqdm(range(epochs), desc="GAT-KG Training")
            for epoch in pbar:
                train_loss, train_acc = model.train_epoch(self.trainer.X_train, self.trainer.y_train, self.trainer.kg_train_emb)
                pbar.set_postfix({'loss': f'{train_loss:.4f}', 'acc': f'{train_acc:.4f}'})
            model.save_model('models/gat_kg_model.pt')
            val_metrics, _ = model.evaluate(self.trainer.X_val, self.trainer.y_val, self.trainer.kg_val_emb)
            trained_models['GAT_KG'] = val_metrics
            print(f"GAT-KG - 验证准确率: {val_metrics['accuracy']:.4f}")

        # GAT-KG V2
        if 'GAT-KG V2' in self.results:
            model = GATKGModelV2(config_path='config.yaml')
            for k, v in self.results['GAT-KG V2']['best_params'].items():
                setattr(model, k, v)
            model.fault_to_idx = self.trainer.fault_to_idx
            model.build_model(self.trainer.X_train.shape[1], len(self.trainer.fault_types))
            pbar = tqdm(range(epochs), desc="GAT-KG-V2 Training")
            for epoch in pbar:
                train_loss, train_acc = model.train_epoch(self.trainer.X_train, self.trainer.y_train, self.trainer.kg_train_emb)
                pbar.set_postfix({'loss': f'{train_loss:.4f}', 'acc': f'{train_acc:.4f}'})
            model.save_model('models/gat_kg_v2_model.pt')
            val_metrics, _ = model.evaluate(self.trainer.X_val, self.trainer.y_val, self.trainer.kg_val_emb)
            trained_models['GAT_KG_V2'] = val_metrics
            print(f"GAT-KG V2 - 验证准确率: {val_metrics['accuracy']:.4f}")

        return trained_models

    def save_results(self, path='results/batch_tuning_results.json'):
        """保存批量调优结果"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"\n批量调优结果已保存至: {path}")


def main():
    parser = argparse.ArgumentParser(description='训练故障预测模型')
    parser.add_argument('--mlp', action='store_true', help='训练MLP模型')
    parser.add_argument('--mlp-kg', action='store_true', help='训练MLP-KG模型 (全局嵌入)')
    parser.add_argument('--mlp-kg-v2', action='store_true', help='训练MLP-KG-V2模型 (故障级别嵌入)')
    parser.add_argument('--cnn', action='store_true', help='训练CNN模型')
    parser.add_argument('--cnn-kg', action='store_true', help='训练CNN-KG融合模型')
    parser.add_argument('--cnn-kg-v2', action='store_true', help='训练CNN-KG融合模型 V2 (门控融合)')
    parser.add_argument('--cnn-kg-v3', action='store_true', help='训练CNN-KG融合模型 V3 (残差连接)')
    parser.add_argument('--gnn', action='store_true', help='训练GNN模型')
    parser.add_argument('--gnn-kg', action='store_true', help='训练GNN-KG融合模型')
    parser.add_argument('--gnn-kg-v2', action='store_true', help='训练GNN-KG-V2模型 (交叉注意力+可学习嵌入)')
    parser.add_argument('--gat', action='store_true', help='训练GAT模型 (图注意力网络)')
    parser.add_argument('--gat-kg', action='store_true', help='训练GAT-KG融合模型')
    parser.add_argument('--gat-kg-v2', action='store_true', help='训练GAT-KG-V2模型 (KG投影增强)')
    parser.add_argument('--tune-v2', action='store_true', help='对V2模型进行贝叶斯优化调优')
    parser.add_argument('--tune-all', action='store_true', help='批量调优所有模型')
    parser.add_argument('--n-trials', type=int, default=20, help='调优试验次数 (默认20)')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数 (默认100)')
    parser.add_argument('--all', action='store_true', help='训练所有模型')
    args = parser.parse_args()

    # 批量调优模式
    if args.tune_all:
        print("=" * 60)
        print("批量模型调优模式")
        print("=" * 60)

        trainer = Trainer()
        tuner = BatchTuner(trainer, n_trials=args.n_trials)
        tuner.tune_all()
        tuner.save_results()

        print("\n" + "=" * 60)
        print("使用最佳参数训练所有模型...")
        print("=" * 60)
        tuner.train_all_best(epochs=args.epochs)

        print("\n批量调优完成！")
        return

    # V2调优模式
    if args.tune_v2:
        print("=" * 60)
        print("V2模型调优模式")
        print("=" * 60)

        trainer = Trainer()
        tuner = V2Tuner(trainer, n_trials=args.n_trials)
        best_params, best_value = tuner.tune()

        # 使用最佳参数训练最终模型
        model, val_metrics = tuner.train_best()
        tuner.save_best_params()

        print(f"\n最终模型验证集准确率: {val_metrics['accuracy']:.4f}")
        return

    # 常规训练模式
    if args.all:
        train_mlp = train_mlp_kg = train_mlp_kg_v2 = train_cnn = train_cnn_kg = train_cnn_kg_v2 = train_cnn_kg_v3 = train_gnn = train_gnn_kg = True
        train_gnn_kg_v2 = train_gat = train_gat_kg = train_gat_kg_v2 = True
    else:
        any_model_specified = (args.mlp_kg or args.mlp_kg_v2 or args.cnn or args.cnn_kg or args.cnn_kg_v2 or args.cnn_kg_v3 or args.gnn or args.gnn_kg or args.gnn_kg_v2 or args.gat or args.gat_kg or args.gat_kg_v2)
        train_mlp = args.mlp or not any_model_specified
        train_mlp_kg = args.mlp_kg
        train_mlp_kg_v2 = args.mlp_kg_v2
        train_cnn = args.cnn
        train_cnn_kg = args.cnn_kg
        train_cnn_kg_v2 = args.cnn_kg_v2
        train_cnn_kg_v3 = args.cnn_kg_v3
        train_gnn = args.gnn
        train_gnn_kg = args.gnn_kg
        train_gnn_kg_v2 = args.gnn_kg_v2
        train_gat = args.gat
        train_gat_kg = args.gat_kg
        train_gat_kg_v2 = args.gat_kg_v2

    if not (train_mlp or train_mlp_kg or train_mlp_kg_v2 or train_cnn or train_cnn_kg or train_cnn_kg_v2 or train_cnn_kg_v3 or train_gnn or train_gnn_kg or train_gnn_kg_v2 or train_gat or train_gat_kg or train_gat_kg_v2):
        print("请选择要训练的模型，使用 --mlp, --mlp-kg, --mlp-kg-v2, --cnn, --cnn-kg, --cnn-kg-v2, --cnn-kg-v3, --gnn, --gnn-kg, --gnn-kg-v2, --gat, --gat-kg, --gat-kg-v2, --tune-v2, --tune-all 或 --all")
        return

    trainer = Trainer()
    trainer.run(train_mlp=train_mlp, train_kg_v1=train_mlp_kg, train_kg_v2=train_mlp_kg_v2, train_cnn=train_cnn, train_cnn_kg=train_cnn_kg, train_cnn_kg_v2=train_cnn_kg_v2, train_cnn_kg_v3=train_cnn_kg_v3, train_gnn=train_gnn, train_gnn_kg=train_gnn_kg, train_gnn_kg_v2=train_gnn_kg_v2, train_gat=train_gat, train_gat_kg=train_gat_kg, train_gat_kg_v2=train_gat_kg_v2, epochs=args.epochs)


if __name__ == '__main__':
    main()