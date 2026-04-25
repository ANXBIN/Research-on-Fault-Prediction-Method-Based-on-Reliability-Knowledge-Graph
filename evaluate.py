#!/usr/bin/env python3
"""
评估脚本 - 评估models文件夹中已存在的模型
自动检测并评估
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import torch
import json
import argparse
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from tqdm import tqdm

from src.models.mlp_model import MLPModel, KGEnhancedMLPModel, KGEnhancedMLPV2Model, load_kg_embeddings_v3, load_kg_embeddings_v4, load_kg_embeddings_mlp
from src.models.cnn_model import CNNModel, CNNKGModel, CNNKGModelV2, CNNKGModelV3
from src.models.gnn_model import GNNModel, GNNKGModel


class Evaluator:
    def __init__(self):
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        print(f"[INFO] 使用设备: {self.device}")

        # 最佳配置
        self.best_config = {
            'hidden_dim': 128,
            'dropout': 0.215,
            'lr': 0.0064,
            'weight_decay': 0.00021
        }

        # 加载数据
        self.load_data()

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

        # V1嵌入：使用旧的全局KG嵌入（6维）
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

        # V2嵌入：使用故障级别KG嵌入（33维）
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
        self.kg_train_emb_mlp, self.kg_val_emb_mlp, self.kg_test_emb_mlp = load_kg_embeddings_mlp(
            self.X_train, self.X_val, self.X_test, k=20
        )

        print(f"训练集: {len(self.X_train)} 样本")
        print(f"验证集: {len(self.X_val)} 样本")
        print(f"测试集: {len(self.X_test)} 样本")
        print(f"V1 KG嵌入维度: {self.kg_train_emb_v1.shape}")
        print(f"V2 KG嵌入维度: {self.kg_train_emb.shape}")
        print(f"MLP KNN嵌入维度: {self.kg_train_emb_mlp.shape}")

    def get_available_models(self):
        """检测models文件夹中可用的模型"""
        models_dir = Path('models')
        available = {}

        mlp_path = models_dir / 'mlp_model.pt'
        kg_v1_path = models_dir / 'mlp_kg_model.pt'
        kg_v2_path = models_dir / 'mlp_kg_v2_model.pt'
        cnn_path = models_dir / 'cnn_model.pt'
        cnn_kg_path = models_dir / 'cnn_kg_model.pt'
        cnn_kg_v2_path = models_dir / 'cnn_kg_v2_model.pt'
        cnn_kg_v3_path = models_dir / 'cnn_kg_v3_model.pt'
        gnn_path = models_dir / 'gnn_model.pt'
        gnn_kg_path = models_dir / 'gnn_kg_model.pt'

        if mlp_path.exists():
            available['MLP'] = mlp_path
        if kg_v1_path.exists():
            available['MLP_KG'] = kg_v1_path
        if kg_v2_path.exists():
            available['MLP_KG_V2'] = kg_v2_path
        if cnn_path.exists():
            available['CNN'] = cnn_path
        if cnn_kg_path.exists():
            available['CNN_KG'] = cnn_kg_path
        if cnn_kg_v2_path.exists():
            available['CNN_KG_V2'] = cnn_kg_v2_path
        if cnn_kg_v3_path.exists():
            available['CNN_KG_V3'] = cnn_kg_v3_path
        if gnn_path.exists():
            available['GNN'] = gnn_path
        if gnn_kg_path.exists():
            available['GNN_KG'] = gnn_kg_path

        return available

    def load_mlp(self):
        """加载MLP模型"""
        checkpoint = torch.load('models/mlp_model.pt', map_location=self.device)
        saved_config = checkpoint.get('config', {})

        model = MLPModel(config_path='config.yaml')
        model.hidden_dim = saved_config.get('hidden_dim', model.hidden_dim)
        model.dropout = saved_config.get('dropout', model.dropout)
        model.fault_to_idx = self.fault_to_idx
        model.build_model(self.X_train.shape[1], len(self.fault_types))
        model.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"[INFO] MLP模型已加载 (hidden_dim={model.hidden_dim})")
        return model

    def load_kg_mlp_v1(self):
        """加载MLP-KG模型"""
        checkpoint = torch.load('models/mlp_kg_model.pt', map_location=self.device)
        saved_config = checkpoint.get('config', {})

        model = KGEnhancedMLPModel(config_path='config.yaml')
        model.hidden_dim = saved_config.get('hidden_dim', self.best_config['hidden_dim'])
        model.kg_embedding_dim = saved_config.get('kg_embedding_dim', 64)
        model.dropout = saved_config.get('dropout', self.best_config['dropout'])
        model.learning_rate = saved_config.get('learning_rate', self.best_config['lr'])
        model.fault_to_idx = self.fault_to_idx
        model.build_model(self.X_train.shape[1], len(self.fault_types))
        model.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"[INFO] MLP-KG模型已加载 (hidden_dim={model.hidden_dim}, kg_dim={model.kg_embedding_dim})")
        return model

    def load_kg_mlp_v2(self):
        """加载MLP-KG-V2模型"""
        checkpoint = torch.load('models/mlp_kg_v2_model.pt', map_location=self.device)
        saved_config = checkpoint.get('config', {})

        model = KGEnhancedMLPV2Model(config_path='config.yaml')
        model.hidden_dim = saved_config.get('hidden_dim', self.best_config['hidden_dim'])
        model.kg_embedding_dim = saved_config.get('kg_embedding_dim', 64)
        model.dropout = saved_config.get('dropout', self.best_config['dropout'])
        model.learning_rate = saved_config.get('learning_rate', self.best_config['lr'])
        model.fault_to_idx = self.fault_to_idx
        model.build_model(self.X_train.shape[1], len(self.fault_types))
        model.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"[INFO] MLP-KG-V2模型已加载 (hidden_dim={model.hidden_dim}, kg_dim={model.kg_embedding_dim})")
        return model

    def load_cnn(self):
        """加载CNN模型"""
        checkpoint = torch.load('models/cnn_model.pt', map_location=self.device)
        saved_config = checkpoint.get('config', {})

        model = CNNModel(config_path='config.yaml')
        model.hidden_dim = saved_config.get('hidden_dim', 64)
        model.dropout = saved_config.get('dropout', 0.3)
        model.fault_to_idx = self.fault_to_idx
        model.build_model(self.X_train.shape[1], len(self.fault_types))
        model.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"[INFO] CNN模型已加载")
        return model

    def load_cnn_kg(self):
        """加载CNN-KG融合模型"""
        checkpoint = torch.load('models/cnn_kg_model.pt', map_location=self.device)
        saved_config = checkpoint.get('config', {})

        model = CNNKGModel(config_path='config.yaml')
        model.hidden_dim = saved_config.get('hidden_dim', 64)
        model.kg_embedding_dim = saved_config.get('kg_embedding_dim', 33)
        model.dropout = saved_config.get('dropout', 0.3)
        model.fault_to_idx = self.fault_to_idx
        model.build_model(self.X_train.shape[1], len(self.fault_types))
        model.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"[INFO] CNN-KG模型已加载")
        return model

    def load_cnn_kg_v2(self):
        """加载CNN-KG融合模型V2"""
        checkpoint = torch.load('models/cnn_kg_v2_model.pt', map_location=self.device)
        saved_config = checkpoint.get('config', {})

        model = CNNKGModelV2(config_path='config.yaml')
        model.hidden_dim = saved_config.get('hidden_dim', 64)
        model.kg_embedding_dim = saved_config.get('kg_embedding_dim', 33)
        model.dropout = saved_config.get('dropout', 0.3)
        model.fault_to_idx = self.fault_to_idx
        model.build_model(self.X_train.shape[1], len(self.fault_types))
        model.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"[INFO] CNN-KG V2模型已加载")
        return model

    def load_cnn_kg_v3(self):
        """加载CNN-KG融合模型V3 (残差连接)"""
        checkpoint = torch.load('models/cnn_kg_v3_model.pt', map_location=self.device)
        saved_config = checkpoint.get('config', {})

        model = CNNKGModelV3(config_path='config.yaml')
        model.hidden_dim = saved_config.get('hidden_dim', 64)
        model.kg_embedding_dim = saved_config.get('kg_embedding_dim', 33)
        model.dropout = saved_config.get('dropout', 0.2)
        model.fault_to_idx = self.fault_to_idx
        model.build_model(self.X_train.shape[1], len(self.fault_types))
        model.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"[INFO] CNN-KG V3模型已加载")
        return model

    def load_gnn(self):
        """加载GNN模型 (V2改进版)"""
        checkpoint = torch.load('models/gnn_model.pt', map_location=self.device)
        saved_config = checkpoint.get('config', {})

        model = GNNModel(config_path='config.yaml')
        model.hidden_dim = saved_config.get('hidden_dim', 256)
        model.num_layers = saved_config.get('num_layers', 4)
        model.dropout = saved_config.get('dropout', 0.3)
        model.batch_size = saved_config.get('batch_size', 256)
        model.fault_to_idx = self.fault_to_idx
        model.build_model(self.X_train.shape[1], len(self.fault_types))
        model.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"[INFO] GNN模型已加载 (hidden_dim={model.hidden_dim})")
        return model

    def load_gnn_kg(self):
        """加载GNN-KG融合模型 (V2改进版)"""
        checkpoint = torch.load('models/gnn_kg_model.pt', map_location=self.device)
        saved_config = checkpoint.get('config', {})

        model = GNNKGModel(config_path='config.yaml')
        model.hidden_dim = saved_config.get('hidden_dim', 256)
        model.num_layers = saved_config.get('num_layers', 4)
        model.dropout = saved_config.get('dropout', 0.3)
        model.kg_embedding_dim = saved_config.get('kg_embedding_dim', 33)
        model.batch_size = saved_config.get('batch_size', 256)
        model.fault_to_idx = self.fault_to_idx
        model.build_model(self.X_train.shape[1], len(self.fault_types))
        model.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"[INFO] GNN-KG模型已加载 (hidden_dim={model.hidden_dim})")
        return model

    def evaluate_model(self, model, X, y, kg_emb=None, model_name="Model"):
        """评估单个模型"""
        if kg_emb is not None:
            metrics, y_pred = model.evaluate(X, y, kg_emb)
        else:
            metrics, y_pred = model.evaluate(X, y)

        acc = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred, average='weighted')

        return {'accuracy': acc, 'f1': f1}, y_pred

    def run(self):
        """运行评估流程"""
        # 检测可用模型
        available = self.get_available_models()

        if not available:
            print("[ERROR] models文件夹中没有找到任何模型文件！")
            print("请先运行训练脚本: python train.py --all")
            return None

        print("\n" + "=" * 60)
        print("自动检测到以下模型:")
        print("=" * 60)
        for name in available.keys():
            print(f"  - {name}")
        print()

        results = {
            'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'best_config': self.best_config,
            'validation': {},
            'test': {}
        }

        model_count = len(available)
        for i, (model_name, model_path) in enumerate(available.items(), 1):
            print("\n" + "=" * 60)
            print(f"{i}/{model_count} 评估 {model_name}")
            print("=" * 60)

            # 加载模型并评估
            if model_name == 'MLP':
                model = self.load_mlp()
                val, _ = self.evaluate_model(model, self.X_val, self.y_val, None, model_name)
                test, _ = self.evaluate_model(model, self.X_test, self.y_test, None, model_name)
            elif model_name == 'MLP_KG':
                model = self.load_kg_mlp_v1()
                val, _ = self.evaluate_model(model, self.X_val, self.y_val, self.kg_val_emb_mlp, model_name)
                test, _ = self.evaluate_model(model, self.X_test, self.y_test, self.kg_test_emb_mlp, model_name)
            elif model_name == 'MLP_KG_V2':
                model = self.load_kg_mlp_v2()
                val, _ = self.evaluate_model(model, self.X_val, self.y_val, self.kg_val_emb_mlp, model_name)
                test, _ = self.evaluate_model(model, self.X_test, self.y_test, self.kg_test_emb_mlp, model_name)
            elif model_name == 'CNN':
                model = self.load_cnn()
                val, _ = self.evaluate_model(model, self.X_val, self.y_val, None, model_name)
                test, _ = self.evaluate_model(model, self.X_test, self.y_test, None, model_name)
            elif model_name == 'CNN_KG':
                model = self.load_cnn_kg()
                val, _ = self.evaluate_model(model, self.X_val, self.y_val, self.kg_val_emb, model_name)
                test, _ = self.evaluate_model(model, self.X_test, self.y_test, self.kg_test_emb, model_name)
            elif model_name == 'CNN_KG_V2':
                model = self.load_cnn_kg_v2()
                val, _ = self.evaluate_model(model, self.X_val, self.y_val, self.kg_val_emb, model_name)
                test, _ = self.evaluate_model(model, self.X_test, self.y_test, self.kg_test_emb, model_name)
            elif model_name == 'CNN_KG_V3':
                model = self.load_cnn_kg_v3()
                val, _ = self.evaluate_model(model, self.X_val, self.y_val, self.kg_val_emb, model_name)
                test, _ = self.evaluate_model(model, self.X_test, self.y_test, self.kg_test_emb, model_name)
            elif model_name == 'GNN':
                model = self.load_gnn()
                val, _ = self.evaluate_model(model, self.X_val, self.y_val, None, model_name)
                test, _ = self.evaluate_model(model, self.X_test, self.y_test, None, model_name)
            elif model_name == 'GNN_KG':
                model = self.load_gnn_kg()
                val, _ = self.evaluate_model(model, self.X_val, self.y_val, self.kg_val_emb, model_name)
                test, _ = self.evaluate_model(model, self.X_test, self.y_test, self.kg_test_emb, model_name)

            results['validation'][model_name] = val
            results['test'][model_name] = test
            print(f"{model_name}:")
            print(f"  验证集 - 准确率: {val['accuracy']:.4f}, F1: {val['f1']:.4f}")
            print(f"  测试集 - 准确率: {test['accuracy']:.4f}, F1: {test['f1']:.4f}")

        # 保存结果
        with open('results/evaluation_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        # 打印汇总表格
        print("\n" + "=" * 60)
        print("评估结果汇总")
        print("=" * 60)
        print(f"{'模型':<25} | {'验证集准确率':<12} | {'测试集准确率':<12} | {'验证F1':<10} | {'测试F1':<10}")
        print("-" * 80)
        for model_name in ['MLP', 'MLP_KG', 'MLP_KG_V2', 'CNN', 'CNN_KG', 'CNN_KG_V2', 'GNN', 'GNN_KG']:
            if model_name in results['validation']:
                val = results['validation'][model_name]
                test = results['test'][model_name]
                print(f"{model_name:<25} | {val['accuracy']:.4f}      | {test['accuracy']:.4f}      | {val['f1']:.4f}     | {test['f1']:.4f}")

        print(f"\n详细结果已保存至: results/evaluation_results.json")

        return results


def main():
    evaluator = Evaluator()
    evaluator.run()


if __name__ == '__main__':
    main()