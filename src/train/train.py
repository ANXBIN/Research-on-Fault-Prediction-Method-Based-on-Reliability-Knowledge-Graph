#!/usr/bin/env python3
"""
故障预测训练主程序
对比实验：传统方法 vs 知识图谱增强方法
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from datetime import datetime
import torch
from torch_geometric.data import DataLoader

# 添加src目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.gnn_model import GNNModel, FaultGraphDataset
from models.rf_model import RFModel
from models.kg_enhanced_models import KGEnhancedRF, create_kg_enhanced_features
from models.mlp_model import MLPModel, KGEnhancedMLPModel, load_kg_embeddings_v3
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


class FaultPredictionTrainer:
    """故障预测训练器"""

    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.data_config = self.config['preprocessing']
        self.training_config = self.config['training']
        self.output_path = Path(self.config['project']['output_path'])
        self.output_path.mkdir(parents=True, exist_ok=True)

        self.results = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'models': {}
        }

        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[INFO] 使用设备: {self.device}")

    def load_data(self, data_path='data/processed/processed_features.csv'):
        """加载预处理后的数据"""
        print("\n" + "=" * 60)
        print("加载数据")
        print("=" * 60)

        df = pd.read_csv(data_path)
        print(f"数据集大小: {len(df)} 样本")

        # 分离特征和标签
        self.fault_types = df['fault_type'].unique()
        print(f"故障类型数: {len(self.fault_types)}")
        print(f"故障类型: {list(self.fault_types)}")

        # 编码标签
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.fault_types)
        y = self.label_encoder.transform(df['fault_type'])
        self.fault_to_idx = dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))
        self.idx_to_fault = {v: k for k, v in self.fault_to_idx.items()}

        # 提取特征（排除fault_type和channel列）
        feature_cols = [col for col in df.columns if col not in ['fault_type', 'channel']]
        X = df[feature_cols].values

        # 标准化
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)

        print(f"特征维度: {X.shape[1]}")

        # 划分数据集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.training_config['test_size'],
            random_state=self.training_config['random_seed'],
            stratify=y
        )

        # 进一步划分验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train,
            test_size=self.training_config['val_size'] / (1 - self.training_config['test_size']),
            random_state=self.training_config['random_seed'],
            stratify=y_train
        )

        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        self.feature_names = feature_cols

        print(f"\n训练集: {len(X_train)} 样本")
        print(f"验证集: {len(X_val)} 样本")
        print(f"测试集: {len(X_test)} 样本")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def train_gnn(self):
        """训练GNN模型"""
        print("\n" + "=" * 60)
        print("训练 GNN 模型")
        print("=" * 60)

        # 加载KG嵌入（如果有）
        kg_embed_path = 'data/processed/kg_embeddings.json'
        kg_adj = None
        if Path(kg_embed_path).exists():
            with open(kg_embed_path, 'r') as f:
                kg_data = json.load(f)
                kg_adj = np.array(kg_data.get('adjacency_matrix', []))

        # 创建图数据集
        train_dataset = FaultGraphDataset(
            self.X_train.tolist(), self.y_train.tolist(),
            self.fault_to_idx, kg_adj, self.idx_to_fault
        )

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        # 构建模型
        in_channels = self.X_train.shape[1]
        out_channels = len(self.fault_types)
        gnn_model = GNNModel(config_path='config.yaml')
        gnn_model.build_model(in_channels, out_channels)

        # 训练
        best_val_acc = 0
        patience = self.training_config['early_stopping_patience']
        patience_counter = 0

        for epoch in range(gnn_model.epochs):
            train_loss, train_acc = gnn_model.train_epoch(train_loader)

            if epoch % 10 == 0 or epoch == gnn_model.epochs - 1:
                print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}")

            # 早停
            if train_acc > 0.95 and patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
            patience_counter += 1

        # 保存结果
        self.results['models']['GNN'] = {
            'train_accuracy': float(train_acc),
            'epochs_trained': epoch + 1
        }

        return gnn_model

    def train_rf(self):
        """训练随机森林模型"""
        print("\n" + "=" * 60)
        print("训练 RF 模型 (传统方法)")
        print("=" * 60)

        rf_model = RFModel(config_path='config.yaml')
        rf_model.fault_to_idx = self.fault_to_idx

        # 构建和训练
        rf_model.build_model(self.X_train.shape[1], len(self.fault_types))
        rf_model.fit(self.X_train, self.y_train)

        # 评估
        train_metrics, _ = rf_model.evaluate(self.X_train, self.y_train)
        val_metrics, val_pred = rf_model.evaluate(self.X_val, self.y_val)

        print(f"训练集 - 准确率: {train_metrics['accuracy']:.4f}, F1: {train_metrics['f1']:.4f}")
        print(f"验证集 - 准确率: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}")

        # 特征重要性
        importance = rf_model.get_feature_importance(self.feature_names)
        top_features = list(importance.items())[:10]
        print("\nTop 10 重要特征:")
        for feat, imp in top_features:
            print(f"  {feat}: {imp:.4f}")

        # 保存模型
        model_path = self.output_path / 'models' / 'rf_model.joblib'
        model_path.parent.mkdir(parents=True, exist_ok=True)
        rf_model.save_model(str(model_path))

        self.results['models']['RF'] = {
            'train_accuracy': float(train_metrics['accuracy']),
            'train_f1': float(train_metrics['f1']),
            'val_accuracy': float(val_metrics['accuracy']),
            'val_f1': float(val_metrics['f1']),
            'top_features': top_features
        }

        return rf_model

    def train_kg_enhanced_rf(self):
        """训练知识图谱增强的RF模型"""
        print("\n" + "=" * 60)
        print("训练 KG-Enhanced RF 模型 (知识图谱增强)")
        print("=" * 60)

        # 加载KG嵌入
        kg_embed_path = 'data/processed/kg_embeddings.json'
        kg_features = None
        if Path(kg_embed_path).exists():
            kg_features = create_kg_enhanced_features(
                self.X_train, kg_embed_path
            )
            print(f"KG增强特征维度: {kg_features.shape}")
        else:
            print("[WARNING] KG嵌入文件不存在，使用原始特征")
            kg_features = self.X_train

        kg_rf_model = KGEnhancedRF(config_path='config.yaml')
        kg_rf_model.fault_to_idx = self.fault_to_idx

        # 构建和训练
        kg_rf_model.build_model(kg_features.shape[1], len(self.fault_types))
        kg_rf_model.fit(kg_features, self.y_train)

        # 评估
        train_metrics, _ = kg_rf_model.evaluate(kg_features, self.y_train)

        # 验证集
        kg_val_features = create_kg_enhanced_features(self.X_val, kg_embed_path) if kg_features is not None else self.X_val
        val_metrics, val_pred = kg_rf_model.evaluate(kg_val_features, self.y_val)

        print(f"训练集 - 准确率: {train_metrics['accuracy']:.4f}, F1: {train_metrics['f1']:.4f}")
        print(f"验证集 - 准确率: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}")

        # 保存模型
        model_path = self.output_path / 'models' / 'kg_enhanced_rf_model.joblib'
        kg_rf_model.save_model(str(model_path))

        self.results['models']['KG_Enhanced_RF'] = {
            'train_accuracy': float(train_metrics['accuracy']),
            'train_f1': float(train_metrics['f1']),
            'val_accuracy': float(val_metrics['accuracy']),
            'val_f1': float(val_metrics['f1'])
        }

        return kg_rf_model

    def train_mlp(self):
        """训练普通MLP模型"""
        print("\n" + "=" * 60)
        print("训练 MLP 模型 (传统方法)")
        print("=" * 60)

        mlp_model = MLPModel(config_path='config.yaml')
        mlp_model.fault_to_idx = self.fault_to_idx

        # 构建模型
        mlp_model.build_model(self.X_train.shape[1], len(self.fault_types))

        # 训练
        for epoch in range(mlp_model.epochs):
            train_loss, train_acc = mlp_model.train_epoch(self.X_train, self.y_train)

            if epoch % 20 == 0 or epoch == mlp_model.epochs - 1:
                print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}")

        # 评估
        train_metrics, _ = mlp_model.evaluate(self.X_train, self.y_train)
        val_metrics, val_pred = mlp_model.evaluate(self.X_val, self.y_val)

        print(f"训练集 - 准确率: {train_metrics['accuracy']:.4f}")
        print(f"验证集 - 准确率: {val_metrics['accuracy']:.4f}")

        # 保存模型
        model_path = self.output_path / 'models' / 'mlp_model.pt'
        model_path.parent.mkdir(parents=True, exist_ok=True)
        mlp_model.save_model(str(model_path))

        self.results['models']['MLP'] = {
            'train_accuracy': float(train_metrics['accuracy']),
            'train_f1': float(train_metrics.get('f1', train_metrics['accuracy'])),
            'val_accuracy': float(val_metrics['accuracy']),
            'val_f1': float(val_metrics.get('f1', val_metrics['accuracy']))
        }

        return mlp_model

    def train_kg_enhanced_mlp(self):
        """训练知识图谱增强MLP模型"""
        print("\n" + "=" * 60)
        print("训练 KG-Enhanced MLP 模型 (知识图谱增强)")
        print("=" * 60)

        kg_embed_path = 'data/processed/kg_embeddings.json'

        # 加载KG嵌入 - 使用修复版，基于特征KNN而非fault_labels
        from models.mlp_model import load_kg_embeddings_v3

        kg_train_emb = load_kg_embeddings_v3(
            kg_embed_path, len(self.X_train),
            None, self.X_train
        )
        kg_val_emb = load_kg_embeddings_v3(
            kg_embed_path, len(self.X_val),
            None, self.X_train, self.X_val
        )

        print(f"KG嵌入维度: {kg_train_emb.shape}")

        kg_mlp_model = KGEnhancedMLPModel(config_path='config.yaml')
        kg_mlp_model.fault_to_idx = self.fault_to_idx

        # 构建模型
        kg_mlp_model.build_model(self.X_train.shape[1], len(self.fault_types))

        # 训练
        for epoch in range(kg_mlp_model.epochs):
            train_loss, train_acc = kg_mlp_model.train_epoch(
                self.X_train, self.y_train, kg_train_emb
            )

            if epoch % 20 == 0 or epoch == kg_mlp_model.epochs - 1:
                print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}")

        # 评估
        train_metrics, _ = kg_mlp_model.evaluate(self.X_train, self.y_train, kg_train_emb)
        val_metrics, val_pred = kg_mlp_model.evaluate(self.X_val, self.y_val, kg_val_emb)

        print(f"训练集 - 准确率: {train_metrics['accuracy']:.4f}")
        print(f"验证集 - 准确率: {val_metrics['accuracy']:.4f}")

        # 保存模型
        model_path = self.output_path / 'models' / 'kg_enhanced_mlp_model.pt'
        kg_mlp_model.save_model(str(model_path))

        self.results['models']['KG_Enhanced_MLP'] = {
            'train_accuracy': float(train_metrics['accuracy']),
            'train_f1': float(train_metrics.get('f1', train_metrics['accuracy'])),
            'val_accuracy': float(val_metrics['accuracy']),
            'val_f1': float(val_metrics.get('f1', val_metrics['accuracy']))
        }

        return kg_mlp_model

    def evaluate_all_models(self):
        """在测试集上评估所有模型"""
        print("\n" + "=" * 60)
        print("测试集评估")
        print("=" * 60)

        results = {}

        # RF评估
        if 'RF' in self.results['models']:
            rf_path = self.output_path / 'models' / 'rf_model.joblib'
            if rf_path.exists():
                import joblib
                rf_data = joblib.load(rf_path)
                rf_model = rf_data['model']

                test_metrics = {}
                test_metrics['accuracy'] = rf_model.score(self.X_test, self.y_test)
                y_pred = rf_model.predict(self.X_test)
                test_metrics['precision'] = sum(y_pred == self.y_test) / len(self.y_test)

                from sklearn.metrics import f1_score
                test_metrics['f1'] = f1_score(self.y_test, y_pred, average='weighted')

                results['RF'] = test_metrics
                print(f"RF - 准确率: {test_metrics['accuracy']:.4f}, F1: {test_metrics['f1']:.4f}")

        # KG-Enhanced RF评估
        if 'KG_Enhanced_RF' in self.results['models']:
            kg_rf_path = self.output_path / 'models' / 'kg_enhanced_rf_model.joblib'
            if kg_rf_path.exists():
                import joblib
                kg_rf_data = joblib.load(kg_rf_path)
                kg_rf_model = kg_rf_data['model']

                kg_test_features = create_kg_enhanced_features(
                    self.X_test, 'data/processed/kg_embeddings.json'
                )

                test_metrics = {}
                test_metrics['accuracy'] = kg_rf_model.score(kg_test_features, self.y_test)
                y_pred = kg_rf_model.predict(kg_test_features)

                from sklearn.metrics import f1_score
                test_metrics['f1'] = f1_score(self.y_test, y_pred, average='weighted')

                results['KG_Enhanced_RF'] = test_metrics
                print(f"KG-Enhanced RF - 准确率: {test_metrics['accuracy']:.4f}, F1: {test_metrics['f1']:.4f}")

        # MLP评估
        if 'MLP' in self.results['models']:
            mlp_path = self.output_path / 'models' / 'mlp_model.pt'
            if mlp_path.exists():
                import torch
                from models.mlp_model import MLPModel

                mlp_model = MLPModel(config_path='config.yaml')
                mlp_model.build_model(self.X_test.shape[1], len(self.fault_types))
                checkpoint = torch.load(mlp_path, map_location=mlp_model.device)
                mlp_model.model.load_state_dict(checkpoint['model_state_dict'])

                test_metrics, _ = mlp_model.evaluate(self.X_test, self.y_test)
                from sklearn.metrics import f1_score
                y_pred = mlp_model.predict(self.X_test)
                test_metrics['f1'] = f1_score(self.y_test, y_pred, average='weighted')

                results['MLP'] = test_metrics
                print(f"MLP - 准确率: {test_metrics['accuracy']:.4f}, F1: {test_metrics['f1']:.4f}")

        # KG-Enhanced MLP评估
        if 'KG_Enhanced_MLP' in self.results['models']:
            kg_mlp_path = self.output_path / 'models' / 'kg_enhanced_mlp_model.pt'
            if kg_mlp_path.exists():
                import torch
                from models.mlp_model import KGEnhancedMLPModel, load_kg_embeddings_v3

                kg_mlp_model = KGEnhancedMLPModel(config_path='config.yaml')
                kg_mlp_model.build_model(self.X_test.shape[1], len(self.fault_types))
                checkpoint = torch.load(kg_mlp_path, map_location=kg_mlp_model.device)
                kg_mlp_model.model.load_state_dict(checkpoint['model_state_dict'])

                # 加载测试集KG嵌入
                kg_test_emb = load_kg_embeddings_v3(
                    'data/processed/kg_embeddings.json', len(self.X_test),
                    None, self.X_train, self.X_test
                )

                test_metrics, _ = kg_mlp_model.evaluate(self.X_test, self.y_test, kg_test_emb)
                from sklearn.metrics import f1_score
                y_pred = kg_mlp_model.predict(self.X_test, kg_test_emb)
                test_metrics['f1'] = f1_score(self.y_test, y_pred, average='weighted')

                results['KG_Enhanced_MLP'] = test_metrics
                print(f"KG-Enhanced MLP - 准确率: {test_metrics['accuracy']:.4f}, F1: {test_metrics['f1']:.4f}")

        self.results['test_results'] = results

        # 保存结果
        results_path = self.output_path / 'training_results.json'
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)

        print(f"\n结果已保存至: {results_path}")

        return results

    def save_results(self):
        """保存训练结果"""
        results_path = self.output_path / 'training_results.json'
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)

        print(f"[INFO] 结果已保存至: {results_path}")

    def compare_models(self):
        """模型对比"""
        print("\n" + "=" * 60)
        print("模型对比")
        print("=" * 60)

        print("\n| 模型 | 训练准确率 | 验证准确率 | F1分数 |")
        print("|------|-----------|-----------|--------|")

        for model_name, model_results in self.results['models'].items():
            train_acc = model_results.get('train_accuracy', 0)
            val_acc = model_results.get('val_accuracy', 0)
            f1 = model_results.get('val_f1', 0)
            print(f"| {model_name} | {train_acc:.4f} | {val_acc:.4f} | {f1:.4f} |")


def main():
    """主函数"""
    import argparse
    parser = argparse.ArgumentParser(description='故障预测模型训练')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='配置文件路径')
    parser.add_argument('--skip_gnn', action='store_true',
                       help='跳过GNN训练')
    parser.add_argument('--skip_rf', action='store_true',
                       help='跳过RF训练')
    parser.add_argument('--skip_kg', action='store_true',
                       help='跳过KG增强训练')
    parser.add_argument('--skip_mlp', action='store_true',
                       help='跳过MLP训练')
    parser.add_argument('--skip_kg_mlp', action='store_true',
                       help='跳过KG增强MLP训练')
    args = parser.parse_args()

    # 初始化训练器
    trainer = FaultPredictionTrainer(config_path=args.config)

    # 加载数据
    trainer.load_data()

    # 训练模型
    if not args.skip_gnn:
        trainer.train_gnn()

    if not args.skip_rf:
        trainer.train_rf()

    if not args.skip_kg:
        trainer.train_kg_enhanced_rf()

    if not args.skip_mlp:
        trainer.train_mlp()

    if not args.skip_kg_mlp:
        trainer.train_kg_enhanced_mlp()

    # 模型对比
    trainer.compare_models()

    # 测试集评估
    trainer.evaluate_all_models()

    # 保存结果
    trainer.save_results()

    print("\n" + "=" * 60)
    print("训练完成!")
    print("=" * 60)


if __name__ == '__main__':
    main()
