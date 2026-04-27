#!/usr/bin/env python3
"""
统一训练脚本 - 训练MLP、CNN、GNN模型及其KG增强版本
支持选择性训练：--mlp, --mlp-kg, --cnn, --cnn-kg, --gnn, --gnn-kg
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

from src.models.mlp_model import MLPModel, KGEnhancedMLPV2Model, load_kg_embeddings_v4, load_kg_embeddings_mlp
from src.models.cnn_model import CNNModel, CNNKGModelV3
from src.models.gnn_model import GNNModel, GNNKGModel


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

    def train_mlp_kg(self, config=None, epochs=100, verbose=True, use_early_stopping=True):
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
            print(f"MLP-KG 验证集准确率: {val_metrics['accuracy']:.4f}")
            model.save_model('models/mlp_kg_model.pt')

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
            print(f"CNN-KG 验证集准确率: {val_metrics['accuracy']:.4f}")
        model.save_model('models/cnn_kg_model.pt')

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

    def save_results(self, results):
        """保存训练结果"""
        with open('results/training_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存至: results/training_results.json")

    def run(self, train_mlp=True, train_mlp_kg=True, train_cnn=True, train_cnn_kg=True, train_gnn=True, train_gnn_kg=True, epochs=100):
        """运行完整训练流程"""
        print("\n" + "=" * 60)
        print("开始训练流程")
        print("=" * 60)

        results = {
            'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'best_config': self.best_config,
            'models': {}
        }

        if train_mlp:
            mlp_model, mlp_val = self.train_mlp(epochs=epochs)
            results['models']['MLP'] = {'val_accuracy': float(mlp_val['accuracy'])}

        if train_mlp_kg:
            mlp_kg_model, mlp_kg_val = self.train_mlp_kg(epochs=epochs)
            results['models']['MLP_KG'] = {'val_accuracy': float(mlp_kg_val['accuracy'])}

        if train_cnn:
            cnn_model, cnn_val = self.train_cnn(epochs=epochs)
            results['models']['CNN'] = {'val_accuracy': float(cnn_val['accuracy'])}

        if train_cnn_kg:
            cnn_kg_model, cnn_kg_val = self.train_cnn_kg(epochs=epochs)
            results['models']['CNN_KG'] = {'val_accuracy': float(cnn_kg_val['accuracy'])}

        if train_gnn:
            gnn_model, gnn_val = self.train_gnn(epochs=epochs)
            results['models']['GNN'] = {'val_accuracy': float(gnn_val['accuracy'])}

        if train_gnn_kg:
            gnn_kg_model, gnn_kg_val = self.train_gnn_kg(epochs=epochs)
            results['models']['GNN_KG'] = {'val_accuracy': float(gnn_kg_val['accuracy'])}

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




def main():
    parser = argparse.ArgumentParser(description='训练故障预测模型')
    parser.add_argument('--mlp', action='store_true', help='训练MLP模型')
    parser.add_argument('--mlp-kg', action='store_true', help='训练MLP-KG模型 (KNN嵌入)')
    parser.add_argument('--cnn', action='store_true', help='训练CNN模型')
    parser.add_argument('--cnn-kg', action='store_true', help='训练CNN-KG模型')
    parser.add_argument('--gnn', action='store_true', help='训练GNN模型')
    parser.add_argument('--gnn-kg', action='store_true', help='训练GNN-KG融合模型')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数 (默认100)')
    parser.add_argument('--all', action='store_true', help='训练所有模型')
    args = parser.parse_args()

    # 常规训练模式
    if args.all:
        train_mlp = train_mlp_kg = train_cnn = train_cnn_kg = train_gnn = train_gnn_kg = True
    else:
        any_model_specified = (args.mlp_kg or args.cnn or args.cnn_kg or args.gnn or args.gnn_kg)
        train_mlp = args.mlp or not any_model_specified
        train_mlp_kg = args.mlp_kg
        train_cnn = args.cnn
        train_cnn_kg = args.cnn_kg
        train_gnn = args.gnn
        train_gnn_kg = args.gnn_kg

    if not (train_mlp or train_mlp_kg or train_cnn or train_cnn_kg or train_gnn or train_gnn_kg):
        print("请选择要训练的模型")
        return

    trainer = Trainer()
    trainer.run(train_mlp=train_mlp, train_mlp_kg=train_mlp_kg, train_cnn=train_cnn, train_cnn_kg=train_cnn_kg, train_gnn=train_gnn, train_gnn_kg=train_gnn_kg, epochs=args.epochs)


if __name__ == '__main__':
    main()