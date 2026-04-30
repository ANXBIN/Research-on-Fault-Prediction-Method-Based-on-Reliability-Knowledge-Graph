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
from tqdm import tqdm

from src.data.loader import load_and_split_data
from src.models.mlp_model import MLPModel, KGEnhancedMLPV2Model
from src.models.cnn_model import CNNModel, CNNKGModelV3
from src.models.gnn_model import GNNModel, GNNKGModel


class EarlyStopping:
    """早停机制 - 验证集loss不再下降时停止训练"""

    def __init__(self, patience=30, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.best_state = None
        self.should_stop = False

    def step(self, val_loss, model_state):
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
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


class Trainer:
    def __init__(self):
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        print(f"[INFO] 使用设备: {self.device}")

        Path('models').mkdir(exist_ok=True)
        Path('results').mkdir(exist_ok=True)

        with open('config.yaml', 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        self.es_patience = self.config.get('training', {}).get('early_stopping_patience', 30)

        self._load_data()

        # 贝叶斯优化得出的最佳配置
        self.best_config = {
            'hidden_dim': 128,
            'dropout': 0.215,
            'lr': 0.0064,
            'weight_decay': 0.00021
        }

    def _load_data(self):
        """使用共享模块加载数据"""
        print("\n" + "=" * 60)
        print("加载数据")
        print("=" * 60)

        data = load_and_split_data()

        self.X_train = data['X_train']
        self.X_val = data['X_val']
        self.X_test = data['X_test']
        self.y_train = data['y_train']
        self.y_val = data['y_val']
        self.y_test = data['y_test']
        self.fault_types = data['fault_types']
        self.fault_to_idx = data['fault_to_idx']
        self.label_encoder = data['label_encoder']
        self.scaler = data['scaler']
        self.kg_train_emb = data['kg_train_emb']
        self.kg_val_emb = data['kg_val_emb']
        self.kg_test_emb = data['kg_test_emb']
        self.kg_train_emb_mlp = data['kg_train_emb_mlp']
        self.kg_val_emb_mlp = data['kg_val_emb_mlp']
        self.kg_test_emb_mlp = data['kg_test_emb_mlp']

        print(f"数据集大小: {len(self.X_train) + len(self.X_val) + len(self.X_test)} 样本")
        print(f"故障类型数: {len(self.fault_types)}")
        print(f"训练集: {len(self.X_train)} 样本")
        print(f"验证集: {len(self.X_val)} 样本")
        print(f"测试集: {len(self.X_test)} 样本")
        print(f"V2 KG嵌入维度: {self.kg_train_emb.shape}")
        print(f"MLP KNN嵌入维度: {self.kg_train_emb_mlp.shape}")

    def _train_model(self, model, train_fn, val_fn, epochs, desc, early_stop):
        """通用训练循环"""
        model.set_class_weights(self.y_train)
        pbar = tqdm(range(epochs), desc=desc)
        for epoch in pbar:
            train_loss, train_acc = train_fn()
            pbar.set_postfix({'loss': f'{train_loss:.4f}', 'acc': f'{train_acc:.4f}'})

            if early_stop:
                val_m, _ = val_fn()
                if early_stop.step(val_m['loss'], model.model.state_dict()):
                    print(f"  早停触发 (epoch {epoch+1})，恢复最佳模型")
                    early_stop.restore_best(model.model)
                    break

        val_metrics, _ = val_fn()
        return val_metrics

    def train_mlp(self, epochs=100):
        """训练普通MLP"""
        print("\n" + "=" * 60)
        print("训练 MLP 模型 (传统方法)")
        print("=" * 60)

        model = MLPModel(config_path='config.yaml')
        model.fault_to_idx = self.fault_to_idx
        model.build_model(self.X_train.shape[1], len(self.fault_types))

        early_stop = EarlyStopping(patience=self.es_patience)
        val_metrics = self._train_model(
            model,
            train_fn=lambda: model.train_epoch(self.X_train, self.y_train),
            val_fn=lambda: model.evaluate(self.X_val, self.y_val),
            epochs=epochs, desc="MLP Training", early_stop=early_stop
        )

        print(f"MLP 验证集准确率: {val_metrics['accuracy']:.4f}")
        model.save_model('models/mlp_model.pt')
        return model, val_metrics

    def train_mlp_kg(self, config=None, epochs=100):
        """训练MLP-KG-V2模型 - 门控融合架构"""
        if config is None:
            config = self.best_config

        print("\n" + "=" * 60)
        print("训练 MLP-KG-V2 模型 (门控融合)")
        print("=" * 60)
        print(f"配置: hidden_dim={config['hidden_dim']}, dropout={config['dropout']:.3f}, lr={config['lr']:.6f}")

        model = KGEnhancedMLPV2Model(config_path='config.yaml')
        model.hidden_dim = config['hidden_dim']
        model.dropout = config['dropout']
        model.learning_rate = config['lr']
        model.kg_embedding_dim = 64
        model.fault_to_idx = self.fault_to_idx
        model.build_model(self.X_train.shape[1], len(self.fault_types))

        model.optimizer = torch.optim.Adam(
            model.model.parameters(),
            lr=config['lr'],
            weight_decay=config['weight_decay']
        )

        early_stop = EarlyStopping(patience=self.es_patience)
        val_metrics = self._train_model(
            model,
            train_fn=lambda: model.train_epoch(self.X_train, self.y_train, self.kg_train_emb_mlp),
            val_fn=lambda: model.evaluate(self.X_val, self.y_val, self.kg_val_emb_mlp),
            epochs=epochs, desc="MLP-KG-V2 Training", early_stop=early_stop
        )

        print(f"MLP-KG 验证集准确率: {val_metrics['accuracy']:.4f}")
        model.save_model('models/mlp_kg_model.pt')
        return model, val_metrics

    def train_cnn(self, epochs=100):
        """训练CNN模型"""
        print("\n" + "=" * 60)
        print("训练 CNN 模型 (1D卷积神经网络)")
        print("=" * 60)

        model = CNNModel(config_path='config.yaml')
        model.fault_to_idx = self.fault_to_idx
        model.build_model(self.X_train.shape[1], len(self.fault_types))

        early_stop = EarlyStopping(patience=self.es_patience)
        val_metrics = self._train_model(
            model,
            train_fn=lambda: model.train_epoch(self.X_train, self.y_train),
            val_fn=lambda: model.evaluate(self.X_val, self.y_val),
            epochs=epochs, desc="CNN Training", early_stop=early_stop
        )

        print(f"CNN 验证集准确率: {val_metrics['accuracy']:.4f}")
        model.save_model('models/cnn_model.pt')
        return model, val_metrics

    def train_cnn_kg(self, epochs=100):
        """训练CNN + KG融合模型 V3 (残差连接)"""
        print("\n" + "=" * 60)
        print("训练 CNN-KG V3 融合模型 (残差连接)")
        print("=" * 60)

        model = CNNKGModelV3(config_path='config.yaml')
        model.fault_to_idx = self.fault_to_idx
        model.build_model(self.X_train.shape[1], len(self.fault_types))

        early_stop = EarlyStopping(patience=self.es_patience)
        val_metrics = self._train_model(
            model,
            train_fn=lambda: model.train_epoch(self.X_train, self.y_train, self.kg_train_emb),
            val_fn=lambda: model.evaluate(self.X_val, self.y_val, self.kg_val_emb),
            epochs=epochs, desc="CNN-KG V3 Training", early_stop=early_stop
        )

        print(f"CNN-KG 验证集准确率: {val_metrics['accuracy']:.4f}")
        model.save_model('models/cnn_kg_model.pt')
        return model, val_metrics

    def train_gnn(self, epochs=100):
        """训练GNN模型"""
        print("\n" + "=" * 60)
        print("训练 GNN 模型 (图神经网络)")
        print("=" * 60)

        model = GNNModel(config_path='config.yaml')
        model.fault_to_idx = self.fault_to_idx
        model.build_model(self.X_train.shape[1], len(self.fault_types))

        early_stop = EarlyStopping(patience=self.es_patience)
        val_metrics = self._train_model(
            model,
            train_fn=lambda: model.train_epoch(self.X_train, self.y_train),
            val_fn=lambda: model.evaluate(self.X_val, self.y_val),
            epochs=epochs, desc="GNN Training", early_stop=early_stop
        )

        print(f"GNN 验证集准确率: {val_metrics['accuracy']:.4f}")
        model.save_model('models/gnn_model.pt')
        return model, val_metrics

    def train_gnn_kg(self, epochs=100):
        """训练GNN + KG融合模型"""
        print("\n" + "=" * 60)
        print("训练 GNN-KG 融合模型")
        print("=" * 60)

        model = GNNKGModel(config_path='config.yaml')
        model.fault_to_idx = self.fault_to_idx
        model.build_model(self.X_train.shape[1], len(self.fault_types))

        early_stop = EarlyStopping(patience=self.es_patience)
        val_metrics = self._train_model(
            model,
            train_fn=lambda: model.train_epoch(self.X_train, self.y_train, self.kg_train_emb),
            val_fn=lambda: model.evaluate(self.X_val, self.y_val, self.kg_val_emb),
            epochs=epochs, desc="GNN-KG Training", early_stop=early_stop
        )

        print(f"GNN-KG 验证集准确率: {val_metrics['accuracy']:.4f}")
        model.save_model('models/gnn_kg_model.pt')
        return model, val_metrics

    def save_results(self, results):
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
            _, mlp_val = self.train_mlp(epochs=epochs)
            results['models']['MLP'] = {'val_accuracy': float(mlp_val['accuracy'])}

        if train_mlp_kg:
            _, mlp_kg_val = self.train_mlp_kg(epochs=epochs)
            results['models']['MLP_KG'] = {'val_accuracy': float(mlp_kg_val['accuracy'])}

        if train_cnn:
            _, cnn_val = self.train_cnn(epochs=epochs)
            results['models']['CNN'] = {'val_accuracy': float(cnn_val['accuracy'])}

        if train_cnn_kg:
            _, cnn_kg_val = self.train_cnn_kg(epochs=epochs)
            results['models']['CNN_KG'] = {'val_accuracy': float(cnn_kg_val['accuracy'])}

        if train_gnn:
            _, gnn_val = self.train_gnn(epochs=epochs)
            results['models']['GNN'] = {'val_accuracy': float(gnn_val['accuracy'])}

        if train_gnn_kg:
            _, gnn_kg_val = self.train_gnn_kg(epochs=epochs)
            results['models']['GNN_KG'] = {'val_accuracy': float(gnn_kg_val['accuracy'])}

        self.save_results(results)

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
