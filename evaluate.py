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
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from src.data.loader import load_and_split_data
from src.models.mlp_model import MLPModel, KGEnhancedMLPV2Model
from src.models.cnn_model import CNNModel, CNNKGModelV3
from src.models.gnn_model import GNNModel, GNNKGModel


class Evaluator:
    def __init__(self):
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        print(f"[INFO] 使用设备: {self.device}")

        self.best_config = {
            'hidden_dim': 128,
            'dropout': 0.215,
            'lr': 0.0064,
            'weight_decay': 0.00021
        }

        self._load_data()

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
        self.kg_train_emb = data['kg_train_emb']
        self.kg_val_emb = data['kg_val_emb']
        self.kg_test_emb = data['kg_test_emb']
        self.kg_train_emb_mlp = data['kg_train_emb_mlp']
        self.kg_val_emb_mlp = data['kg_val_emb_mlp']
        self.kg_test_emb_mlp = data['kg_test_emb_mlp']

        print(f"训练集: {len(self.X_train)} 样本")
        print(f"验证集: {len(self.X_val)} 样本")
        print(f"测试集: {len(self.X_test)} 样本")

    def get_available_models(self):
        """检测models文件夹中可用的模型"""
        model_files = {
            'MLP': 'mlp_model.pt',
            'MLP_KG': 'mlp_kg_model.pt',
            'CNN': 'cnn_model.pt',
            'CNN_KG': 'cnn_kg_model.pt',
            'GNN': 'gnn_model.pt',
            'GNN_KG': 'gnn_kg_model.pt',
        }
        models_dir = Path('models')
        return {name: models_dir / fname for name, fname in model_files.items() if (models_dir / fname).exists()}

    def _load_model(self, model_name):
        """统一加载模型"""
        model_map = {
            'MLP': (MLPModel, {}),
            'MLP_KG': (KGEnhancedMLPV2Model, {'kg_embedding_dim': 64}),
            'CNN': (CNNModel, {}),
            'CNN_KG': (CNNKGModelV3, {'kg_embedding_dim': 33}),
            'GNN': (GNNModel, {}),
            'GNN_KG': (GNNKGModel, {'kg_embedding_dim': 33}),
        }
        path_map = {
            'MLP': 'mlp_model.pt', 'MLP_KG': 'mlp_kg_model.pt',
            'CNN': 'cnn_model.pt', 'CNN_KG': 'cnn_kg_model.pt',
            'GNN': 'gnn_model.pt', 'GNN_KG': 'gnn_kg_model.pt',
        }

        cls, extra_kwargs = model_map[model_name]
        checkpoint = torch.load(f"models/{path_map[model_name]}", map_location=self.device, weights_only=True)
        saved_config = checkpoint.get('config', {})

        model = cls(config_path='config.yaml')
        for key, val in extra_kwargs.items():
            setattr(model, key, saved_config.get(key, val))
        for key in ['hidden_dim', 'dropout', 'num_layers', 'batch_size']:
            if key in saved_config:
                setattr(model, key, saved_config[key])
        if 'learning_rate' in saved_config:
            model.learning_rate = saved_config['learning_rate']

        model.fault_to_idx = self.fault_to_idx
        model.build_model(self.X_train.shape[1], len(self.fault_types))
        model.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"[INFO] {model_name} 模型已加载")
        return model

    def evaluate_model(self, model, X, y, kg_emb=None):
        """评估单个模型"""
        if kg_emb is not None:
            _, y_pred = model.evaluate(X, y, kg_emb)
        else:
            _, y_pred = model.evaluate(X, y)

        acc = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred, average='weighted')
        return {'accuracy': acc, 'f1': f1}, y_pred

    def plot_confusion_matrix(self, y_true, y_pred, model_name, save_path='results/figures'):
        """生成并保存混淆矩阵图"""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        Path(save_path).mkdir(parents=True, exist_ok=True)

        cm = confusion_matrix(y_true, y_pred)
        labels = self.label_encoder.classes_

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.figure.colorbar(im, ax=ax)

        ax.set(xticks=np.arange(len(labels)),
               yticks=np.arange(len(labels)),
               xticklabels=labels, yticklabels=labels,
               title=f'{model_name} 混淆矩阵',
               ylabel='真实标签',
               xlabel='预测标签')

        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

        thresh = cm.max() / 2.
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, format(cm[i, j], 'd'),
                        ha='center', va='center',
                        color='white' if cm[i, j] > thresh else 'black')

        fig.tight_layout()
        filepath = Path(save_path) / f'{model_name.lower().replace(" ", "_")}_confusion_matrix.png'
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  混淆矩阵已保存: {filepath}")

    def _get_kg_emb(self, model_name, split='test'):
        """根据模型名和数据集获取对应的KG嵌入"""
        if model_name == 'MLP_KG':
            return self.kg_test_emb_mlp if split == 'test' else self.kg_val_emb_mlp
        elif model_name in ('CNN_KG', 'GNN_KG'):
            return self.kg_test_emb if split == 'test' else self.kg_val_emb
        return None

    def run(self):
        """运行评估流程"""
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

            model = self._load_model(model_name)
            val_kg = self._get_kg_emb(model_name, 'val')
            test_kg = self._get_kg_emb(model_name, 'test')

            val, _ = self.evaluate_model(model, self.X_val, self.y_val, val_kg)
            test, _ = self.evaluate_model(model, self.X_test, self.y_test, test_kg)

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
        best_model_name = None
        best_test_acc = 0
        for model_name in ['MLP', 'MLP_KG', 'CNN', 'CNN_KG', 'GNN', 'GNN_KG']:
            if model_name in results['validation']:
                val = results['validation'][model_name]
                test = results['test'][model_name]
                print(f"{model_name:<25} | {val['accuracy']:.4f}      | {test['accuracy']:.4f}      | {val['f1']:.4f}     | {test['f1']:.4f}")
                if test['accuracy'] > best_test_acc:
                    best_test_acc = test['accuracy']
                    best_model_name = model_name

        # 为最佳模型生成混淆矩阵
        if best_model_name and best_model_name in available:
            print(f"\n为最佳模型 {best_model_name} 生成混淆矩阵...")
            model = self._load_model(best_model_name)
            test_kg = self._get_kg_emb(best_model_name, 'test')
            _, y_pred = self.evaluate_model(model, self.X_test, self.y_test, test_kg)
            self.plot_confusion_matrix(self.y_test, y_pred, best_model_name.replace('_', ' '))

        print(f"\n详细结果已保存至: results/evaluation_results.json")
        return results


def main():
    evaluator = Evaluator()
    evaluator.run()


if __name__ == '__main__':
    main()
