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
import argparse
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from src.models.mlp_model import MLPModel, KGEnhancedMLPModel, KGEnhancedMLPV2Model, load_kg_embeddings_v3, load_kg_embeddings_v4


class Trainer:
    def __init__(self):
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        print(f"[INFO] 使用设备: {self.device}")

        # 创建目录
        Path('models').mkdir(exist_ok=True)
        Path('results').mkdir(exist_ok=True)

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

        print(f"训练集: {len(self.X_train)} 样本")
        print(f"验证集: {len(self.X_val)} 样本")
        print(f"测试集: {len(self.X_test)} 样本")
        print(f"V1 KG嵌入维度: {self.kg_train_emb_v1.shape}")
        print(f"V2 KG嵌入维度: {self.kg_train_emb.shape}")

    def train_mlp(self):
        """训练普通MLP"""
        print("\n" + "=" * 60)
        print("训练 MLP 模型 (传统方法)")
        print("=" * 60)

        model = MLPModel(config_path='config.yaml')
        model.fault_to_idx = self.fault_to_idx
        model.build_model(self.X_train.shape[1], len(self.fault_types))

        # 进度条
        pbar = tqdm(range(model.epochs), desc="MLP Training")
        for epoch in pbar:
            train_loss, train_acc = model.train_epoch(self.X_train, self.y_train)
            pbar.set_postfix({'loss': f'{train_loss:.4f}', 'acc': f'{train_acc:.4f}'})

        # 验证集评估
        val_metrics, _ = model.evaluate(self.X_val, self.y_val)
        print(f"MLP 验证集准确率: {val_metrics['accuracy']:.4f}")

        # 保存模型
        model.save_model('models/mlp_model.pt')

        return model, val_metrics

    def train_kg_mlp_v1(self, config=None):
        """训练KG增强MLP V1 - 使用全局KG嵌入"""
        print("\n" + "=" * 60)
        print("训练 KG-Enhanced MLP 模型 (知识图谱增强 V1) - 全局嵌入")
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

        # 进度条
        pbar = tqdm(range(100), desc="KG-MLP V1 Training")
        for epoch in pbar:
            train_loss, train_acc = model.train_epoch(self.X_train, self.y_train, self.kg_train_emb_v1)
            pbar.set_postfix({'loss': f'{train_loss:.4f}', 'acc': f'{train_acc:.4f}'})

        # 验证集评估
        val_metrics, _ = model.evaluate(self.X_val, self.y_val, self.kg_val_emb_v1)
        print(f"KG-MLP V1 验证集准确率: {val_metrics['accuracy']:.4f}")

        # 保存模型
        model.save_model('models/kg_enhanced_mlp_v1_model.pt')

        return model, val_metrics

    def train_kg_mlp_v2(self, config=None, epochs=100, verbose=True):
        """训练KG增强MLP V2 - 使用故障级别嵌入（33维）

        Args:
            config: 超参数字典，None则使用best_config
            epochs: 训练轮数
            verbose: 是否打印详细信息
        """
        if config is None:
            config = self.best_config

        if verbose:
            print("\n" + "=" * 60)
            print("训练 KG-Enhanced MLP 模型 (故障级别嵌入 V2)")
            print("=" * 60)
            print(f"配置: hidden_dim={config['hidden_dim']}, dropout={config['dropout']:.3f}, lr={config['lr']:.6f}")

        model = KGEnhancedMLPV2Model(config_path='config.yaml')
        model.hidden_dim = config['hidden_dim']
        model.dropout = config['dropout']
        model.learning_rate = config['lr']
        model.kg_embedding_dim = 33  # V2使用33维故障级别嵌入
        model.fault_to_idx = self.fault_to_idx
        model.build_model(self.X_train.shape[1], len(self.fault_types))

        model.optimizer = torch.optim.Adam(
            model.model.parameters(),
            lr=config['lr'],
            weight_decay=config['weight_decay']
        )

        # 进度条
        if verbose:
            pbar = tqdm(range(epochs), desc="KG-MLP V2 Training")
        else:
            pbar = range(epochs)

        for epoch in pbar:
            train_loss, train_acc = model.train_epoch(self.X_train, self.y_train, self.kg_train_emb)
            if verbose:
                pbar.set_postfix({'loss': f'{train_loss:.4f}', 'acc': f'{train_acc:.4f}'})

        # 验证集评估
        val_metrics, _ = model.evaluate(self.X_val, self.y_val, self.kg_val_emb)

        if verbose:
            print(f"KG-MLP V2 验证集准确率: {val_metrics['accuracy']:.4f}")

        return model, val_metrics

    def save_results(self, results):
        """保存训练结果"""
        with open('results/training_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存至: results/training_results.json")

    def run(self, train_mlp=True, train_kg_v1=True, train_kg_v2=True):
        """运行完整训练流程"""
        print("\n" + "=" * 60)
        print("开始训练流程")
        print("=" * 60)
        print(f"训练选项: MLP={train_mlp}, KG-V1={train_kg_v1}, KG-V2={train_kg_v2}")

        results = {
            'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'best_config': self.best_config,
            'models': {}
        }

        if train_mlp:
            mlp_model, mlp_val = self.train_mlp()
            results['models']['MLP'] = {'val_accuracy': float(mlp_val['accuracy'])}

        if train_kg_v1:
            kg_mlp_v1, kg_mlp_v1_val = self.train_kg_mlp_v1()
            results['models']['KG_Enhanced_MLP_V1'] = {'val_accuracy': float(kg_mlp_v1_val['accuracy'])}

        if train_kg_v2:
            kg_mlp_v2, kg_mlp_v2_val = self.train_kg_mlp_v2()
            results['models']['KG_Enhanced_MLP_V2'] = {'val_accuracy': float(kg_mlp_v2_val['accuracy'])}

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

    def __init__(self, trainer, n_trials=30, timeout_per_trial=120):
        """
        Args:
            trainer: Trainer实例
            n_trials: 试验次数
            timeout_per_trial: 每个试验超时时间（秒）
        """
        self.trainer = trainer
        self.n_trials = n_trials
        self.timeout_per_trial = timeout_per_trial
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
        print(f"试验次数: {self.n_trials}, 每试验超时: {self.timeout_per_trial}秒")
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
            timeout=self.timeout_per_trial,
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

        # 保存模型
        model.save_model('models/kg_enhanced_mlp_v2_model.pt')

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


def main():
    parser = argparse.ArgumentParser(description='训练故障预测模型')
    parser.add_argument('--mlp', action='store_true', help='训练MLP模型')
    parser.add_argument('--kg-v1', action='store_true', help='训练KG-MLP V1模型 (全局嵌入)')
    parser.add_argument('--kg-v2', action='store_true', help='训练KG-MLP V2模型 (故障级别嵌入)')
    parser.add_argument('--tune-v2', action='store_true', help='对V2模型进行贝叶斯优化调优')
    parser.add_argument('--n-trials', type=int, default=30, help='调优试验次数 (默认30)')
    parser.add_argument('--timeout', type=int, default=180, help='每试验超时秒数 (默认180)')
    parser.add_argument('--all', action='store_true', help='训练所有模型')
    args = parser.parse_args()

    # 调优模式
    if args.tune_v2:
        print("=" * 60)
        print("V2模型调优模式")
        print("=" * 60)

        trainer = Trainer()
        tuner = V2Tuner(trainer, n_trials=args.n_trials, timeout_per_trial=args.timeout)
        best_params, best_value = tuner.tune()

        # 使用最佳参数训练最终模型
        model, val_metrics = tuner.train_best()
        tuner.save_best_params()

        print(f"\n最终模型验证集准确率: {val_metrics['accuracy']:.4f}")
        return

    # 常规训练模式
    if args.all:
        train_mlp = train_kg_v1 = train_kg_v2 = True
    else:
        train_mlp = args.mlp or (not (args.kg_v1 or args.kg_v2))
        train_kg_v1 = args.kg_v1
        train_kg_v2 = args.kg_v2

    if not (train_mlp or train_kg_v1 or train_kg_v2):
        print("请选择要训练的模型，使用 --mlp, --kg-v1, --kg-v2, --tune-v2 或 --all")
        return

    trainer = Trainer()
    trainer.run(train_mlp=train_mlp, train_kg_v1=train_kg_v1, train_kg_v2=train_kg_v2)


if __name__ == '__main__':
    main()