#!/usr/bin/env python3
"""
交互式故障预测与LLM解释脚本

用法:
  python predict.py --sample                    # 使用随机示例数据
  python predict.py --data "0.5,-1.2,..."      # 输入传感器数据
  python predict.py --model GNN-KG             # 指定模型
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np
import pandas as pd
import json
import requests
import argparse
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split

from src.models.mlp_model import MLPModel, KGEnhancedMLPV2Model, load_kg_embeddings_v4
from src.models.cnn_model import CNNModel, CNNKGModelV3
from src.models.gnn_model import GNNModel, GNNKGModel, build_batch_adjacency


class OllamaClient:
    """Ollama LLM 客户端"""

    def __init__(self, base_url="http://localhost:11434"):
        self.base_url = base_url
        self.model = "qwen3:4b"

    def is_available(self):
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except (requests.RequestException, ConnectionError):
            return False

    def generate(self, prompt):
        payload = {
            "model": self.model,
            "prompt": "/no_think\n" + prompt,
            "stream": False
        }
        try:
            response = requests.post(f"{self.base_url}/api/generate", json=payload)
            if response.status_code == 200:
                return response.json().get("response", "")
            return f"Error: {response.status_code}"
        except Exception as e:
            return f"Connection error: {str(e)}"


# 模型名 -> (类, checkpoint文件名, 额外配置字段)
MODEL_REGISTRY = {
    'MLP':    (MLPModel,            'mlp_model.pt',    {}),
    'MLP-KG': (KGEnhancedMLPV2Model, 'mlp_kg_model.pt', {'kg_embedding_dim': 64}),
    'CNN':    (CNNModel,             'cnn_model.pt',    {}),
    'CNN-KG': (CNNKGModelV3,         'cnn_kg_model.pt', {'kg_embedding_dim': 33}),
    'GNN':    (GNNModel,             'gnn_model.pt',    {}),
    'GNN-KG': (GNNKGModel,           'gnn_kg_model.pt', {'kg_embedding_dim': 33}),
}


class Predictor:
    """故障预测器"""

    def __init__(self, model_name='GNN-KG'):
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.ollama = OllamaClient()
        self.model_name = model_name
        self.model = None
        self.scaler = None
        self.n_features = 0
        self.feature_names = []

        self._load_data()
        self._load_model()

    def _load_data(self):
        """加载scaler、label_encoder和训练数据"""
        print("加载数据...")
        df = pd.read_csv('data/processed/processed_features.csv')

        feature_cols = [col for col in df.columns if col not in ['fault_type', 'channel']]
        self.feature_names = feature_cols
        self.n_features = len(feature_cols)

        # 加载训练阶段保存的scaler和label_encoder
        scaler_path = Path('models/scaler.joblib')
        le_path = Path('models/label_encoder.joblib')
        if scaler_path.exists() and le_path.exists():
            self.scaler = joblib.load(scaler_path)
            self.label_encoder = joblib.load(le_path)
        else:
            from sklearn.preprocessing import LabelEncoder
            y = df['fault_type'].values
            train_idx, _ = train_test_split(
                np.arange(len(df)), test_size=0.2, random_state=42, stratify=y
            )
            self.scaler = StandardScaler()
            self.scaler.fit(df[feature_cols].values[train_idx])
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(df['fault_type'].unique())

        self.fault_to_idx = dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))

        # 仅用训练集部分构建KNN索引
        y = df['fault_type'].values
        train_idx, _ = train_test_split(
            np.arange(len(df)), test_size=0.2, random_state=42, stratify=y
        )
        self.X_train_raw = df[feature_cols].values[train_idx]

        print(f"  特征数量: {self.n_features}")
        print(f"  故障类型: {len(self.label_encoder.classes_)}")
        print(f"  Ollama可用: {'是' if self.ollama.is_available() else '否'}")

    def _load_model(self):
        """加载指定模型"""
        if self.model_name not in MODEL_REGISTRY:
            print(f"[错误] 未知模型: {self.model_name}")
            return

        cls, filename, extra_kwargs = MODEL_REGISTRY[self.model_name]
        path = Path('models') / filename
        if not path.exists():
            print(f"[错误] 模型文件不存在: {path}")
            return

        print(f"加载模型: {self.model_name}...")
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        saved_config = checkpoint.get('config', {})

        self.model = cls(config_path='config.yaml')
        for key, default in extra_kwargs.items():
            setattr(self.model, key, saved_config.get(key, default))
        for key in ['hidden_dim', 'dropout', 'num_layers', 'batch_size', 'learning_rate']:
            if key in saved_config:
                setattr(self.model, key, saved_config[key])

        self.model.fault_to_idx = self.fault_to_idx
        self.model.build_model(self.n_features, len(self.label_encoder.classes_))
        self.model.model.load_state_dict(checkpoint['model_state_dict'])
        print("  模型已加载")

    def _get_kg_embedding(self, fault_type):
        """获取指定故障类型的KG嵌入"""
        try:
            kg_emb = load_kg_embeddings_v4(
                'data/processed/fault_embeddings.json',
                [fault_type],
                'data/processed/kg_embeddings.json'
            )
            return kg_emb[0]
        except (FileNotFoundError, KeyError, json.JSONDecodeError):
            return np.zeros(33)

    def _get_avg_kg_embedding(self):
        """获取所有故障类型的平均KG嵌入"""
        try:
            all_embs = load_kg_embeddings_v4(
                'data/processed/fault_embeddings.json',
                list(self.label_encoder.classes_),
                'data/processed/kg_embeddings.json'
            )
            return np.mean(all_embs, axis=0)
        except (FileNotFoundError, KeyError, json.JSONDecodeError):
            return np.zeros(33)

    def _compute_knn_embedding(self, X_sample, k=20):
        """为单个样本计算KNN嵌入（64维）"""
        X_train_norm = self.X_train_raw / (np.linalg.norm(self.X_train_raw, axis=1, keepdims=True) + 1e-8)
        X_sample_norm = X_sample.reshape(1, -1) / (np.linalg.norm(X_sample) + 1e-8)

        nn_model = NearestNeighbors(n_neighbors=min(k + 1, len(self.X_train_raw)), metric='euclidean')
        nn_model.fit(X_train_norm)
        distances, indices = nn_model.kneighbors(X_sample_norm)

        neighbor_indices = indices[0, 1:]
        neighbor_features = X_train_norm[neighbor_indices]
        neighbor_distances = distances[0, 1:]

        weights = 1.0 / np.maximum(neighbor_distances, 1e-8)
        weights = weights / weights.sum()
        neighbor_weighted = np.dot(weights, neighbor_features)
        diff = X_sample_norm[0] - neighbor_weighted

        n_feat = self.X_train_raw.shape[1]
        emb = np.zeros(64, dtype=np.float32)
        emb[:n_feat] = X_sample_norm[0]
        emb[n_feat:2 * n_feat] = neighbor_weighted
        emb[2 * n_feat:3 * n_feat] = diff
        return emb

    def predict(self, features):
        """预测故障类型，返回 (fault_name, probabilities)"""
        X_scaled = self.scaler.transform(np.array(features).reshape(1, -1))
        avg_kg_emb = self._get_avg_kg_embedding()

        self.model.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X_scaled, dtype=torch.float).to(self.model.device)

            if isinstance(self.model, GNNModel):
                adj = build_batch_adjacency(X_scaled, k=30).to(self.model.device)
                output = self.model.model(X_tensor, adj)
            elif isinstance(self.model, GNNKGModel):
                adj = build_batch_adjacency(X_scaled, k=30).to(self.model.device)
                kg_tensor = torch.tensor(avg_kg_emb.reshape(1, -1), dtype=torch.float).to(self.model.device)
                output = self.model.model(X_tensor, kg_tensor, adj)
            elif isinstance(self.model, KGEnhancedMLPV2Model):
                knn_emb = self._compute_knn_embedding(X_scaled[0], k=20)
                kg_tensor = torch.tensor(knn_emb.reshape(1, -1), dtype=torch.float).to(self.model.device)
                output = self.model.model(X_tensor, kg_tensor)
            elif isinstance(self.model, CNNKGModelV3):
                kg_tensor = torch.tensor(avg_kg_emb.reshape(1, -1), dtype=torch.float).to(self.model.device)
                output = self.model.model(X_tensor, kg_tensor)
            else:
                output = self.model.model(X_tensor)

            pred_idx = output.argmax(dim=1).item()
            fault_name = self.label_encoder.inverse_transform([pred_idx])[0]

            # 二阶段预测：GNN-KG用预测故障的具体嵌入重新推理
            if isinstance(self.model, GNNKGModel):
                specific_emb = self._get_kg_embedding(fault_name)
                kg_tensor = torch.tensor(specific_emb.reshape(1, -1), dtype=torch.float).to(self.model.device)
                output = self.model.model(X_tensor, kg_tensor, adj)
                pred_idx = output.argmax(dim=1).item()
                fault_name = self.label_encoder.inverse_transform([pred_idx])[0]

            probabilities = torch.softmax(output, dim=1).cpu().numpy()[0]

        return fault_name, probabilities

    def get_top_k_predictions(self, features, k=3):
        """获取Top-K预测"""
        fault_name, probabilities = self.predict(features)
        top_k_idx = np.argsort(probabilities)[::-1][:k]
        return [(self.label_encoder.inverse_transform([idx])[0], float(probabilities[idx])) for idx in top_k_idx]

    def get_llm_explanation(self, top3, fault_name):
        """获取LLM解释"""
        if not self.ollama.is_available():
            return "Ollama未运行，无法获取解释"

        prompt = f"""你是一个变速箱故障诊断专家，基于以下齿轮箱故障预测模型的预测结果，给出简洁的解释和建议。

预测故障类型: {fault_name}

Top-3预测及概率:
"""
        for name, prob in top3:
            prompt += f"- {name}: {prob:.1%}\n"

        prompt += """
请给出：
1. 该故障类型的主要特征
2. 可能的原因
3. 建议的维护措施

回答要简洁，控制在100字以内。"""

        return self.ollama.generate(prompt)


def main():
    parser = argparse.ArgumentParser(description='变速箱故障预测')
    parser.add_argument('--sample', action='store_true', help='使用随机示例数据')
    parser.add_argument('--data', type=str, help='传感器数据，用逗号分隔')
    parser.add_argument('--model', type=str, default='GNN-KG',
                        choices=list(MODEL_REGISTRY.keys()), help='选择模型')
    parser.add_argument('--no-llm', action='store_true', help='禁用LLM解释')
    parser.add_argument('--idx', type=int, help='指定样本索引')
    args = parser.parse_args()

    print("=" * 60)
    print("变速箱故障预测系统")
    print("=" * 60)

    predictor = Predictor(model_name=args.model)
    df = pd.read_csv('data/processed/processed_features.csv')

    if args.idx is not None:
        sample_idx = args.idx
        features = df.iloc[sample_idx][predictor.feature_names].values
        actual_fault = df.iloc[sample_idx]['fault_type']
        print(f"\n使用样本 {sample_idx} (实际故障: {actual_fault})")
    elif args.sample:
        sample_idx = np.random.randint(len(df))
        features = df.iloc[sample_idx][predictor.feature_names].values
        actual_fault = df.iloc[sample_idx]['fault_type']
        print(f"\n使用随机样本 {sample_idx} (实际故障: {actual_fault})")
    elif args.data:
        try:
            features = [float(x.strip()) for x in args.data.split(',')]
            if len(features) != predictor.n_features:
                print(f"[错误] 特征数量不匹配，需要{predictor.n_features}个数值")
                return
            actual_fault = None
        except ValueError:
            print("[错误] 请输入有效的数值")
            return
    else:
        print("请使用 --sample 或 --data 参数")
        return

    print("\n" + "=" * 60)
    print("预测结果")
    print("=" * 60)

    top3 = predictor.get_top_k_predictions(features, k=3)
    fault_name = top3[0][0]

    print(f"预测故障: {fault_name}")
    print(f"Top-3预测:")
    for i, (name, prob) in enumerate(top3, 1):
        marker = " <-- 实际故障" if actual_fault and name == actual_fault else ""
        print(f"  {i}. {name}: {prob:.1%}{marker}")

    if not args.no_llm:
        print("\n" + "=" * 60)
        print("LLM 解释")
        print("=" * 60)
        explanation = predictor.get_llm_explanation(top3, fault_name)
        print(explanation if explanation else "[警告] 未获得LLM解释", flush=True)

    if actual_fault:
        print(f"\n实际故障: {actual_fault}")
        print(f"预测结果: {'正确' if fault_name == actual_fault else '错误'}")


if __name__ == '__main__':
    main()
