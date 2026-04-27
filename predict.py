#!/usr/bin/env python3
"""
交互式故障预测与剩余寿命预测脚本
支持模型预测和LLM解释

用法:
  python predict.py --sample                    # 使用随机示例数据
  python predict.py --data "0.5,-1.2,..."      # 输入传感器数据
  python predict.py --model MLP-KG-V2          # 指定模型
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import yaml
import numpy as np
import pandas as pd
import json
import requests
import argparse
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 导入模型
from src.models.mlp_model import MLPModel, KGEnhancedMLPV2Model, load_kg_embeddings_v4
from src.models.cnn_model import CNNModel, CNNKGModelV3
from src.models.gnn_model import GNNModel, GNNKGModel


class OllamaClient:
    """Ollama LLM 客户端"""

    def __init__(self, base_url="http://localhost:11434"):
        self.base_url = base_url
        self.model = "qwen3:4b"

    def is_available(self):
        """检查Ollama是否可用"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False

    def generate(self, prompt):
        """生成LLM响应"""
        payload = {
            "model": self.model,
            "prompt": "/no_think\n" + prompt,
            "stream": False
        }

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload
            )
            if response.status_code == 200:
                return response.json().get("response", "")
            else:
                return f"Error: {response.status_code}"
        except Exception as e:
            return f"Connection error: {str(e)}"


class Predictor:
    """故障预测器"""

    def __init__(self, model_name='MLP-KG-V2'):
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.ollama = OllamaClient()
        self.model_name = model_name
        self.model = None
        self.scaler = None
        self.label_encoder = LabelEncoder()
        self.fault_to_idx = {}
        self.n_features = 0
        self.feature_names = []
        self.fault_types = []

        self.load_data()
        self.load_model()

    def load_data(self):
        """加载数据处理器"""
        print("加载数据...")

        # 加载处理后的数据获取标签信息
        df = pd.read_csv('data/processed/processed_features.csv')
        self.fault_types = df['fault_type'].unique()
        self.label_encoder.fit(self.fault_types)
        self.fault_to_idx = dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))

        # 特征列
        feature_cols = [col for col in df.columns if col not in ['fault_type', 'channel']]

        # 标准化器
        self.scaler = StandardScaler()
        X = df[feature_cols].values
        self.scaler.fit(X)
        self.X_train_raw = X  # 保存原始训练数据用于KNN
        self.n_features = len(feature_cols)
        self.feature_names = feature_cols

        print(f"  特征数量: {self.n_features}")
        print(f"  故障类型: {len(self.fault_types)}")
        print(f"  Ollama可用: {'是' if self.ollama.is_available() else '否'}")

    def load_model(self):
        """加载模型"""
        model_path_map = {
            'MLP': 'mlp_model.pt',
            'MLP-KG': 'mlp_kg_model.pt',
            'CNN': 'cnn_model.pt',
            'CNN-KG': 'cnn_kg_model.pt',
            'GNN': 'gnn_model.pt',
            'GNN-KG': 'gnn_kg_model.pt'
        }

        path = Path('models') / model_path_map.get(self.model_name, 'mlp_kg_model.pt')

        if not path.exists():
            print(f"[错误] 模型文件不存在: {path}")
            return

        print(f"加载模型: {self.model_name}...")

        checkpoint = torch.load(path, map_location=self.device)
        saved_config = checkpoint.get('config', {})

        if self.model_name == 'MLP':
            self.model = MLPModel(config_path='config.yaml')
            self.model.hidden_dim = saved_config.get('hidden_dim', 128)
            self.model.dropout = saved_config.get('dropout', 0.3)
        elif self.model_name == 'MLP-KG':
            self.model = KGEnhancedMLPV2Model(config_path='config.yaml')
            self.model.hidden_dim = saved_config.get('hidden_dim', 192)
            self.model.kg_embedding_dim = saved_config.get('kg_embedding_dim', 64)
        elif self.model_name == 'CNN':
            self.model = CNNModel(config_path='config.yaml')
            self.model.hidden_dim = saved_config.get('hidden_dim', 64)
            self.model.dropout = saved_config.get('dropout', 0.3)
        elif self.model_name == 'CNN-KG':
            self.model = CNNKGModelV3(config_path='config.yaml')
            self.model.hidden_dim = saved_config.get('hidden_dim', 64)
            self.model.kg_embedding_dim = saved_config.get('kg_embedding_dim', 33)
        elif self.model_name == 'GNN':
            self.model = GNNModel(config_path='config.yaml')
            self.model.hidden_dim = saved_config.get('hidden_dim', 256)
            self.model.num_layers = saved_config.get('num_layers', 4)
            self.model.dropout = saved_config.get('dropout', 0.3)
            self.model.batch_size = saved_config.get('batch_size', 256)
        elif self.model_name == 'GNN-KG':
            self.model = GNNKGModel(config_path='config.yaml')
            self.model.hidden_dim = saved_config.get('hidden_dim', 256)
            self.model.num_layers = saved_config.get('num_layers', 4)
            self.model.dropout = saved_config.get('dropout', 0.3)
            self.model.kg_embedding_dim = saved_config.get('kg_embedding_dim', 33)
            self.model.batch_size = saved_config.get('batch_size', 256)

        self.model.dropout = saved_config.get('dropout', 0.2)
        self.model.fault_to_idx = self.fault_to_idx
        self.model.build_model(self.n_features, len(self.fault_types))
        self.model.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  模型已加载")

    def load_kg_embedding(self, fault_type):
        """加载故障类型的KG嵌入"""
        try:
            kg_emb = load_kg_embeddings_v4(
                'data/processed/fault_embeddings.json',
                [fault_type],
                'data/processed/kg_embeddings.json'
            )
            return kg_emb[0]
        except Exception as e:
            print(f"  [警告] KG嵌入加载失败: {e}")
            return np.zeros(33)

    def compute_knn_embedding(self, X_sample, X_train, k=20):
        """为单个样本计算KNN嵌入"""
        from sklearn.neighbors import NearestNeighbors

        n_features = X_train.shape[1]
        X_sample = X_sample.reshape(1, -1)

        # 归一化
        X_train_norm = X_train / (np.linalg.norm(X_train, axis=1, keepdims=True) + 1e-8)
        X_sample_norm = X_sample / (np.linalg.norm(X_sample, axis=1, keepdims=True) + 1e-8)

        # KNN
        nn_model = NearestNeighbors(n_neighbors=min(k+1, len(X_train)), metric='euclidean')
        nn_model.fit(X_train_norm)
        distances, indices = nn_model.kneighbors(X_sample_norm)

        # 邻居特征
        neighbor_indices = indices[0, 1:]  # 排除自己
        neighbor_features = X_train_norm[neighbor_indices]
        neighbor_distances = distances[0, 1:]

        # 距离倒数加权
        weights = 1.0 / np.maximum(neighbor_distances, 1e-8)
        weights = weights / weights.sum()

        # 邻居特征加权和
        neighbor_weighted = np.dot(weights, neighbor_features)

        # 差异
        diff = X_sample_norm[0] - neighbor_weighted

        # 构建64维嵌入
        emb = np.zeros(64, dtype=np.float32)
        emb[:n_features] = X_sample_norm[0]
        emb[n_features:2*n_features] = neighbor_weighted
        emb[2*n_features:3*n_features] = diff

        return emb

    def predict(self, features):
        """预测故障"""
        # 预处理
        X = np.array(features).reshape(1, -1)
        X_scaled = self.scaler.transform(X)

        # 获取KG嵌入（根据故障类型）
        def get_kg_embedding(fault_type):
            try:
                from src.models.mlp_model import load_kg_embeddings_v4
                kg_emb = load_kg_embeddings_v4(
                    'data/processed/fault_embeddings.json',
                    [fault_type],
                    'data/processed/kg_embeddings.json'
                )
                return kg_emb[0]
            except:
                return np.zeros(33)

        # 获取MLP用的KNN嵌入
        def get_knn_embedding(X_sample, X_train, k=20):
            from sklearn.neighbors import NearestNeighbors
            n_features = X_train.shape[1]
            X_sample = X_sample.reshape(1, -1)
            X_train_norm = X_train / (np.linalg.norm(X_train, axis=1, keepdims=True) + 1e-8)
            X_sample_norm = X_sample / (np.linalg.norm(X_sample, axis=1, keepdims=True) + 1e-8)
            nn_model = NearestNeighbors(n_neighbors=min(k+1, len(X_train)), metric='euclidean')
            nn_model.fit(X_train_norm)
            distances, indices = nn_model.kneighbors(X_sample_norm)
            neighbor_indices = indices[0, 1:]
            neighbor_features = X_train_norm[neighbor_indices]
            neighbor_distances = distances[0, 1:]
            weights = 1.0 / np.maximum(neighbor_distances, 1e-8)
            weights = weights / weights.sum()
            neighbor_weighted = np.dot(weights, neighbor_features)
            diff = X_sample_norm[0] - neighbor_weighted
            emb = np.zeros(64, dtype=np.float32)
            emb[:n_features] = X_sample_norm[0]
            emb[n_features:2*n_features] = neighbor_weighted
            emb[2*n_features:3*n_features] = diff
            return emb

        # 两阶段预测：第一阶段用平均嵌入得到初步预测
        avg_kg_emb = np.zeros(33)
        try:
            from src.models.mlp_model import load_kg_embeddings_v4
            all_kg_embs = load_kg_embeddings_v4(
                'data/processed/fault_embeddings.json',
                list(self.label_encoder.classes_),
                'data/processed/kg_embeddings.json'
            )
            avg_kg_emb = np.mean(all_kg_embs, axis=0)
        except:
            pass

        self.model.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X_scaled, dtype=torch.float).to(self.model.device)

            if isinstance(self.model, (GNNModel, GNNKGModel)):
                from src.models.gnn_model import build_batch_adjacency
                adj = build_batch_adjacency(X_scaled, k=30).to(self.model.device)

                if isinstance(self.model, GNNKGModel):
                    kg_tensor = torch.tensor(avg_kg_emb.reshape(1, -1), dtype=torch.float).to(self.model.device)
                    output = self.model.model(X_tensor, kg_tensor, adj)
                else:
                    output = self.model.model(X_tensor, adj)
            elif isinstance(self.model, KGEnhancedMLPV2Model):
                # MLP模型：使用KNN嵌入
                knn_emb = get_knn_embedding(X_scaled, self.X_train_raw, k=20)
                kg_tensor = torch.tensor(knn_emb.reshape(1, -1), dtype=torch.float).to(self.model.device)
                output = self.model.model(X_tensor, kg_tensor)
            elif isinstance(self.model, CNNKGModelV3):
                kg_tensor = torch.tensor(avg_kg_emb.reshape(1, -1), dtype=torch.float).to(self.model.device)
                output = self.model.model(X_tensor, kg_tensor)
            else:
                # MLP和CNN基准模型
                output = self.model.model(X_tensor)

            pred_idx = output.argmax(dim=1).item()
            fault_name = self.label_encoder.inverse_transform([pred_idx])[0]

            # 第二阶段：用预测结果的KG嵌入重新预测（仅KG模型）
            if isinstance(self.model, GNNKGModel):
                specific_kg_emb = get_kg_embedding(fault_name)
                kg_tensor = torch.tensor(specific_kg_emb.reshape(1, -1), dtype=torch.float).to(self.model.device)
                output = self.model.model(X_tensor, kg_tensor, adj)
                pred_idx = output.argmax(dim=1).item()
                fault_name = self.label_encoder.inverse_transform([pred_idx])[0]

            probabilities = torch.softmax(output, dim=1).cpu().numpy()[0]

        return fault_name, probabilities

    def get_top_k_predictions(self, features, k=3):
        """获取Top-K预测"""
        fault_name, probabilities = self.predict(features)
        top_k_idx = np.argsort(probabilities)[::-1][:k]
        top_k_probs = probabilities[top_k_idx]
        top_k_names = self.label_encoder.inverse_transform(top_k_idx)

        return [(n, float(p)) for n, p in zip(top_k_names, top_k_probs)]

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
    parser.add_argument('--model', type=str, default='MLP-KG-V2',
                        choices=['MLP', 'MLP-KG', 'CNN', 'CNN-KG', 'GNN', 'GNN-KG'], help='选择模型')
    parser.add_argument('--no-llm', action='store_true', help='禁用LLM解释')
    parser.add_argument('--idx', type=int, help='指定样本索引')

    args = parser.parse_args()

    print("=" * 60)
    print("变速箱故障预测系统")
    print("=" * 60)

    # 初始化预测器
    predictor = Predictor(model_name=args.model)

    # 获取数据
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
            sample_idx = None
        except ValueError:
            print("[错误] 请输入有效的数值")
            return
    else:
        print("请使用 --sample 或 --data 参数")
        print("使用 --help 查看更多信息")
        return

    # 预测
    print("\n" + "=" * 60)
    print("预测结果")
    print("=" * 60)

    top3 = predictor.get_top_k_predictions(features, k=3)
    fault_name, _ = predictor.predict(features)

    print(f"预测故障: {fault_name}")
    print(f"Top-3预测:")
    for i, (name, prob) in enumerate(top3, 1):
        marker = " <-- 实际故障" if actual_fault and name == actual_fault else ""
        print(f"  {i}. {name}: {prob:.1%}{marker}")

    # LLM解释
    if not args.no_llm:
        print("\n" + "=" * 60)
        print("LLM 解释")
        print("=" * 60)
        explanation = predictor.get_llm_explanation(top3, fault_name)
        if explanation:
            print(explanation, flush=True)
        else:
            print("[警告] 未获得LLM解释", flush=True)

    # 显示实际故障（如果有）
    if actual_fault:
        is_correct = "正确" if fault_name == actual_fault else "错误"
        print(f"\n实际故障: {actual_fault}")
        print(f"预测结果: {is_correct}")


if __name__ == '__main__':
    main()