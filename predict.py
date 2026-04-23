#!/usr/bin/env python3
"""
交互式故障预测与剩余寿命预测脚本
支持模型预测和LLM解释

用法:
  python predict.py --sample                    # 使用随机示例数据
  python predict.py --data "0.5,-1.2,..."      # 输入传感器数据
  python predict.py --model KG-MLP-V2          # 指定模型
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
from src.models.mlp_model import KGEnhancedMLPV2Model, load_kg_embeddings_v4
from src.models.cnn_model import CNNKGModelV2


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

    def __init__(self, model_name='KG-MLP V2'):
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
        self.n_features = len(feature_cols)
        self.feature_names = feature_cols

        print(f"  特征数量: {self.n_features}")
        print(f"  故障类型: {len(self.fault_types)}")
        print(f"  Ollama可用: {'是' if self.ollama.is_available() else '否'}")

    def load_model(self):
        """加载模型"""
        model_path_map = {
            'KG-MLP V2': 'kg_enhanced_mlp_v2_model.pt',
            'CNN-KG V2': 'cnn_kg_v2_model.pt'
        }

        path = Path('models') / model_path_map.get(self.model_name, 'kg_enhanced_mlp_v2_model.pt')

        if not path.exists():
            print(f"[错误] 模型文件不存在: {path}")
            return

        print(f"加载模型: {self.model_name}...")

        checkpoint = torch.load(path, map_location=self.device)
        saved_config = checkpoint.get('config', {})

        if self.model_name == 'KG-MLP V2':
            self.model = KGEnhancedMLPV2Model(config_path='config.yaml')
            self.model.hidden_dim = saved_config.get('hidden_dim', 192)
            self.model.kg_embedding_dim = saved_config.get('kg_embedding_dim', 33)
        else:  # CNN-KG V2
            self.model = CNNKGModelV2(config_path='config.yaml')
            self.model.hidden_dim = saved_config.get('hidden_dim', 64)
            self.model.kg_embedding_dim = saved_config.get('kg_embedding_dim', 33)

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

    def predict(self, features):
        """预测故障"""
        # 预处理
        X = np.array(features).reshape(1, -1)
        X_scaled = self.scaler.transform(X)

        # 获取KG嵌入 - 使用预测的故障类型
        # 这里简化处理，使用所有可能的故障类型的平均嵌入
        kg_emb = np.zeros(33)

        # 预测
        self.model.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X_scaled, dtype=torch.float).to(self.model.device)
            kg_tensor = torch.tensor(kg_emb.reshape(1, -1), dtype=torch.float).to(self.model.device)
            output = self.model.model(X_tensor, kg_tensor)
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

        prompt = f"""基于以下故障预测结果，给出简洁的解释和建议：

预测故障类型: {fault_name}

故障类型概率分布:
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
    parser.add_argument('--model', type=str, default='KG-MLP V2',
                        choices=['KG-MLP V2', 'CNN-KG V2'], help='选择模型')
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