#!/usr/bin/env python3
"""
数据预处理脚本：加载XJTU变速箱数据集，提取特征，构建样本
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats, signal
from scipy.fftpack import fft
from tqdm import tqdm
import yaml
import json
import warnings
warnings.filterwarnings('ignore')


class GearboxDataLoader:
    """变速箱数据加载器"""

    FAULT_LABELS = {
        '1ndBearing_ball': 'Ball_Fault',
        '1ndBearing_inner': 'Inner_Race_Fault',
        '1ndBearing_outer': 'Outer_Race_Fault',
        '1ndBearing_mix(inner+outer+ball)': 'Mixed_Fault',
        '2ndPlanetary_brokentooth': 'Broken_Tooth',
        '2ndPlanetary_missingtooth': 'Missing_Tooth',
        '2ndPlanetary_normalstate': 'Normal',
        '2ndPlanetary_rootcracks': 'Root_Crack',
        '2ndPlanetary_toothwear': 'Tooth_Wear'
    }

    def __init__(self, data_path):
        self.data_path = Path(data_path)
        self.fault_types = list(self.FAULT_LABELS.keys())

    def load_all_data(self):
        """加载所有故障类型的数据"""
        all_data = []
        for fault_type in self.fault_types:
            fault_path = self.data_path / fault_type
            if not fault_path.exists():
                continue
            data = self.load_fault_type(fault_type)
            all_data.extend(data)
        return all_data

    def load_fault_type(self, fault_type):
        """加载特定故障类型的数据"""
        fault_path = self.data_path / fault_type
        samples = []

        for txt_file in sorted(fault_path.glob('Data_Chan*.txt')):
            # 尝试加载数据，跳过可能的文本头部
            try:
                # 尝试直接加载
                channel_data = np.loadtxt(txt_file)
            except ValueError:
                # 如果失败，手动读取并过滤
                with open(txt_file, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()

                # 找到第一行纯数字的行
                data_lines = []
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 1:
                        try:
                            float(parts[0])
                            data_lines.append([float(x) for x in parts[:1]])  # 只取第一列
                        except ValueError:
                            continue

                if data_lines:
                    channel_data = np.array(data_lines).flatten()
                else:
                    continue

            # 确保数据是一维的
            channel_data = channel_data.flatten()

            # 使用滑动窗口提取样本
            for i in range(0, len(channel_data) - 1024, 512):
                window = channel_data[i:i+1024]
                samples.append({
                    'fault_type': self.FAULT_LABELS[fault_type],
                    'channel': txt_file.stem,
                    'data': window
                })
        return samples


class FeatureExtractor:
    """特征提取器"""

    def __init__(self, sampling_rate=25600):
        self.sampling_rate = sampling_rate

    def extract_time_domain(self, signal_data):
        """提取时域特征"""
        signal_data = np.array(signal_data)

        mean_val = np.mean(signal_data)
        std_val = np.std(signal_data)
        rms_val = np.sqrt(np.mean(signal_data**2))
        max_val = np.max(signal_data)
        min_val = np.min(signal_data)
        peak_val = max(abs(max_val), abs(min_val))

        # 偏度和峰度
        skewness = stats.skew(signal_data)
        kurtosis = stats.kurtosis(signal_data)

        # 峰值因子
        crest_factor = peak_val / rms_val if rms_val > 0 else 0

        # 裕度因子
        sqrt_abs = np.sqrt(np.abs(signal_data))
        shape_factor = rms_val / np.mean(sqrt_abs) if np.mean(sqrt_abs) > 0 else 0

        # 脉冲因子
        impulse_factor = peak_val / np.mean(sqrt_abs) if np.mean(sqrt_abs) > 0 else 0

        return {
            'mean': mean_val,
            'std': std_val,
            'rms': rms_val,
            'max': max_val,
            'min': min_val,
            'peak': peak_val,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'crest_factor': crest_factor,
            'shape_factor': shape_factor,
            'impulse_factor': impulse_factor
        }

    def extract_frequency_domain(self, signal_data):
        """提取频域特征"""
        signal_data = np.array(signal_data)
        n = len(signal_data)

        # FFT变换
        fft_vals = fft(signal_data)
        fft_magnitude = np.abs(fft_vals[:n//2])
        freqs = np.fft.fftfreq(n, 1/self.sampling_rate)[:n//2]

        # 频谱能量
        spectral_energy = np.sum(fft_magnitude**2)

        # 频谱质心
        spectral_centroid = np.sum(freqs * fft_magnitude) / np.sum(fft_magnitude) if np.sum(fft_magnitude) > 0 else 0

        # 频谱熵
        psd = fft_magnitude**2
        psd_norm = psd / np.sum(psd) if np.sum(psd) > 0 else psd
        spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))

        # 主频率
        dominant_freq = freqs[np.argmax(fft_magnitude)] if len(fft_magnitude) > 0 else 0

        # 频带能量
        n_bands = 5
        band_size = len(fft_magnitude) // n_bands
        band_energies = []
        for i in range(n_bands):
            start = i * band_size
            end = start + band_size
            band_energies.append(np.sum(fft_magnitude[start:end]**2))

        return {
            'spectral_energy': spectral_energy,
            'spectral_centroid': spectral_centroid,
            'spectral_entropy': spectral_entropy,
            'dominant_frequency': dominant_freq,
            'band_energy_1': band_energies[0] if len(band_energies) > 0 else 0,
            'band_energy_2': band_energies[1] if len(band_energies) > 1 else 0,
            'band_energy_3': band_energies[2] if len(band_energies) > 2 else 0,
            'band_energy_4': band_energies[3] if len(band_energies) > 3 else 0,
            'band_energy_5': band_energies[4] if len(band_energies) > 4 else 0
        }

    def extract_all_features(self, signal_data):
        """提取所有特征"""
        time_features = self.extract_time_domain(signal_data)
        freq_features = self.extract_frequency_domain(signal_data)
        return {**time_features, **freq_features}


class KnowledgeGraphBuilder:
    """知识图谱构建器"""

    def __init__(self):
        self.nodes = []
        self.edges = []
        self.node_id_counter = 0

    def create_fault_node(self, fault_type, features):
        """创建故障节点"""
        node_id = f"Fault_{fault_type}_{self.node_id_counter}"
        self.node_id_counter += 1

        node = {
            'id': node_id,
            'type': 'Fault',
            'label': fault_type,
            'properties': features
        }
        self.nodes.append(node)
        return node_id

    def create_component_node(self, component_name):
        """创建部件节点"""
        node_id = f"Component_{component_name}"
        existing = [n for n in self.nodes if n['id'] == node_id]
        if existing:
            return node_id

        node = {
            'id': node_id,
            'type': 'Component',
            'label': component_name,
            'properties': {}
        }
        self.nodes.append(node)
        return node_id

    def create_feature_node(self, feature_name, value):
        """创建特征节点"""
        node_id = f"Feature_{feature_name}_{self.node_id_counter}"
        self.node_id_counter += 1

        node = {
            'id': node_id,
            'type': 'Feature',
            'label': feature_name,
            'properties': {'value': float(value)}
        }
        self.nodes.append(node)
        return node_id

    def create_edge(self, source_id, target_id, relation_type, properties=None):
        """创建关系边"""
        edge = {
            'source': source_id,
            'target': target_id,
            'type': relation_type,
            'properties': properties or {}
        }
        self.edges.append(edge)

    def build_graph_for_sample(self, fault_type, features):
        """为单个样本构建知识图谱子图"""
        # 创建故障节点
        fault_node_id = self.create_fault_node(fault_type, features)

        # 创建特征节点并连接
        for feature_name, value in features.items():
            feature_node_id = self.create_feature_node(feature_name, value)
            self.create_edge(fault_node_id, feature_node_id, 'HAS_FEATURE')

        # 创建部件关系（基于故障类型）
        component = self._get_component_from_fault(fault_type)
        if component:
            component_node_id = self.create_component_node(component)
            self.create_edge(fault_node_id, component_node_id, 'LOCATED_AT')

        # 创建故障关系
        self._create_fault_relations(fault_node_id, fault_type)

        return fault_node_id

    def _get_component_from_fault(self, fault_type):
        """从故障类型获取部件"""
        if 'Bearing' in fault_type:
            return 'Bearing'
        elif 'Planetary' in fault_type:
            return 'Planetary_Gear'
        return 'Unknown'

    def _create_fault_relations(self, fault_node_id, fault_type):
        """创建故障关联关系"""
        # 基于故障类型创建因果关系
        if 'Ball' in fault_type:
            cause_component = self.create_component_node('Rolling_Element')
            self.create_edge(fault_node_id, cause_component, 'CAUSED_BY')
        elif 'Inner' in fault_type:
            cause_component = self.create_component_node('Inner_Race')
            self.create_edge(fault_node_id, cause_component, 'CAUSED_BY')
        elif 'Outer' in fault_type:
            cause_component = self.create_component_node('Outer_Race')
            self.create_edge(fault_node_id, cause_component, 'CAUSED_BY')
        elif 'Tooth' in fault_type or 'Crack' in fault_type or 'Wear' in fault_type:
            cause_component = self.create_component_node('Gear_Tooth')
            self.create_edge(fault_node_id, cause_component, 'CAUSED_BY')

    def export_to_neo4j(self, output_path):
        """导出为Neo4j格式"""
        output = {
            'nodes': self.nodes,
            'edges': self.edges
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        return output_path


def main():
    """主函数"""
    import argparse
    parser = argparse.ArgumentParser(description='预处理XJTU变速箱数据')
    parser.add_argument('--data_path', type=str,
                        default='data/raw/XJTU_Gearbox',
                        help='数据路径')
    parser.add_argument('--output_path', type=str,
                        default='data/processed',
                        help='输出路径')
    parser.add_argument('--sampling_rate', type=int, default=25600,
                        help='采样率')
    args = parser.parse_args()

    # 创建输出目录
    output_dir = Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("XJTU变速箱数据预处理")
    print("=" * 60)

    # 加载数据
    print("\n[1/4] 加载数据...")
    loader = GearboxDataLoader(args.data_path)
    all_samples = loader.load_all_data()
    print(f"  总样本数: {len(all_samples)}")

    # 提取特征
    print("\n[2/4] 提取特征...")
    extractor = FeatureExtractor(sampling_rate=args.sampling_rate)
    processed_data = []

    for sample in tqdm(all_samples, desc="特征提取"):
        features = extractor.extract_all_features(sample['data'])
        features['fault_type'] = sample['fault_type']
        features['channel'] = sample['channel']
        processed_data.append(features)

    # 保存处理后的数据
    print("\n[3/4] 保存处理后的数据...")
    df = pd.DataFrame(processed_data)
    df.to_csv(output_dir / 'processed_features.csv', index=False)
    print(f"  保存至: {output_dir / 'processed_features.csv'}")

    # 构建知识图谱
    print("\n[4/4] 构建知识图谱...")
    kg_builder = KnowledgeGraphBuilder()

    for sample in tqdm(processed_data, desc="构建知识图谱"):
        kg_builder.build_graph_for_sample(
            sample['fault_type'],
            {k: v for k, v in sample.items()
             if k not in ['fault_type', 'channel']}
        )

    # 导出知识图谱
    kg_path = output_dir / 'knowledge_graph.json'
    kg_builder.export_to_neo4j(kg_path)
    print(f"  知识图谱保存至: {kg_path}")
    print(f"  节点数: {len(kg_builder.nodes)}")
    print(f"  边数: {len(kg_builder.edges)}")

    # 保存故障类型映射
    fault_mapping = {v: k for k, v in GearboxDataLoader.FAULT_LABELS.items()}
    with open(output_dir / 'fault_mapping.json', 'w') as f:
        json.dump(fault_mapping, f, indent=2)

    print("\n" + "=" * 60)
    print("预处理完成!")
    print("=" * 60)


if __name__ == '__main__':
    main()
