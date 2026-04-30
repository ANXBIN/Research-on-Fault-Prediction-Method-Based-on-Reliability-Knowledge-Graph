#!/usr/bin/env python3
"""
基于规则的故障推理模块

利用知识图谱中的故障-部件-特征关系，结合特征统计阈值进行规则推理
"""

import numpy as np
import json
from pathlib import Path


class RuleBasedReasoner:
    """基于规则的故障推理器"""

    FAULT_TYPES = [
        'Ball_Fault', 'Broken_Tooth', 'Inner_Race_Fault', 'Missing_Tooth',
        'Mixed_Fault', 'Normal', 'Outer_Race_Fault', 'Root_Crack', 'Tooth_Wear'
    ]

    COMPONENT_MAP = {
        'Ball_Fault': ['Rolling_Element'],
        'Broken_Tooth': ['Gear_Tooth'],
        'Inner_Race_Fault': ['Inner_Race'],
        'Missing_Tooth': ['Gear_Tooth'],
        'Mixed_Fault': ['Rolling_Element', 'Inner_Race', 'Outer_Race', 'Gear_Tooth'],
        'Normal': [],
        'Outer_Race_Fault': ['Outer_Race'],
        'Root_Crack': ['Gear_Tooth'],
        'Tooth_Wear': ['Gear_Tooth'],
    }

    FAULT_FEATURE_RULES = {
        'Normal': {
            'mean': (0, 500000),
            'std': (0, 200000),
            'rms': (0, 300000),
            'kurtosis': (2.5, 4.5),
            'skewness': (-1.0, 1.0),
            'crest_factor': (2.0, 4.0),
        },
        'Ball_Fault': {
            'kurtosis': (4.0, 100),
            'skewness': (-100, -1.5),
            'peak': (1000000, 1e12),
            'crest_factor': (4.0, 100),
        },
        'Inner_Race_Fault': {
            'kurtosis': (3.5, 100),
            'impulse_factor': (5.0, 100),
            'peak': (800000, 1e12),
        },
        'Outer_Race_Fault': {
            'kurtosis': (3.5, 100),
            'shape_factor': (1.2, 100),
            'peak': (600000, 1e12),
        },
        'Broken_Tooth': {
            'peak': (1200000, 1e12),
            'crest_factor': (5.0, 100),
            'impulse_factor': (6.0, 100),
            'max': (1000000, 1e12),
        },
        'Missing_Tooth': {
            'spectral_entropy': (0, 2.0),
            'dominant_frequency': (0, 1000),
            'peak': (500000, 1e12),
        },
        'Tooth_Wear': {
            'spectral_energy': (0, 1e10),
            'band_energy_1': (0, 1e10),
            'mean': (0, 400000),
        },
        'Root_Crack': {
            'kurtosis': (3.0, 100),
            'skewness': (-100, -1.0),
            'band_energy_3': (0, 1e10),
        },
        'Mixed_Fault': {
            'kurtosis': (3.0, 100),
            'spectral_entropy': (2.5, 100),
        },
    }

    def __init__(self, feature_names=None):
        self.feature_names = feature_names or [
            'mean', 'std', 'rms', 'max', 'min', 'peak', 'skewness', 'kurtosis',
            'crest_factor', 'shape_factor', 'impulse_factor',
            'spectral_energy', 'spectral_centroid', 'spectral_entropy', 'dominant_frequency',
            'band_energy_1', 'band_energy_2', 'band_energy_3', 'band_energy_4', 'band_energy_5'
        ]
        self.feature_idx = {name: i for i, name in enumerate(self.feature_names)}
        self.kg_graph = self._load_kg_graph()

    def _load_kg_graph(self):
        kg_path = Path('data/processed/knowledge_graph.json')
        if not kg_path.exists():
            return None
        try:
            with open(kg_path, 'r') as f:
                data = json.load(f)
            return data
        except (json.JSONDecodeError, MemoryError):
            return None

    def _score_fault(self, features, fault_type):
        rules = self.FAULT_FEATURE_RULES.get(fault_type, {})
        if not rules:
            return 0.0

        score = 0.0
        matched = 0
        for feat_name, (low, high) in rules.items():
            if feat_name not in self.feature_idx:
                continue
            val = features[self.feature_idx[feat_name]]
            if low <= val <= high:
                score += 1.0
            elif val > high:
                score += 0.5
            matched += 1

        return score / matched if matched > 0 else 0.0

    def _get_kg_component_score(self, fault_type):
        components = self.COMPONENT_MAP.get(fault_type, [])
        if not components:
            return 0.8
        return 1.0

    def _get_kg_similarity_boost(self, fault_type, features):
        try:
            fault_emb_path = Path('data/processed/fault_embeddings.json')
            if not fault_emb_path.exists():
                return 0.0

            with open(fault_emb_path, 'r') as f:
                fault_embs = json.load(f)

            fault_sim = fault_embs.get('fault_similarity', {})
            fault_types = fault_embs.get('fault_types', [])

            # 处理两种格式：字典或列表矩阵
            if isinstance(fault_sim, dict):
                if fault_type in fault_sim:
                    sims = fault_sim[fault_type]
                    max_sim = max(sims.values()) if isinstance(sims, dict) else max(sims) if sims else 0
                    return max_sim * 0.1
            elif isinstance(fault_sim, list) and fault_type in fault_types:
                idx = fault_types.index(fault_type)
                if idx < len(fault_sim):
                    sims = fault_sim[idx]
                    max_sim = max(sims) if sims else 0
                    return max_sim * 0.1
        except (json.JSONDecodeError, KeyError, ValueError):
            pass
        return 0.0

    def diagnose(self, features, top_k=5):
        scores = {}
        details = {}

        for fault_type in self.FAULT_TYPES:
            feature_score = self._score_fault(features, fault_type)
            kg_score = self._get_kg_component_score(fault_type)
            kg_boost = self._get_kg_similarity_boost(fault_type, features)

            total_score = feature_score * 0.7 + kg_score * 0.2 + kg_boost * 0.1
            scores[fault_type] = total_score
            details[fault_type] = {
                'feature_score': round(feature_score, 3),
                'kg_component_score': round(kg_score, 3),
                'kg_similarity_boost': round(kg_boost, 3),
                'total_score': round(total_score, 3),
                'components': self.COMPONENT_MAP.get(fault_type, []),
            }

        sorted_faults = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        total = sum(s[1] for s in sorted_faults) or 1.0
        probabilities = {k: v / total for k, v in sorted_faults}

        return sorted_faults[0][0], probabilities, details

    def explain(self, fault_type, features):
        components = self.COMPONENT_MAP.get(fault_type, [])
        rules = self.FAULT_FEATURE_RULES.get(fault_type, {})

        explanation = {
            'fault_type': fault_type,
            'related_components': components,
            'rule_matches': [],
            'knowledge_graph_info': self._get_kg_info(fault_type),
        }

        for feat_name, (low, high) in rules.items():
            if feat_name not in self.feature_idx:
                continue
            val = features[self.feature_idx[feat_name]]
            matched = low <= val <= high
            explanation['rule_matches'].append({
                'feature': feat_name,
                'value': round(float(val), 4),
                'expected_range': f'[{low}, {high}]',
                'matched': matched,
            })

        return explanation

    def _get_kg_info(self, fault_type):
        info = {
            'components': self.COMPONENT_MAP.get(fault_type, []),
            'similar_faults': [],
        }

        if self.kg_graph:
            try:
                fault_embs_path = Path('data/processed/fault_embeddings.json')
                if fault_embs_path.exists():
                    with open(fault_embs_path, 'r') as f:
                        fault_embs = json.load(f)
                    fault_sim = fault_embs.get('fault_similarity', {})
                    fault_types = fault_embs.get('fault_types', [])

                    # 处理两种格式
                    if isinstance(fault_sim, dict):
                        if fault_type in fault_sim:
                            sims = fault_sim[fault_type]
                            if isinstance(sims, dict):
                                sorted_sims = sorted(sims.items(), key=lambda x: x[1], reverse=True)
                                info['similar_faults'] = [
                                    {'fault': k, 'similarity': round(v, 3)}
                                    for k, v in sorted_sims[1:4]
                                ]
                    elif isinstance(fault_sim, list) and fault_type in fault_types:
                        idx = fault_types.index(fault_type)
                        if idx < len(fault_sim):
                            sims = fault_sim[idx]
                            # 创建(fault_name, similarity)对
                            sim_pairs = [(fault_types[j], sims[j]) for j in range(len(fault_types)) if j != idx and j < len(sims)]
                            sorted_sims = sorted(sim_pairs, key=lambda x: x[1], reverse=True)
                            info['similar_faults'] = [
                                {'fault': k, 'similarity': round(v, 3)}
                                for k, v in sorted_sims[:3]
                            ]
            except (json.JSONDecodeError, KeyError, ValueError):
                pass

        return info


if __name__ == '__main__':
    reasoner = RuleBasedReasoner()
    test_features = np.random.randn(20) * 100000
    fault, probs, details = reasoner.diagnose(test_features)
    print(f"规则推理结果: {fault}")
    print(f"概率分布: {probs}")
