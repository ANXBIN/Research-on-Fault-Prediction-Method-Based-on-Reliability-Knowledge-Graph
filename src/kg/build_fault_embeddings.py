#!/usr/bin/env python3
"""
构建故障类型级别的KG嵌入矩阵
从knowledge_graph.json提取:
1. 故障相似度矩阵 (基于共享部件/特征)
2. 故障-部件关联矩阵
3. 故障-特征权重矩阵
"""

import json
import re
import numpy as np
from collections import defaultdict

FAULT_TYPES = [
    'Ball_Fault', 'Broken_Tooth', 'Inner_Race_Fault', 'Missing_Tooth',
    'Mixed_Fault', 'Normal', 'Outer_Race_Fault', 'Root_Crack', 'Tooth_Wear'
]

def build_fault_embeddings_from_kg(kg_path):
    """从KG构建故障类型级别的嵌入"""
    fault_to_idx = {f: i for i, f in enumerate(FAULT_TYPES)}
    n_faults = len(FAULT_TYPES)

    fault_nodes = defaultdict(int)  # fault_label -> count
    fault_components = defaultdict(set)  # fault_label -> set of components
    fault_features = defaultdict(set)  # fault_label -> set of feature types
    all_components = set()
    all_feature_labels = set()

    print("加载知识图谱文件...")
    with open(kg_path, 'r', encoding='utf-8') as f:
        content = f.read()

    print("Step 1: 解析故障节点...")
    # 构建Fault ID到标签的映射: Fault_Ball_Fault_0 -> Ball_Fault
    fault_id_to_label = {}
    fault_node_pattern = re.compile(
        r'"id":\s*"Fault_([^"]+)"[^}]*"label":\s*"([^"]+)"',
        re.DOTALL
    )
    for match in fault_node_pattern.finditer(content):
        fault_id = match.group(1)  # e.g., "Ball_Fault_0"
        fault_label = match.group(2)  # e.g., "Ball_Fault"
        fault_id_to_label[fault_id] = fault_label
        fault_nodes[fault_label] += 1

    print(f"故障类型数: {len(fault_id_to_label)}, 节点统计: {dict(fault_nodes)}")

    print("Step 2: 解析故障-部件关联 (CAUSED_BY)...")
    caused_by_positions = [m.start() for m in re.finditer(r'\"type\":\s*\"CAUSED_BY\"', content)]
    print(f"找到 {len(caused_by_positions)} 个CAUSED_BY边")

    for pos in caused_by_positions:
        # 找到这个JSON对象的起始位置
        start = pos - 1
        depth = 0
        while start >= 0:
            if content[start] == '}':
                depth += 1
            elif content[start] == '{':
                if depth == 0:
                    break
                depth -= 1
            start -= 1

        obj_text = content[start:pos+300]
        src_match = re.search(r'"source":\s*"Fault_([^"]+)"', obj_text)
        tgt_match = re.search(r'"target":\s*"Component_([^"]+)"', obj_text)

        if src_match and tgt_match:
            fault_id = src_match.group(1)
            component = tgt_match.group(1)

            if fault_id in fault_id_to_label:
                fault_label = fault_id_to_label[fault_id]
                if fault_label in fault_to_idx:
                    fault_components[fault_label].add(component)
                    all_components.add(component)

    print(f"故障-部件关联数: {sum(len(v) for v in fault_components.values())}")
    for f in FAULT_TYPES:
        if f in fault_components and fault_components[f]:
            print(f"  {f}: {fault_components[f]}")

    print("\nStep 3: 解析故障-特征关联 (HAS_FEATURE)...")
    has_feature_positions = [m.start() for m in re.finditer(r'"type":\s*"HAS_FEATURE"', content)]
    print(f"找到 {len(has_feature_positions)} 个HAS_FEATURE边")

    # 只统计每种故障类型的特征类型（不重复计算）
    for pos in has_feature_positions:
        start = pos - 1
        depth = 0
        while start >= 0:
            if content[start] == '}':
                depth += 1
            elif content[start] == '{':
                if depth == 0:
                    break
                depth -= 1
            start -= 1

        obj_text = content[start:pos+300]
        src_match = re.search(r'"source":\s*"Fault_([^"]+)"', obj_text)
        tgt_match = re.search(r'"target":\s*"Feature_([^_]+)_', obj_text)

        if src_match and tgt_match:
            fault_id = src_match.group(1)
            feature_name = tgt_match.group(1)

            if fault_id in fault_id_to_label:
                fault_label = fault_id_to_label[fault_id]
                if fault_label in fault_to_idx:
                    fault_features[fault_label].add(feature_name)
                    all_feature_labels.add(feature_name)

    print(f"故障-特征关联（唯一特征类型）:")
    for f in FAULT_TYPES:
        if f in fault_features and fault_features[f]:
            print(f"  {f}: {fault_features[f]}")

    # 整理部件和特征列表
    comp_list = sorted(all_components) if all_components else ['Unknown']
    feat_list = sorted(all_feature_labels) if all_feature_labels else ['mean', 'std', 'rms']
    comp_to_idx = {c: i for i, c in enumerate(comp_list)}
    feat_to_idx = {f: i for i, f in enumerate(feat_list)}

    print(f"\n部件类型数: {len(comp_list)}, 列表: {comp_list}")
    print(f"特征类型数: {len(feat_list)}, 列表: {feat_list}")

    # 构建故障-部件矩阵
    n_comp = max(len(comp_list), 1)
    fault_component_matrix = np.zeros((n_faults, n_comp), dtype=np.float32)
    for fault_label, components in fault_components.items():
        if fault_label in fault_to_idx:
            for comp in components:
                if comp in comp_to_idx:
                    fault_component_matrix[fault_to_idx[fault_label], comp_to_idx[comp]] = 1

    # 构建故障-特征矩阵
    n_feat = max(len(feat_list), 1)
    fault_feature_matrix = np.zeros((n_faults, n_feat), dtype=np.float32)
    for fault_label, features in fault_features.items():
        if fault_label in fault_to_idx:
            for feat in features:
                if feat in feat_to_idx:
                    fault_feature_matrix[fault_to_idx[fault_label], feat_to_idx[feat]] = 1

    # 计算故障相似度矩阵 (基于Jaccard相似度)
    fault_similarity = np.zeros((n_faults, n_faults), dtype=np.float32)
    for i in range(n_faults):
        for j in range(n_faults):
            # 基于部件相似度
            set_i = fault_components.get(FAULT_TYPES[i], set())
            set_j = fault_components.get(FAULT_TYPES[j], set())
            if len(set_i | set_j) > 0:
                comp_sim = len(set_i & set_j) / len(set_i | set_j)
            else:
                comp_sim = 0

            # 基于特征相似度
            set_i = fault_features.get(FAULT_TYPES[i], set())
            set_j = fault_features.get(FAULT_TYPES[j], set())
            if len(set_i | set_j) > 0:
                feat_sim = len(set_i & set_j) / len(set_i | set_j)
            else:
                feat_sim = 0

            # 加权平均
            fault_similarity[i, j] = 0.6 * comp_sim + 0.4 * feat_sim

    return {
        'fault_similarity': fault_similarity.tolist(),
        'fault_component_matrix': fault_component_matrix.tolist(),
        'fault_feature_matrix': fault_feature_matrix.tolist(),
        'fault_types': FAULT_TYPES,
        'components': comp_list,
        'feature_labels': feat_list,
        'fault_nodes': dict(fault_nodes)
    }


def save_fault_embeddings(kg_path, output_path):
    """保存故障嵌入矩阵"""
    print(f"\n从 {kg_path} 构建故障嵌入...")
    embeddings = build_fault_embeddings_from_kg(kg_path)

    print(f"\n故障相似度矩阵 (9x9):")
    sim = np.array(embeddings['fault_similarity'])
    for i, ft in enumerate(FAULT_TYPES):
        print(f"  {ft}: {sim[i].round(3)}")

    comp_matrix = np.array(embeddings['fault_component_matrix'])
    feat_matrix = np.array(embeddings['fault_feature_matrix'])
    print(f"\n故障-部件矩阵形状: {comp_matrix.shape}")
    print(f"故障-特征矩阵形状: {feat_matrix.shape}")

    with open(output_path, 'w') as f:
        json.dump(embeddings, f, indent=2)

    print(f"\n已保存至 {output_path}")
    return embeddings


if __name__ == '__main__':
    save_fault_embeddings(
        'data/processed/knowledge_graph.json',
        'data/processed/fault_embeddings.json'
    )