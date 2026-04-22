#!/usr/bin/env python3
"""
知识图谱嵌入模块：从Neo4j提取结构信息，生成嵌入向量
"""

import json
import numpy as np
from pathlib import Path
from neo4j import GraphDatabase
import yaml
from collections import defaultdict
import torch
import torch.nn as nn


class KGEmbedder:
    """知识图谱嵌入生成器"""

    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        neo4j_config = self.config['neo4j']
        self.uri = neo4j_config['uri']
        self.username = neo4j_config['username']
        self.password = neo4j_config['password']

        self.kg_config = self.config['knowledge_graph']
        self.embedding_dim = self.config['models']['kg_enhanced_gnn']['kg_embedding_dim']

    def get_driver(self):
        return GraphDatabase.driver(self.uri, auth=(self.username, self.password))

    def extract_subgraph(self, fault_node_id):
        """提取与故障节点相关的子图"""
        driver = self.get_driver()
        subgraph = {'nodes': [], 'edges': [], 'features': {}}

        with driver.session() as session:
            # 提取一跳邻居
            result = session.run("""
                MATCH (f:Fault {id: $fault_id})-[r1]-(n)-[r2]-(m)
                RETURN f, r1, n, r2, m
                LIMIT 1000
            """, fault_id=fault_node_id)

            for record in result:
                # 处理节点和关系
                pass

        driver.close()
        return subgraph

    def compute_structural_features(self):
        """计算图的结构特征"""
        print("[INFO] 计算知识图谱结构特征...")

        driver = self.get_driver()
        features = {
            'node_counts': {},
            'edge_counts': {},
            'avg_degree': {},
            'component_distribution': {}
        }

        with driver.session() as session:
            # 统计各类节点数量
            result = session.run("""
                MATCH (n)
                RETURN labels(n)[0] as type, count(*) as count
            """)
            features['node_counts'] = {r['type']: r['count'] for r in result}

            # 统计各类关系数量
            result = session.run("""
                MATCH ()-[r]->()
                RETURN type(r) as type, count(*) as count
            """)
            features['edge_counts'] = {r['type']: r['count'] for r in result}

            # 计算各类节点的平均度数
            result = session.run("""
                MATCH (n)-[r]-(m)
                WITH labels(n)[0] as type, count(r) as degree
                RETURN type, avg(degree) as avg_degree
            """)
            features['avg_degree'] = {r['type']: r['avg_degree'] for r in result}

        driver.close()
        return features

    def build_adjacency_matrix(self):
        """构建邻接矩阵"""
        print("[INFO] 构建邻接矩阵...")

        driver = self.get_driver()
        edges = []

        with driver.session() as session:
            result = session.run("""
                MATCH (a)-[r]->(b)
                RETURN id(a) as source, id(b) as target, type(r) as rel_type
            """)

            for record in result:
                edges.append({
                    'source': record['source'],
                    'target': record['target'],
                    'type': record['rel_type']
                })

        driver.close()

        # 提取所有节点ID
        all_nodes = set()
        for edge in edges:
            all_nodes.add(edge['source'])
            all_nodes.add(edge['target'])

        node_to_idx = {node: idx for idx, node in enumerate(sorted(all_nodes))}
        n_nodes = len(node_to_idx)

        # 构建邻接矩阵
        adj_matrix = np.zeros((n_nodes, n_nodes))
        for edge in edges:
            src_idx = node_to_idx[edge['source']]
            tgt_idx = node_to_idx[edge['target']]
            adj_matrix[src_idx][tgt_idx] = 1
            adj_matrix[tgt_idx][src_idx] = 1  # 无向化

        return adj_matrix, node_to_idx

    def generate_node_embeddings(self, adj_matrix, method='spectral'):
        """生成节点嵌入"""
        print(f"[INFO] 使用{method}方法生成节点嵌入...")

        if method == 'spectral':
            # 谱嵌入
            from scipy.sparse.linalg import eigsh

            # 计算度矩阵
            degrees = np.sum(adj_matrix, axis=1)
            D = np.diag(degrees)

            # 计算拉普拉斯矩阵
            L = D - adj_matrix

            # 计算最小的k个特征值和特征向量
            k = min(self.embedding_dim, len(adj_matrix) - 1)
            eigenvalues, eigenvectors = eigsh(L, k=k+1, which='SM')

            # 跳过第一个特征向量（对应特征值0）
            embeddings = eigenvectors[:, 1:k+1]

        elif method == 'random_walk':
            # 随机游走嵌入（简化版）
            n_nodes = len(adj_matrix)
            embeddings = np.random.randn(n_nodes, self.embedding_dim)

            # 归一化
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / (norms + 1e-8)

        else:
            raise ValueError(f"Unknown embedding method: {method}")

        return embeddings

    def get_fault_embeddings(self, fault_labels):
        """获取故障节点的嵌入"""
        print("[INFO] 获取故障节点嵌入...")

        driver = self.get_driver()
        adj_matrix, node_to_idx = self.build_adjacency_matrix()
        node_embeddings = self.generate_node_embeddings(adj_matrix)

        fault_embeddings = {}

        with driver.session() as session:
            for fault_label in fault_labels:
                result = session.run("""
                    MATCH (f:Fault {label: $label})
                    RETURN f.id as node_id
                    LIMIT 10
                """, label=fault_label)

                embeddings = []
                for record in result:
                    node_id = record['node_id']
                    # 通过node_id找到对应的索引
                    # 这里需要从图中获取实际的neo4j内部id
                    pass

        driver.close()
        return fault_embeddings

    def compute_fault_similarity(self):
        """计算故障类型之间的相似度"""
        print("[INFO] 计算故障相似度...")

        driver = self.get_driver()
        similarity_matrix = {}

        with driver.session() as session:
            # 获取所有故障类型
            result = session.run("""
                MATCH (f:Fault)
                RETURN DISTINCT f.label as label, count(*) as count
            """)
            fault_types = [r['label'] for r in result]

            # 对每对故障类型计算共享的邻居特征
            for i, type1 in enumerate(fault_types):
                for type2 in fault_types[i+1:]:
                    result = session.run("""
                        MATCH (f1:Fault {label: $label1})-[r1]-(n)-[r2]-(f2:Fault {label: $label2})
                        RETURN count(DISTINCT n) as shared_neighbors
                    """, label1=type1, label2=type2)

                    shared = result.single()['shared_neighbors']
                    similarity_matrix[(type1, type2)] = shared

        driver.close()
        return similarity_matrix

    def export_embeddings(self, output_path):
        """导出嵌入向量"""
        adj_matrix, node_to_idx = self.build_adjacency_matrix()
        embeddings = self.generate_node_embeddings(adj_matrix)

        structural_features = self.compute_structural_features()
        similarity_matrix = self.compute_fault_similarity()

        output_data = {
            'adjacency_matrix': adj_matrix.tolist(),
            'node_to_idx': node_to_idx,
            'node_embeddings': embeddings.tolist(),
            'structural_features': structural_features,
            'fault_similarity': similarity_matrix
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)

        print(f"[INFO] 嵌入已保存至: {output_path}")
        return output_data


class KGEnhancedFeatures:
    """知识图谱增强特征生成器"""

    def __init__(self, embedding_path):
        with open(embedding_path, 'r', encoding='utf-8') as f:
            self.kg_data = json.load(f)

        self.node_embeddings = np.array(self.kg_data['node_embeddings'])
        self.adjacency_matrix = np.array(self.kg_data['adjacency_matrix'])
        self.node_to_idx = self.kg_data['node_to_idx']
        self.structural_features = self.kg_data.get('structural_features', {})
        self.fault_similarity = self.kg_data.get('fault_similarity', {})

    def get_kg_enhanced_features(self, original_features, fault_type):
        """基于原始特征和知识图谱嵌入生成增强特征"""
        # 知识图谱结构特征
        kg_features = []

        # 添加结构特征
        if 'node_counts' in self.structural_features:
            for node_type in ['Fault', 'Component', 'Feature']:
                count = self.structural_features['node_counts'].get(node_type, 0)
                kg_features.append(count / 1000)  # 归一化

        # 添加度数特征
        if 'avg_degree' in self.structural_features:
            for node_type in ['Fault', 'Component', 'Feature']:
                degree = self.structural_features['avg_degree'].get(node_type, 0)
                kg_features.append(degree / 10)

        # 添加相似度特征
        fault_types = list(set([v for v in self.fault_similarity.keys()]))
        similarity_sum = sum([v for v in self.fault_similarity.values()])
        kg_features.append(similarity_sum / 1000)

        # 填充到固定长度
        target_len = 20
        while len(kg_features) < target_len:
            kg_features.append(0)

        kg_features = kg_features[:target_len]

        # 拼接原始特征和知识图谱特征
        enhanced_features = np.concatenate([
            original_features,
            np.array(kg_features)
        ])

        return enhanced_features


def main():
    """主函数"""
    import argparse
    parser = argparse.ArgumentParser(description='生成知识图谱嵌入')
    parser.add_argument('--embedding_path', type=str,
                       default='data/processed/kg_embeddings.json',
                       help='嵌入输出路径')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='配置文件路径')
    args = parser.parse_args()

    print("=" * 60)
    print("知识图谱嵌入生成")
    print("=" * 60)

    embedder = KGEmbedder(config_path=args.config)
    embedder.export_embeddings(args.embedding_path)

    print("\n" + "=" * 60)
    print("嵌入生成完成!")
    print("=" * 60)


if __name__ == '__main__':
    main()
