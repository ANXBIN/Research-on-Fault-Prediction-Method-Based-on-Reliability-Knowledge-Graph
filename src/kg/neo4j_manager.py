#!/usr/bin/env python3
"""
Neo4j知识图谱管理脚本：启动容器、清除数据、导入知识图谱
"""

import json
import subprocess
import time
import sys
from pathlib import Path
from neo4j import GraphDatabase
import yaml


class Neo4jManager:
    """Neo4j管理器"""

    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)['neo4j']

        self.uri = self.config['uri']
        self.username = self.config['username']
        self.password = self.config['password']
        self.container_name = self.config['container_name']

    def is_neo4j_running(self):
        """检查Neo4j容器是否运行"""
        result = subprocess.run(
            ['docker', 'ps', '--filter', f'name={self.container_name}', '--format', '{{.Names}}'],
            capture_output=True, text=True
        )
        return self.container_name in result.stdout

    def start_neo4j_container(self, neo4j_password=None):
        """启动Neo4j Docker容器"""
        if self.is_neo4j_running():
            print(f"[INFO] Neo4j容器 {self.container_name} 已在运行")
            return True

        password = neo4j_password or self.password

        print(f"[INFO] 启动Neo4j容器 {self.container_name}...")

        # 检查是否存在旧容器
        result = subprocess.run(
            ['docker', 'ps', '-a', '--filter', f'name={self.container_name}', '--format', '{{.Names}}'],
            capture_output=True, text=True
        )

        if self.container_name in result.stdout:
            print(f"[INFO] 移除旧容器...")
            subprocess.run(['docker', 'rm', '-f', self.container_name],
                         capture_output=True)

        # 启动新容器
        cmd = [
            'docker', 'run', '-d',
            '--name', self.container_name,
            '-p', '7474:7474',
            '-p', '7687:7687',
            '-e', f'NEO4J_AUTH=neo4j/{password}',
            '-e', 'NEO4J_PLUGINS=["apoc"]',
            '-v', f'{self.container_name}_data:/data',
            '-v', f'{self.container_name}_logs:/logs',
            '-v', f'{self.container_name}_plugins:/plugins',
            'neo4j:5.16.0'
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"[ERROR] 启动容器失败: {result.stderr}")
            return False

        # 等待Neo4j启动
        print("[INFO] 等待Neo4j服务启动...")
        time.sleep(30)

        # 验证连接
        for i in range(10):
            try:
                driver = GraphDatabase.driver(
                    self.uri,
                    auth=(self.username, password)
                )
                with driver.session() as session:
                    session.run("RETURN 1")
                driver.close()
                print("[INFO] Neo4j连接成功!")
                return True
            except Exception as e:
                print(f"[INFO] 等待连接... ({i+1}/10)")
                time.sleep(5)

        print("[ERROR] Neo4j连接失败")
        return False

    def stop_neo4j_container(self):
        """停止Neo4j容器"""
        if not self.is_neo4j_running():
            print(f"[INFO] Neo4j容器 {self.container_name} 未运行")
            return True

        print(f"[INFO] 停止Neo4j容器 {self.container_name}...")
        subprocess.run(['docker', 'stop', self.container_name],
                      capture_output=True)
        return True

    def clear_database(self):
        """清除数据库中的所有数据"""
        print("[INFO] 清除Neo4j数据库...")

        driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
        with driver.session() as session:
            # 删除所有节点和关系
            session.run("MATCH (n) DETACH DELETE n")

        driver.close()
        print("[INFO] 数据库已清除")
        return True

    def get_driver(self):
        """获取Neo4j驱动"""
        return GraphDatabase.driver(self.uri, auth=(self.username, self.password))

    def import_knowledge_graph(self, kg_file_path):
        """导入知识图谱数据到Neo4j"""
        print(f"[INFO] 导入知识图谱: {kg_file_path}")

        with open(kg_file_path, 'r', encoding='utf-8') as f:
            kg_data = json.load(f)

        driver = self.get_driver()

        with driver.session() as session:
            # 创建节点
            print(f"[INFO] 创建 {len(kg_data['nodes'])} 个节点...")
            for node in kg_data['nodes']:
                labels = node['type']
                node_id = node['id']
                properties = node.get('properties', {})

                # 构建CQL
                prop_str = ', '.join([f"{k}: ${k}" for k in properties.keys()])
                if prop_str:
                    prop_str = ', ' + prop_str

                cql = f"CREATE (n:{labels} {{id: $id {prop_str}}})"

                params = {'id': node_id, **properties}
                session.run(cql, params)

            # 创建关系
            print(f"[INFO] 创建 {len(kg_data['edges'])} 条关系...")
            for edge in kg_data['edges']:
                source = edge['source']
                target = edge['target']
                rel_type = edge['type']
                properties = edge.get('properties', {})

                prop_str = ', '.join([f"{k}: ${k}" for k in properties.keys()])
                if prop_str:
                    prop_str = ', ' + prop_str

                cql = f"""
                MATCH (a), (b)
                WHERE a.id = $source AND b.id = $target
                CREATE (a)-[r:{rel_type} {{id: $rel_id {prop_str}}}]->(b)
                """

                params = {
                    'source': source,
                    'target': target,
                    'rel_id': f"{source}_{rel_type}_{target}",
                    **properties
                }
                session.run(cql, params)

        driver.close()
        print("[INFO] 知识图谱导入完成")
        return True

    def create_indexes(self):
        """创建索引以提高查询性能"""
        print("[INFO] 创建索引...")

        driver = self.get_driver()
        with driver.session() as session:
            # 为Fault节点创建索引
            session.run("CREATE INDEX fault_id_index IF NOT EXISTS FOR (n:Fault) ON (n.id)")
            session.run("CREATE INDEX fault_label_index IF NOT EXISTS FOR (n:Fault) ON (n.label)")

            # 为Component节点创建索引
            session.run("CREATE INDEX component_id_index IF NOT EXISTS FOR (n:Component) ON (n.id)")

            # 为Feature节点创建索引
            session.run("CREATE INDEX feature_id_index IF NOT EXISTS FOR (n:Feature) ON (n.id)")

        driver.close()
        print("[INFO] 索引创建完成")
        return True

    def verify_import(self):
        """验证导入结果"""
        print("[INFO] 验证导入结果...")

        driver = self.get_driver()
        with driver.session() as session:
            # 统计节点数量
            result = session.run("MATCH (n) RETURN labels(n)[0] as type, count(*) as count")
            node_counts = {record['type']: record['count'] for record in result}
            print(f"[INFO] 节点统计: {node_counts}")

            # 统计关系数量
            result = session.run("MATCH ()-[r]->() RETURN type(r) as type, count(*) as count")
            edge_counts = {record['type']: record['count'] for record in result}
            print(f"[INFO] 关系统计: {edge_counts}")

        driver.close()
        return node_counts, edge_counts

    def export_graph_structure(self, output_path):
        """导出图结构用于模型训练"""
        print(f"[INFO] 导出图结构至: {output_path}")

        driver = self.get_driver()
        graph_data = {
            'nodes': [],
            'edges': []
        }

        with driver.session() as session:
            # 导出所有节点
            result = session.run("""
                MATCH (n)
                RETURN id(n) as neo_id, labels(n)[0] as type, n.id as node_id, n.label as label
            """)
            for record in result:
                graph_data['nodes'].append({
                    'neo_id': record['neo_id'],
                    'type': record['type'],
                    'node_id': record['node_id'],
                    'label': record['label']
                })

            # 导出所有关系
            result = session.run("""
                MATCH (a)-[r]->(b)
                RETURN id(a) as source_id, id(b) as target_id, type(r) as rel_type
            """)
            for record in result:
                graph_data['edges'].append({
                    'source': record['source_id'],
                    'target': record['target_id'],
                    'type': record['rel_type']
                })

        driver.close()

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, indent=2)

        return graph_data


def main():
    """主函数"""
    import argparse
    parser = argparse.ArgumentParser(description='Neo4j知识图谱管理')
    parser.add_argument('--action', type=str, required=True,
                       choices=['start', 'stop', 'clear', 'import', 'full'],
                       help='操作类型')
    parser.add_argument('--kg_file', type=str,
                       default='data/processed/knowledge_graph.json',
                       help='知识图谱文件路径')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='配置文件路径')
    parser.add_argument('--password', type=str, default='password',
                       help='Neo4j密码')
    args = parser.parse_args()

    manager = Neo4jManager(config_path=args.config)

    if args.action == 'start':
        manager.start_neo4j_container(neo4j_password=args.password)

    elif args.action == 'stop':
        manager.stop_neo4j_container()

    elif args.action == 'clear':
        if not manager.is_neo4j_running():
            print("[ERROR] Neo4j容器未运行")
            sys.exit(1)
        manager.clear_database()

    elif args.action == 'import':
        if not manager.is_neo4j_running():
            print("[ERROR] Neo4j容器未运行")
            sys.exit(1)
        manager.clear_database()
        manager.import_knowledge_graph(args.kg_file)
        manager.create_indexes()
        manager.verify_import()

    elif args.action == 'full':
        # 完整流程：启动 -> 清除 -> 导入 -> 验证
        manager.start_neo4j_container(neo4j_password=args.password)
        time.sleep(10)
        manager.clear_database()
        manager.import_knowledge_graph(args.kg_file)
        manager.create_indexes()
        manager.verify_import()

        # 导出图结构
        output_path = 'data/processed/graph_structure.json'
        manager.export_graph_structure(output_path)


if __name__ == '__main__':
    main()
