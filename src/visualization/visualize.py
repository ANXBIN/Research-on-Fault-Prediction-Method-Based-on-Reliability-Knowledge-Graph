#!/usr/bin/env python3
"""
可视化脚本：生成训练结果可视化图表
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class ResultsVisualizer:
    """结果可视化器"""

    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.viz_config = self.config['visualization']
        self.figure_size = tuple(self.viz_config['figure_size'])
        self.dpi = self.viz_config['dpi']

        sns.set_style(self.viz_config.get('style', 'whitegrid'))

        # 输出目录
        self.output_dir = Path('results/figures')
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_results(self, results_path='results/training_results.json'):
        """加载训练结果"""
        with open(results_path, 'r', encoding='utf-8') as f:
            self.results = json.load(f)

        return self.results

    def plot_model_comparison(self, save_path=None):
        """绘制模型对比图"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        models = list(self.results.get('models', {}).keys())
        if not models:
            print("[WARNING] 没有模型结果可可视化")
            return

        train_accs = []
        val_accs = []
        val_f1s = []

        for model in models:
            m = self.results['models'][model]
            train_accs.append(m.get('train_accuracy', 0))
            val_accs.append(m.get('val_accuracy', 0))
            val_f1s.append(m.get('val_f1', 0))

        # 准确率对比
        x = np.arange(len(models))
        width = 0.35

        axes[0].bar(x - width/2, train_accs, width, label='Train Accuracy', color='steelblue')
        axes[0].bar(x + width/2, val_accs, width, label='Val Accuracy', color='coral')
        axes[0].set_xlabel('Model')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_title('Model Accuracy Comparison')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(models, rotation=15)
        axes[0].legend()
        axes[0].set_ylim(0, 1.1)

        # F1分数对比
        axes[1].bar(x, val_f1s, width, color='seagreen')
        axes[1].set_xlabel('Model')
        axes[1].set_ylabel('F1 Score')
        axes[1].set_title('Validation F1 Score Comparison')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(models, rotation=15)
        axes[1].set_ylim(0, 1.1)

        # 测试集结果
        test_results = self.results.get('test_results', {})
        if test_results:
            test_accs = [test_results.get(m, {}).get('accuracy', 0) for m in models]
            test_f1s = [test_results.get(m, {}).get('f1', 0) for m in models]

            axes[2].bar(x - width/2, test_accs, width, label='Test Accuracy', color='steelblue')
            axes[2].bar(x + width/2, test_f1s, width, label='Test F1', color='coral')
            axes[2].set_xlabel('Model')
            axes[2].set_ylabel('Score')
            axes[2].set_title('Test Set Performance')
            axes[2].set_xticks(x)
            axes[2].set_xticklabels(models, rotation=15)
            axes[2].legend()
            axes[2].set_ylim(0, 1.1)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"[INFO] 图表已保存至: {save_path}")

        return fig

    def plot_confusion_matrix(self, y_true, y_pred, labels, save_path=None):
        """绘制混淆矩阵"""
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots(figsize=self.figure_size)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=labels, yticklabels=labels)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title('Confusion Matrix')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"[INFO] 混淆矩阵已保存至: {save_path}")

        return fig

    def plot_feature_importance(self, feature_importance, top_n=20, save_path=None):
        """绘制特征重要性图"""
        # 排序
        sorted_features = sorted(feature_importance.items(),
                                key=lambda x: x[1], reverse=True)[:top_n]

        features = [f[0] for f in sorted_features]
        importance = [f[1] for f in sorted_features]

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(range(len(features)), importance, color='steelblue')
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features)
        ax.invert_yaxis()
        ax.set_xlabel('Feature Importance')
        ax.set_title(f'Top {top_n} Feature Importance')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"[INFO] 特征重要性图已保存至: {save_path}")

        return fig

    def plot_training_history(self, history, save_path=None):
        """绘制训练历史"""
        if not history:
            return None

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # 损失曲线
        if 'loss' in history:
            axes[0].plot(history['loss'], label='Train Loss')
            if 'val_loss' in history:
                axes[0].plot(history['val_loss'], label='Val Loss')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].set_title('Training Loss')
            axes[0].legend()

        # 准确率曲线
        if 'accuracy' in history:
            axes[1].plot(history['accuracy'], label='Train Acc')
            if 'val_accuracy' in history:
                axes[1].plot(history['val_accuracy'], label='Val Acc')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Accuracy')
            axes[1].set_title('Training Accuracy')
            axes[1].legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"[INFO] 训练历史图已保存至: {save_path}")

        return fig

    def plot_kg_structure(self, kg_data_path='data/processed/kg_embeddings.json', save_path=None):
        """绘制知识图谱结构"""
        try:
            with open(kg_data_path, 'r') as f:
                kg_data = json.load(f)
        except FileNotFoundError:
            print(f"[WARNING] KG数据文件不存在: {kg_data_path}")
            return None

        structural_features = kg_data.get('structural_features', {})

        if not structural_features:
            print("[WARNING] 没有结构特征数据")
            return None

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 节点类型分布
        node_counts = structural_features.get('node_counts', {})
        if node_counts:
            axes[0].pie(node_counts.values(), labels=node_counts.keys(),
                       autopct='%1.1f%%', startangle=90)
            axes[0].set_title('Node Type Distribution')

        # 度数分布
        avg_degree = structural_features.get('avg_degree', {})
        if avg_degree:
            axes[1].bar(avg_degree.keys(), avg_degree.values(), color='seagreen')
            axes[1].set_xlabel('Node Type')
            axes[1].set_ylabel('Average Degree')
            axes[1].set_title('Average Degree by Node Type')
            axes[1].tick_params(axis='x', rotation=15)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"[INFO] KG结构图已保存至: {save_path}")

        return fig

    def plot_signal_sample(self, data_path='data/processed/processed_features.csv', save_path=None):
        """绘制信号样本"""
        try:
            df = pd.read_csv(data_path)
        except FileNotFoundError:
            print(f"[WARNING] 数据文件不存在: {data_path}")
            return None

        fault_types = df['fault_type'].unique()

        # 选择每种故障类型的一个样本
        n_types = min(len(fault_types), 6)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for i, fault_type in enumerate(fault_types[:n_types]):
            sample = df[df['fault_type'] == fault_type].iloc[0]

            # 使用RMS作为y轴示例
            rms_values = df[df['fault_type'] == fault_type]['rms'].values[:50]

            axes[i].plot(rms_values, color='steelblue')
            axes[i].set_title(fault_type)
            axes[i].set_xlabel('Sample Index')
            axes[i].set_ylabel('RMS')

        plt.suptitle('Signal Samples by Fault Type')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"[INFO] 信号样本图已保存至: {save_path}")

        return fig

    def generate_all_visualizations(self):
        """生成所有可视化图表"""
        print("\n" + "=" * 60)
        print("生成可视化图表")
        print("=" * 60)

        # 模型对比图
        try:
            self.plot_model_comparison(
                save_path=self.output_dir / 'model_comparison.png'
            )
        except Exception as e:
            print(f"[ERROR] 生成模型对比图失败: {e}")

        # KG结构图
        try:
            self.plot_kg_structure(
                kg_data_path='data/processed/kg_embeddings.json',
                save_path=self.output_dir / 'kg_structure.png'
            )
        except Exception as e:
            print(f"[ERROR] 生成KG结构图失败: {e}")

        # 特征重要性（如果有RF模型结果）
        try:
            if 'models' in self.results and 'RF' in self.results['models']:
                rf_results = self.results['models']['RF']
                if 'top_features' in rf_results:
                    importance_dict = dict(rf_results['top_features'])
                    self.plot_feature_importance(
                        importance_dict,
                        save_path=self.output_dir / 'feature_importance.png'
                    )
        except Exception as e:
            print(f"[ERROR] 生成特征重要性图失败: {e}")

        # 信号样本图
        try:
            self.plot_signal_sample(
                data_path='data/processed/processed_features.csv',
                save_path=self.output_dir / 'signal_samples.png'
            )
        except Exception as e:
            print(f"[ERROR] 生成信号样本图失败: {e}")

        print(f"\n所有图表已保存至: {self.output_dir}")

    def create_summary_report(self, save_path='results/visualization_report.html'):
        """创建可视化总结报告（HTML）"""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>故障预测系统 - 训练结果报告</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #333; }}
        h2 {{ color: #666; border-bottom: 1px solid #ccc; padding-bottom: 10px; }}
        .section {{ margin: 30px 0; }}
        img {{ max-width: 100%; border: 1px solid #ddd; margin: 10px 0; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <h1>基于知识图谱的故障预测系统 - 训练结果报告</h1>

    <div class="section">
        <h2>训练信息</h2>
        <p>训练时间: {self.results.get('timestamp', 'N/A')}</p>
    </div>

    <div class="section">
        <h2>模型性能对比</h2>
        <img src="figures/model_comparison.png" alt="模型对比">
    </div>

    <div class="section">
        <h2>知识图谱结构</h2>
        <img src="figures/kg_structure.png" alt="KG结构">
    </div>

    <div class="section">
        <h2>特征重要性</h2>
        <img src="figures/feature_importance.png" alt="特征重要性">
    </div>

    <div class="section">
        <h2>信号样本</h2>
        <img src="figures/signal_samples.png" alt="信号样本">
    </div>

    <div class="section">
        <h2>详细结果</h2>
        <table>
            <tr>
                <th>模型</th>
                <th>训练准确率</th>
                <th>验证准确率</th>
                <th>验证F1</th>
            </tr>
"""

        for model_name, model_results in self.results.get('models', {}).items():
            html_content += f"""
            <tr>
                <td>{model_name}</td>
                <td>{model_results.get('train_accuracy', 0):.4f}</td>
                <td>{model_results.get('val_accuracy', 0):.4f}</td>
                <td>{model_results.get('val_f1', 0):.4f}</td>
            </tr>
"""

        html_content += """
        </table>
    </div>

    <div class="section">
        <h2>结论</h2>
        <p>本报告对比了传统机器学习方法（RF）和知识图谱增强方法（KG-Enhanced RF）
           在变速箱故障预测任务上的性能表现。</p>
    </div>
</body>
</html>
"""

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"[INFO] HTML报告已生成: {save_path}")
        return save_path


def main():
    """主函数"""
    import argparse
    parser = argparse.ArgumentParser(description='生成可视化结果')
    parser.add_argument('--results', type=str,
                       default='results/training_results.json',
                       help='训练结果JSON文件路径')
    parser.add_argument('--output_dir', type=str,
                       default='results/figures',
                       help='输出目录')
    parser.add_argument('--all', action='store_true',
                       help='生成所有可视化')
    args = parser.parse_args()

    visualizer = ResultsVisualizer()

    if Path(args.results).exists():
        visualizer.load_results(args.results)
    else:
        print(f"[WARNING] 结果文件不存在: {args.results}")
        print("[INFO] 尝试从处理后的数据生成可视化...")

    if args.all:
        visualizer.generate_all_visualizations()
        visualizer.create_summary_report()
    else:
        # 生成默认图表
        visualizer.plot_model_comparison(
            save_path=visualizer.output_dir / 'model_comparison.png'
        )
        visualizer.plot_signal_sample(
            data_path='data/processed/processed_features.csv',
            save_path=visualizer.output_dir / 'signal_samples.png'
        )

    print("\n" + "=" * 60)
    print("可视化生成完成!")
    print("=" * 60)


if __name__ == '__main__':
    main()
