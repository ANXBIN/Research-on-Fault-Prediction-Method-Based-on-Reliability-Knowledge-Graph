#!/usr/bin/env python3
"""
Visualize evaluation results
Reads results/evaluation_results.json and generates comparison charts
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['font.sans-serif'] = ['Heiti SC', 'Heiti TC', 'STHeiti Medium', 'Songti SC', 'Songti TC', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_evaluation_results(results_path='results/evaluation_results.json'):
    """Load evaluation results"""
    with open(results_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def plot_model_comparison(results, save_path='results/figures/model_comparison.png'):
    """Plot model accuracy comparison"""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    models = list(results['validation'].keys())
    val_acc = [results['validation'][m]['accuracy'] for m in models]
    test_acc = [results['test'][m]['accuracy'] for m in models]
    val_f1 = [results['validation'][m]['f1'] for m in models]
    test_f1 = [results['test'][m]['f1'] for m in models]

    # Model name mapping
    name_map = {
        'MLP': 'MLP\n(Baseline)',
        'MLP_KG': 'MLP-KG\n(Global Embedding)',
        'MLP_KG_V2': 'MLP-KG-V2\n(Fault-Level Embedding)',
        'CNN': 'CNN\n(1D Conv)',
        'CNN_KG': 'CNN+KG\n(拼接融合)',
        'CNN_KG_V2': 'CNN+KG V2\n(门控融合)',
        'CNN_KG_V3': 'CNN+KG V3\n(残差连接)'
    }
    model_labels = [name_map.get(m, m) for m in models]

    x = np.arange(len(models))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy comparison
    ax1 = axes[0]
    bars1 = ax1.bar(x - width/2, val_acc, width, label='Validation', color='#4ECDC4', alpha=0.8, edgecolor='white')
    bars2 = ax1.bar(x + width/2, test_acc, width, label='Test', color='#FF6B6B', alpha=0.8, edgecolor='white')

    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_labels, fontsize=9)
    ax1.legend(loc='lower right', fontsize=10)
    ax1.set_ylim(0, 1.05)
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax1.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

    # F1 score comparison
    ax2 = axes[1]
    bars3 = ax2.bar(x - width/2, val_f1, width, label='Validation', color='#4ECDC4', alpha=0.8, edgecolor='white')
    bars4 = ax2.bar(x + width/2, test_f1, width, label='Test', color='#FF6B6B', alpha=0.8, edgecolor='white')

    ax2.set_ylabel('F1 Score', fontsize=12)
    ax2.set_title('F1 Score Comparison', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_labels, fontsize=9)
    ax2.legend(loc='lower right', fontsize=10)
    ax2.set_ylim(0, 1.05)
    ax2.grid(axis='y', alpha=0.3)

    for bar in bars3:
        height = bar.get_height()
        ax2.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    for bar in bars4:
        height = bar.get_height()
        ax2.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def plot_heatmap_comparison(results, save_path='results/figures/metrics_heatmap.png'):
    """Plot metrics heatmap"""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    models = list(results['validation'].keys())
    name_map = {
        'MLP': 'MLP',
        'MLP_KG': 'MLP-KG',
        'MLP_KG_V2': 'MLP-KG-V2',
        'CNN': 'CNN',
        'CNN_KG': 'CNN+KG',
        'CNN_KG_V2': 'CNN+KG V2',
        'CNN_KG_V3': 'CNN+KG V3'
    }
    model_labels = [name_map.get(m, m) for m in models]

    # Build data matrix
    data = np.array([
        [results['validation'][m]['accuracy'], results['test'][m]['accuracy']] for m in models
    ])

    fig, ax = plt.subplots(figsize=(8, len(models) * 0.8 + 2))

    sns.heatmap(data, annot=True, fmt='.4f', cmap='RdYlGn',
                xticklabels=['Val Accuracy', 'Test Accuracy'],
                yticklabels=model_labels,
                ax=ax, vmin=0.5, vmax=1.0,
                annot_kws={'size': 12})

    ax.set_title('Model Accuracy Heatmap', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def plot_improvement_bar(results, save_path='results/figures/improvement_comparison.png'):
    """Plot improvement relative to baseline MLP"""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    models = list(results['validation'].keys())
    mlp_val_acc = results['validation'].get('MLP', {}).get('accuracy', 0)
    mlp_test_acc = results['test'].get('MLP', {}).get('accuracy', 0)

    # Calculate improvement
    improvements = []
    for m in models:
        if m == 'MLP':
            improvements.append(0)
        else:
            val_imp = results['validation'][m]['accuracy'] - mlp_val_acc
            test_imp = results['test'][m]['accuracy'] - mlp_test_acc
            improvements.append((val_imp + test_imp) / 2)

    name_map = {
        'MLP': 'MLP\n(Baseline)',
        'MLP_KG': 'MLP-KG',
        'MLP_KG_V2': 'MLP-KG-V2',
        'CNN': 'CNN',
        'CNN_KG': 'CNN+KG',
        'CNN_KG_V2': 'CNN+KG V2',
        'CNN_KG_V3': 'CNN+KG V3'
    }
    model_labels = [name_map.get(m, m) for m in models]

    fig, ax = plt.subplots(figsize=(10, 5))

    colors = ['#95a5a6'] + ['#27ae60' if i > 0 else '#95a5a6' for i in improvements[1:]]
    bars = ax.bar(model_labels, [i * 100 for i in improvements], color=colors, alpha=0.8, edgecolor='white')

    ax.set_ylabel('Improvement vs MLP (%)', fontsize=12)
    ax.set_title('Knowledge Graph Enhancement Effect', fontsize=14, fontweight='bold')
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.grid(axis='y', alpha=0.3)

    for bar, imp in zip(bars, improvements):
        height = bar.get_height()
        ax.annotate(f'{height:.2f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3 if height >= 0 else -15),
                    textcoords="offset points",
                    ha='center', va='bottom' if height >= 0 else 'top',
                    fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def generate_summary_report(results, save_path='results/figures/summary_report.png'):
    """Generate comprehensive report figure"""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    models = list(results['validation'].keys())
    name_map = {
        'MLP': 'MLP',
        'MLP_KG': 'MLP-KG',
        'MLP_KG_V2': 'MLP-KG-V2',
        'CNN': 'CNN',
        'CNN_KG': 'CNN+KG',
        'CNN_KG_V2': 'CNN+KG V2',
        'CNN_KG_V3': 'CNN+KG V3'
    }
    model_labels = [name_map.get(m, m) for m in models]

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

    # 1. Accuracy comparison (main chart)
    ax1 = fig.add_subplot(gs[0, :2])
    x = np.arange(len(models))
    width = 0.35
    val_acc = [results['validation'][m]['accuracy'] for m in models]
    test_acc = [results['test'][m]['accuracy'] for m in models]

    bars1 = ax1.bar(x - width/2, val_acc, width, label='Validation', color='#3498db', alpha=0.8, edgecolor='white')
    bars2 = ax1.bar(x + width/2, test_acc, width, label='Test', color='#e74c3c', alpha=0.8, edgecolor='white')

    ax1.set_ylabel('Accuracy', fontsize=11)
    ax1.set_title('Model Accuracy Comparison', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_labels, fontsize=10)
    ax1.legend(loc='lower right', fontsize=10)
    ax1.set_ylim(0.5, 1.0)
    ax1.grid(axis='y', alpha=0.3)

    # Mark best model
    best_idx = np.argmax(test_acc)
    ax1.annotate(f'Best: {test_acc[best_idx]:.3f}',
                xy=(x[best_idx] + width/2, test_acc[best_idx]),
                xytext=(0, 10), textcoords="offset points",
                ha='center', fontsize=10, color='red', fontweight='bold')

    # 2. F1 Score comparison
    ax2 = fig.add_subplot(gs[0, 2])
    val_f1 = [results['validation'][m]['f1'] for m in models]
    test_f1 = [results['test'][m]['f1'] for m in models]

    x = np.arange(len(models))
    ax2.barh(x - 0.2, val_f1, 0.4, label='Val', color='#3498db', alpha=0.8)
    ax2.barh(x + 0.2, test_f1, 0.4, label='Test', color='#e74c3c', alpha=0.8)
    ax2.set_xlabel('F1 Score', fontsize=11)
    ax2.set_title('F1 Score', fontsize=13, fontweight='bold')
    ax2.set_yticks(x)
    ax2.set_yticklabels(model_labels, fontsize=10)
    ax2.set_xlim(0.5, 1.0)
    ax2.legend(loc='lower right', fontsize=9)
    ax2.grid(axis='x', alpha=0.3)

    # 3. Improvement percentage
    ax3 = fig.add_subplot(gs[1, 0])
    mlp_test_acc = results['test'].get('MLP', {}).get('accuracy', 0)
    improvements = []
    for m in models:
        if m == 'MLP':
            improvements.append(0)
        else:
            improvements.append((results['test'][m]['accuracy'] - mlp_test_acc) * 100)

    colors = ['#95a5a6'] + ['#27ae60' if i > 0 else '#e74c3c' for i in improvements[1:]]
    ax3.bar(model_labels[1:], improvements[1:], color=colors[1:], alpha=0.8, edgecolor='white')
    ax3.set_ylabel('Improvement vs MLP (%)', fontsize=10)
    ax3.set_title('KG Enhancement Effect', fontsize=13, fontweight='bold')
    ax3.axhline(y=0, color='black', linewidth=0.5)
    ax3.grid(axis='y', alpha=0.3)
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=20, ha='right', fontsize=9)

    # 4. Results table
    ax4 = fig.add_subplot(gs[1, 1:])
    ax4.axis('off')

    table_data = []
    for m in models:
        table_data.append([
            name_map.get(m, m),
            f"{results['validation'][m]['accuracy']:.4f}",
            f"{results['test'][m]['accuracy']:.4f}",
            f"{results['validation'][m]['f1']:.4f}",
            f"{results['test'][m]['f1']:.4f}",
        ])

    table = ax4.table(
        cellText=table_data,
        colLabels=['Model', 'Val Acc', 'Test Acc', 'Val F1', 'Test F1'],
        cellLoc='center',
        loc='center',
        colWidths=[0.25, 0.18, 0.18, 0.17, 0.17]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    # Header style
    for i in range(5):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(color='white', fontweight='bold')

    # Highlight best row
    best_idx = np.argmax([results['test'][m]['accuracy'] for m in models]) + 1
    for i in range(5):
        table[(best_idx, i)].set_facecolor('#d5f5e3')

    ax4.set_title('Detailed Results', fontsize=13, fontweight='bold', pad=20)

    # Timestamp
    fig.text(0.99, 0.01, f"Generated: {results.get('timestamp', 'N/A')}",
             ha='right', va='bottom', fontsize=9, color='gray')

    plt.suptitle('Fault Prediction Model Evaluation Report', fontsize=16, fontweight='bold', y=0.98)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def plot_kg_enhancement_comparison(results, save_path='results/figures/kg_enhancement_comparison.png'):
    """Plot KG enhancement effect - CNN vs CNN+KG and MLP vs MLP+KG"""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    # Define comparison pairs
    mlp_models = ['MLP', 'MLP_KG', 'MLP_KG_V2']
    cnn_models = ['CNN', 'CNN_KG', 'CNN_KG_V2', 'CNN_KG_V3']

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # MLP vs KG-MLP comparison
    ax1 = axes[0]
    mlp_names = ['MLP\n(Baseline)', 'MLP-KG\n(Global)', 'MLP-KG-V2\n(Fault-Level)']
    mlp_val = [results['validation'][m]['accuracy'] for m in mlp_models]
    mlp_test = [results['test'][m]['accuracy'] for m in mlp_models]

    x = np.arange(len(mlp_models))
    width = 0.35

    bars1 = ax1.bar(x - width/2, mlp_val, width, label='Validation', color='#3498db', alpha=0.8, edgecolor='white')
    bars2 = ax1.bar(x + width/2, mlp_test, width, label='Test', color='#e74c3c', alpha=0.8, edgecolor='white')

    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('MLP + Knowledge Graph Enhancement', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(mlp_names, fontsize=10)
    ax1.legend(loc='lower right', fontsize=10)
    ax1.set_ylim(0.5, 0.9)
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels and improvement arrows
    for i, (bar, acc) in enumerate(zip(bars2, mlp_test)):
        height = bar.get_height()
        ax1.annotate(f'{acc:.2%}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

        # Show improvement vs baseline
        if i > 0:
            improvement = (acc - mlp_test[0]) * 100
            ax1.annotate(f'+{improvement:.1f}%',
                        xy=(x[i] + width/2, mlp_test[i]),
                        xytext=(0, 20), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, color='green', fontweight='bold')

    # CNN vs CNN+KG comparison
    ax2 = axes[1]
    cnn_names = ['CNN\n(Baseline)', 'CNN+KG\n(拼接)', 'CNN+KG V2\n(门控)', 'CNN+KG V3\n(残差)']
    cnn_val = [results['validation'][m]['accuracy'] for m in cnn_models]
    cnn_test = [results['test'][m]['accuracy'] for m in cnn_models]

    x = np.arange(len(cnn_models))

    bars3 = ax2.bar(x - width/2, cnn_val, width, label='Validation', color='#3498db', alpha=0.8, edgecolor='white')
    bars4 = ax2.bar(x + width/2, cnn_test, width, label='Test', color='#e74c3c', alpha=0.8, edgecolor='white')

    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('CNN + Knowledge Graph Enhancement', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(cnn_names, fontsize=10)
    ax2.legend(loc='lower right', fontsize=10)
    ax2.set_ylim(0.5, 0.9)
    ax2.grid(axis='y', alpha=0.3)

    # Add value labels and improvement arrows
    for i, (bar, acc) in enumerate(zip(bars4, cnn_test)):
        height = bar.get_height()
        ax2.annotate(f'{acc:.2%}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

        if i > 0:
            improvement = (acc - cnn_test[0]) * 100
            ax2.annotate(f'+{improvement:.1f}%',
                        xy=(x[i] + width/2, cnn_test[i]),
                        xytext=(0, 20), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, color='green', fontweight='bold')

    plt.suptitle('Knowledge Graph Enhancement Effect', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def main():
    results_path = 'results/evaluation_results.json'

    if not Path(results_path).exists():
        print(f"Error: Results file not found: {results_path}")
        print("Please run: python evaluate.py")
        return

    results = load_evaluation_results(results_path)
    print(f"Loaded: {results_path}")
    print(f"Timestamp: {results.get('timestamp', 'N/A')}")
    print(f"Models: {list(results['validation'].keys())}")

    # Generate charts
    plot_model_comparison(results)
    plot_heatmap_comparison(results)
    plot_improvement_bar(results)
    generate_summary_report(results)
    plot_kg_enhancement_comparison(results)

    print("\nAll visualizations generated!")


if __name__ == '__main__':
    main()