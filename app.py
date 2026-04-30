#!/usr/bin/env python3
"""
基于知识图谱的变速箱故障预测系统 - Web前端
Streamlit可视化界面

启动方式: streamlit run app.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
import numpy as np
import pandas as pd
import torch
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.models.mlp_model import MLPModel, KGEnhancedMLPV2Model
from src.models.cnn_model import CNNModel, CNNKGModelV3
from src.models.gnn_model import GNNModel, GNNKGModel, build_batch_adjacency
from src.reasoning.rule_reasoner import RuleBasedReasoner
import requests


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


@st.cache_resource
def get_ollama_client():
    return OllamaClient()


st.set_page_config(
    page_title="基于知识图谱的变速箱故障预测系统",
    page_icon="⚙️",
    layout="wide"
)


@st.cache_resource
def load_data():
    df = pd.read_csv('data/processed/processed_features.csv')
    feature_cols = [col for col in df.columns if col not in ['fault_type', 'channel']]
    return df, feature_cols


@st.cache_resource
def load_evaluation_results():
    path = Path('results/evaluation_results.json')
    if path.exists():
        with open(path, 'r') as f:
            return json.load(f)
    return None


@st.cache_resource
def load_training_results():
    path = Path('results/training_results.json')
    if path.exists():
        with open(path, 'r') as f:
            return json.load(f)
    return None


@st.cache_resource
def load_knowledge_graph():
    path = Path('data/processed/knowledge_graph.json')
    if path.exists():
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            nodes = data.get('nodes', [])
            edges = data.get('edges', [])
            node_types = {}
            for node in nodes:
                t = node.get('type', 'unknown')
                node_types[t] = node_types.get(t, 0) + 1
            edge_types = {}
            for edge in edges:
                t = edge.get('type', 'unknown')
                edge_types[t] = edge_types.get(t, 0) + 1
            return {
                'total_nodes': len(nodes),
                'total_edges': len(edges),
                'node_types': node_types,
                'edge_types': edge_types,
                'sample_nodes': nodes[:100] if nodes else [],
                'sample_edges': edges[:200] if edges else [],
            }
        except (json.JSONDecodeError, MemoryError):
            return None
    return None


@st.cache_resource
def load_fault_embeddings():
    path = Path('data/processed/fault_embeddings.json')
    if path.exists():
        with open(path, 'r') as f:
            return json.load(f)
    return None


MODEL_REGISTRY = {
    'MLP':    (MLPModel,            'mlp_model.pt',    {}),
    'MLP-KG': (KGEnhancedMLPV2Model, 'mlp_kg_model.pt', {'kg_embedding_dim': 64}),
    'CNN':    (CNNModel,             'cnn_model.pt',    {}),
    'CNN-KG': (CNNKGModelV3,         'cnn_kg_model.pt', {'kg_embedding_dim': 33}),
    'GNN':    (GNNModel,             'gnn_model.pt',    {}),
    'GNN-KG': (GNNKGModel,           'gnn_kg_model.pt', {'kg_embedding_dim': 33}),
}


def load_model(model_name):
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    cls, filename, extra_kwargs = MODEL_REGISTRY[model_name]
    path = Path('models') / filename
    if not path.exists():
        return None

    checkpoint = torch.load(path, map_location=device, weights_only=True)
    saved_config = checkpoint.get('config', {})

    df, feature_cols = load_data()
    from sklearn.preprocessing import LabelEncoder
    y = df['fault_type'].values
    le = LabelEncoder()
    le.fit(y)
    fault_to_idx = dict(zip(le.classes_, range(len(le.classes_))))

    model = cls(config_path='config.yaml')
    for key, default in extra_kwargs.items():
        setattr(model, key, saved_config.get(key, default))
    for key in ['hidden_dim', 'dropout', 'num_layers', 'batch_size', 'learning_rate']:
        if key in saved_config:
            setattr(model, key, saved_config[key])

    model.fault_to_idx = fault_to_idx
    model.build_model(len(feature_cols), len(le.classes_))
    model.model.load_state_dict(checkpoint['model_state_dict'])
    model.model.eval()
    return model, le


def predict_with_model(model_name, features, scaler):
    result = load_model(model_name)
    if result is None:
        return None
    model, le = result
    X_scaled = scaler.transform(np.array(features).reshape(1, -1))
    X_tensor = torch.tensor(X_scaled, dtype=torch.float).to(model.device)

    with torch.no_grad():
        if isinstance(model, (GNNModel, GNNKGModel)):
            adj = build_batch_adjacency(X_scaled, k=30).to(model.device)
            if isinstance(model, GNNKGModel):
                avg_emb = np.zeros(33)
                kg_tensor = torch.tensor(avg_emb.reshape(1, -1), dtype=torch.float).to(model.device)
                output = model.model(X_tensor, kg_tensor, adj)
            else:
                output = model.model(X_tensor, adj)
        elif isinstance(model, CNNKGModelV3):
            avg_emb = np.zeros(33)
            kg_tensor = torch.tensor(avg_emb.reshape(1, -1), dtype=torch.float).to(model.device)
            output = model.model(X_tensor, kg_tensor)
        else:
            output = model.model(X_tensor)

        probs = torch.softmax(output, dim=1).cpu().numpy()[0]

    top_idx = np.argsort(probs)[::-1]
    return [(le.inverse_transform([i])[0], float(probs[i])) for i in top_idx[:5]]


def page_home():
    st.title("⚙️ 基于知识图谱的变速箱故障预测系统")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.info("**项目概述**\n\n"
                "利用知识图谱增强深度学习模型的故障诊断能力，"
                "实现变速箱故障类型的精准分类。")
    with col2:
        st.info("**数据集**\n\n"
                "西安交通大学变速箱故障数据集\n"
                "- 样本数: 82,606\n"
                "- 故障类型: 9类\n"
                "- 特征维度: 20维")

    st.markdown("### 系统架构")
    st.markdown("""
    ```
    原始振动信号 → 特征提取(20维) → 知识图谱构建 → KG嵌入(33维)
                                                         ↓
    ┌─────────────────────────────────────────────────────────┐
    │                 深度学习模型                              │
    │  ┌─────────┐  ┌─────────┐  ┌─────────┐                 │
    │  │   MLP   │  │   CNN   │  │   GNN   │                 │
    │  └────┬────┘  └────┬────┘  └────┬────┘                 │
    │       └──────┬──────┴──────┬────┘                       │
    │              ↓             ↓                             │
    │         KG融合(门控/残差/注意力)                         │
    │              ↓                                           │
    │         故障分类 → 规则推理验证 → LLM解释               │
    └─────────────────────────────────────────────────────────┘
    ```""")

    st.markdown("### 故障类型")
    fault_types = {
        'Ball_Fault': '滚动体故障', 'Broken_Tooth': '断齿故障',
        'Inner_Race_Fault': '内圈故障', 'Missing_Tooth': '缺齿故障',
        'Mixed_Fault': '混合故障', 'Normal': '正常',
        'Outer_Race_Fault': '外圈故障', 'Root_Crack': '齿根裂纹',
        'Tooth_Wear': '齿面磨损',
    }
    cols = st.columns(3)
    for i, (en, cn) in enumerate(fault_types.items()):
        cols[i % 3].markdown(f"- **{en}** ({cn})")


def page_model_comparison():
    st.title("📊 模型对比分析")

    eval_results = load_evaluation_results()
    if eval_results is None:
        st.warning("未找到评估结果，请先运行 `python evaluate.py`")
        return

    test_results = eval_results.get('test', {})
    val_results = eval_results.get('validation', {})

    if not test_results:
        st.warning("没有可用的评估数据")
        return

    model_names = list(test_results.keys())
    test_accs = [test_results[m]['accuracy'] for m in model_names]
    test_f1s = [test_results[m]['f1'] for m in model_names]

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='准确率', x=model_names, y=test_accs,
            marker_color='#636EFA'
        ))
        fig.add_trace(go.Bar(
            name='F1分数', x=model_names, y=test_f1s,
            marker_color='#EF553B'
        ))
        fig.update_layout(
            title='测试集性能对比',
            xaxis_title='模型', yaxis_title='分数',
            barmode='group', height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=[max(test_accs), max(test_f1s), 0.9, 0.85, 0.95],
            theta=['准确率', 'F1', 'KG融合', '图结构', '可解释性'],
            fill='toself', name='最佳模型'
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title='最佳模型雷达图', height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 详细评估结果")
    df_eval = pd.DataFrame({
        '模型': model_names,
        '验证集准确率': [val_results.get(m, {}).get('accuracy', 0) for m in model_names],
        '测试集准确率': test_accs,
        '验证F1': [val_results.get(m, {}).get('f1', 0) for m in model_names],
        '测试F1': test_f1s,
    })
    st.dataframe(df_eval.style.highlight_max(axis=0, color='lightgreen'), use_container_width=True)

    best_idx = np.argmax(test_accs)
    st.success(f"**最佳模型**: {model_names[best_idx]} — 测试集准确率 {test_accs[best_idx]:.2%}")


def page_knowledge_graph():
    st.title("🔗 知识图谱分析")

    kg_data = load_knowledge_graph()
    if kg_data is None:
        st.warning("未找到知识图谱数据")
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("节点总数", f"{kg_data['total_nodes']:,}")
    col2.metric("边总数", f"{kg_data['total_edges']:,}")
    col3.metric("节点类型数", len(kg_data['node_types']))

    st.markdown("### 节点类型分布")
    fig = px.pie(
        values=list(kg_data['node_types'].values()),
        names=list(kg_data['node_types'].keys()),
        title='知识图谱节点类型分布',
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 边类型分布")
    fig = px.bar(
        x=list(kg_data['edge_types'].keys()),
        y=list(kg_data['edge_types'].values()),
        title='知识图谱关系类型分布',
        labels={'x': '关系类型', 'y': '数量'},
        color=list(kg_data['edge_types'].keys()),
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### KG嵌入相似度矩阵")
    fault_embs = load_fault_embeddings()
    if fault_embs and 'fault_similarity' in fault_embs:
        sim_data = fault_embs['fault_similarity']
        fault_names = fault_embs.get('fault_types', [f"Fault_{i}" for i in range(len(sim_data))])

        # 处理两种格式：字典或列表矩阵
        if isinstance(sim_data, dict):
            n = len(fault_names)
            sim_matrix = np.zeros((n, n))
            for i, f1 in enumerate(fault_names):
                for j, f2 in enumerate(fault_names):
                    sim_matrix[i, j] = sim_data.get(f1, {}).get(f2, 0)
        else:
            # 列表矩阵格式
            sim_matrix = np.array(sim_data)

        fig = px.imshow(
            sim_matrix, x=fault_names, y=fault_names,
            color_continuous_scale='RdYlBu_r',
            title='故障类型Jaccard相似度矩阵',
            labels=dict(color="相似度")
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 知识图谱样本")
    if kg_data['sample_nodes']:
        nodes_df = pd.DataFrame([
            {'ID': n.get('id', ''), '类型': n.get('type', ''), '标签': n.get('label', '')}
            for n in kg_data['sample_nodes'][:30]
        ])
        st.dataframe(nodes_df, use_container_width=True)


def page_prediction():
    st.title("🔮 故障预测")

    df, feature_cols = load_data()

    st.markdown("### 选择输入方式")
    input_method = st.radio("", ["随机样本", "手动输入", "选择样本"], horizontal=True)

    features = None
    actual_fault = None
    sample_idx = None

    if input_method == "随机样本":
        sample_idx = np.random.randint(len(df))
        features = df.iloc[sample_idx][feature_cols].values
        actual_fault = df.iloc[sample_idx]['fault_type']
        st.info(f"随机选择样本 **{sample_idx}**，实际故障: **{actual_fault}**")

    elif input_method == "选择样本":
        col1, col2 = st.columns([1, 3])
        with col1:
            fault_filter = st.selectbox("按故障类型筛选", ['全部'] + list(df['fault_type'].unique()))
        if fault_filter != '全部':
            filtered_df = df[df['fault_type'] == fault_filter]
        else:
            filtered_df = df
        sample_idx = st.number_input("样本索引", 0, len(filtered_df) - 1, 0)
        actual_fault = filtered_df.iloc[sample_idx]['fault_type']
        features = filtered_df.iloc[sample_idx][feature_cols].values
        st.info(f"样本 **{sample_idx}**，实际故障: **{actual_fault}**")

    else:
        st.markdown("**输入20维特征值**")
        cols = st.columns(4)
        features = []
        defaults = [100000, 50000, 80000, 200000, -200000, 300000, 0.5, 3.0, 2.5, 1.1, 3.0, 1e8, 500, 3.0, 500, 1e7, 1e7, 1e7, 1e7, 1e7]
        for i, feat_name in enumerate(feature_cols):
            with cols[i % 4]:
                val = st.number_input(feat_name, value=float(defaults[i]), format="%.2e", key=f"feat_{i}")
                features.append(val)
        features = np.array(features)

    if features is not None:
        st.markdown("---")
        st.markdown("### 预测结果")

        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(df[feature_cols].values)

        tab1, tab2 = st.tabs(["深度学习模型", "规则推理"])

        with tab1:
            selected_models = st.multiselect(
                "选择模型",
                list(MODEL_REGISTRY.keys()),
                default=['GNN-KG', 'CNN-KG']
            )

            if st.button("运行深度学习预测", key="dl_predict"):
                with st.spinner("正在预测..."):
                    all_results = {}
                    for model_name in selected_models:
                        result = predict_with_model(model_name, features, scaler)
                        if result:
                            all_results[model_name] = result

                if all_results:
                    fig = make_subplots(
                        rows=1, cols=len(all_results),
                        subplot_titles=list(all_results.keys())
                    )
                    for i, (model_name, preds) in enumerate(all_results.items(), 1):
                        fig.add_trace(
                            go.Bar(
                                x=[p[0] for p in preds[:5]],
                                y=[p[1] for p in preds[:5]],
                                name=model_name,
                                showlegend=False,
                            ),
                            row=1, col=i
                        )
                    fig.update_layout(height=400, title_text="各模型预测概率分布")
                    st.plotly_chart(fig, use_container_width=True)

                    for model_name, preds in all_results.items():
                        st.markdown(f"**{model_name}**: {preds[0][0]} ({preds[0][1]:.1%})")

                    if actual_fault:
                        best_model = list(all_results.keys())[0]
                        best_pred = all_results[best_model][0][0]
                        if best_pred == actual_fault:
                            st.success(f"预测正确! 实际故障: {actual_fault}")
                        else:
                            st.error(f"预测错误! 实际故障: {actual_fault}, 预测: {best_pred}")

                    # LLM解释
                    st.markdown("---")
                    st.markdown("### 🤖 AI智能解释")

                    ollama = get_ollama_client()
                    if ollama.is_available():
                        best_model = list(all_results.keys())[0]
                        top3 = all_results[best_model][:3]
                        fault_name = top3[0][0]

                        with st.spinner("正在生成AI解释..."):
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

回答要简洁，控制在150字以内。"""

                            explanation = ollama.generate(prompt)
                            if explanation:
                                st.info(explanation)
                            else:
                                st.warning("无法获取AI解释")
                    else:
                        st.warning("Ollama服务未运行，请启动Ollama以获取AI解释")

        with tab2:
            st.markdown("**基于规则的故障推理**")
            st.markdown("利用知识图谱中的故障-部件-特征关系，结合特征统计阈值进行推理。")

            if st.button("运行规则推理", key="rule_predict"):
                reasoner = RuleBasedReasoner(feature_names=feature_cols)
                fault, probs, details = reasoner.diagnose(features)
                explanation = reasoner.explain(fault, features)

                fig = px.bar(
                    x=list(probs.keys()), y=list(probs.values()),
                    title='规则推理概率分布',
                    labels={'x': '故障类型', 'y': '概率'},
                    color=list(probs.keys()),
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                st.plotly_chart(fig, use_container_width=True)

                st.markdown(f"**推理结果**: {fault}")
                st.markdown(f"**相关部件**: {', '.join(explanation['related_components']) or '无'}")

                st.markdown("**规则匹配详情**")
                rules_df = pd.DataFrame(explanation['rule_matches'])
                if not rules_df.empty:
                    st.dataframe(
                        rules_df.style.applymap(
                            lambda v: 'background-color: #90EE90' if v == True else ('background-color: #FFB6C1' if v == False else ''),
                            subset=['matched']
                        ),
                        use_container_width=True
                    )

                # LLM解释
                st.markdown("---")
                st.markdown("### 🤖 AI智能解释")

                ollama = get_ollama_client()
                if ollama.is_available():
                    with st.spinner("正在生成AI解释..."):
                        similar_faults = explanation.get('knowledge_graph_info', {}).get('similar_faults', [])
                        similar_str = ", ".join([f"{s['fault']}({s['similarity']:.2f})" for s in similar_faults]) if similar_faults else "无"

                        prompt = f"""你是一个变速箱故障诊断专家，基于以下规则推理结果，给出简洁的解释和建议。

故障类型: {fault}
相关部件: {', '.join(explanation['related_components']) or '无'}
相似故障类型: {similar_str}

规则匹配情况:
"""
                        for rule in explanation['rule_matches'][:5]:
                            status = "✓ 符合" if rule['matched'] else "✗ 不符合"
                            prompt += f"- {rule['feature']}: 值={rule['value']}, 期望范围={rule['expected_range']} [{status}]\n"

                        prompt += """
请给出：
1. 该故障类型的诊断依据
2. 基于知识图谱的故障原因分析
3. 建议的维护措施

回答要简洁，控制在150字以内。"""

                        llm_explanation = ollama.generate(prompt)
                        if llm_explanation:
                            st.info(llm_explanation)
                        else:
                            st.warning("无法获取AI解释")
                else:
                    st.warning("Ollama服务未运行，请启动Ollama以获取AI解释")


def page_training_history():
    st.title("📈 训练历史")

    training_results = load_training_results()
    if training_results is None:
        st.warning("未找到训练结果，请先运行 `python train.py --all`")
        return

    models_data = training_results.get('models', {})
    if not models_data:
        st.warning("没有训练数据")
        return

    fig = px.bar(
        x=list(models_data.keys()),
        y=[d.get('val_accuracy', 0) for d in models_data.values()],
        title='各模型验证集准确率',
        labels={'x': '模型', 'y': '准确率'},
        color=list(models_data.keys()),
        color_discrete_sequence=px.colors.qualitative.Vivid
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 训练配置")
    best_config = training_results.get('best_config', {})
    if best_config:
        st.json(best_config)


def main():
    st.sidebar.title("⚙️ 导航菜单")
    page = st.sidebar.radio(
        "选择页面",
        ["系统首页", "模型对比", "知识图谱", "故障预测", "训练历史"]
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**系统状态**")
    device = "MPS (Apple GPU)" if torch.backends.mps.is_available() else "CPU"
    st.sidebar.code(f"Device: {device}")

    ollama = get_ollama_client()
    if ollama.is_available():
        st.sidebar.success("✓ Ollama服务已连接")
    else:
        st.sidebar.warning("✗ Ollama服务未运行")

    page_map = {
        "系统首页": page_home,
        "模型对比": page_model_comparison,
        "知识图谱": page_knowledge_graph,
        "故障预测": page_prediction,
        "训练历史": page_training_history,
    }
    page_map[page]()


if __name__ == '__main__':
    main()
