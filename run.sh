#!/bin/bash
# 基于知识图谱的故障预测系统 - 快速启动脚本

set -e

echo "=========================================="
echo "基于知识图谱的故障预测系统"
echo "西安交通大学 - 毕业设计"
echo "=========================================="

# 检查Python环境
if ! command -v python &> /dev/null; then
    echo "错误: 未找到Python"
    exit 1
fi

# 检查Docker
if ! command -v docker &> /dev/null; then
    echo "警告: 未找到Docker，部分功能将不可用"
fi

# 检查conda环境
if [ -n "$CONDA_DEFAULT_ENV" ]; then
    echo "当前环境: $CONDA_DEFAULT_ENV"
else
    echo "注意: 建议激活fault-prediction环境"
    echo "命令: conda activate fault-prediction"
fi

# 参数解析
ACTION=${1:-"all"}

case $ACTION in
    1)
        echo ""
        echo "[步骤1] 数据预处理..."
        echo "----------------------------------------"
        python src/data/preprocess.py \
            --data_path data/raw/XJTU_Gearbox \
            --output_path data/processed
        ;;

    2)
        echo ""
        echo "[步骤2] 启动Neo4j并导入知识图谱..."
        echo "----------------------------------------"
        read -p "请输入Neo4j密码(默认password): " NEO4J_PASSWORD
        NEO4J_PASSWORD=${NEO4J_PASSWORD:-password}

        python src/kg/neo4j_manager.py --action full \
            --kg_file data/processed/knowledge_graph.json \
            --password "$NEO4J_PASSWORD"
        ;;

    3)
        echo ""
        echo "[步骤3] 生成知识图谱嵌入..."
        echo "----------------------------------------"
        python src/kg/kg_embedder.py \
            --embedding_path data/processed/kg_embeddings.json
        ;;

    4)
        echo ""
        echo "[步骤4] 模型训练..."
        echo "----------------------------------------"
        python src/train/train.py
        ;;

    5)
        echo ""
        echo "[步骤5] 生成可视化结果..."
        echo "----------------------------------------"
        python src/visualization/visualize.py --all
        ;;

    all)
        echo ""
        echo "[完整流程] 数据处理 -> Neo4j导入 -> 训练 -> 可视化"
        echo "----------------------------------------"

        echo ""
        echo "[步骤1/5] 数据预处理..."
        python src/data/preprocess.py \
            --data_path data/raw/XJTU_Gearbox \
            --output_path data/processed

        echo ""
        echo "[步骤2/5] 启动Neo4j并导入知识图谱..."
        read -p "请输入Neo4j密码(默认password): " NEO4J_PASSWORD
        NEO4J_PASSWORD=${NEO4J_PASSWORD:-password}

        python src/kg/neo4j_manager.py --action full \
            --kg_file data/processed/knowledge_graph.json \
            --password "$NEO4J_PASSWORD"

        echo ""
        echo "[步骤3/5] 生成知识图谱嵌入..."
        python src/kg/kg_embedder.py \
            --embedding_path data/processed/kg_embeddings.json

        echo ""
        echo "[步骤4/5] 模型训练..."
        python src/train/train.py

        echo ""
        echo "[步骤5/5] 生成可视化结果..."
        python src/visualization/visualize.py --all

        echo ""
        echo "=========================================="
        echo "全部完成!"
        echo "结果保存在: results/"
        echo "=========================================="
        ;;

    *)
        echo "用法: ./run.sh [选项]"
        echo ""
        echo "选项:"
        echo "  1       - 数据预处理"
        echo "  2       - 启动Neo4j并导入知识图谱"
        echo "  3       - 生成知识图谱嵌入"
        echo "  4       - 模型训练"
        echo "  5       - 生成可视化结果"
        echo "  all     - 运行完整流程 (默认)"
        echo ""
        echo "示例:"
        echo "  ./run.sh          # 运行完整流程"
        echo "  ./run.sh 1        # 仅预处理数据"
        ;;
esac
