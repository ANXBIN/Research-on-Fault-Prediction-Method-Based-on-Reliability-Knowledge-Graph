#!/usr/bin/env python3
"""
基于随机森林(RF)的故障预测模型
"""

import numpy as np
import yaml
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import joblib
import json
from pathlib import Path


class RFModel:
    """随机森林故障预测模型"""

    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.model_config = self.config['models']['rf']
        self.n_estimators = self.model_config['n_estimators']
        self.max_depth = self.model_config['max_depth']
        self.min_samples_split = self.model_config['min_samples_split']
        self.learning_rate = self.model_config['learning_rate']
        self.epochs = self.model_config['epochs']

        self.model = None
        self.fault_to_idx = {}
        self.idx_to_fault = {}
        self.feature_importances_ = None

    def build_model(self, n_features, n_classes):
        """构建随机森林模型"""
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )

        return self.model

    def fit(self, X, y):
        """训练模型"""
        if self.model is None:
            self.build_model(X.shape[1], len(np.unique(y)))

        self.model.fit(X, y)
        self.feature_importances_ = self.model.feature_importances_

        return self

    def predict(self, X):
        """预测"""
        return self.model.predict(X)

    def predict_proba(self, X):
        """预测概率"""
        return self.model.predict_proba(X)

    def evaluate(self, X, y):
        """评估模型"""
        y_pred = self.predict(X)

        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y, y_pred, average='weighted', zero_division=0)
        }

        return metrics, y_pred

    def cross_validate(self, X, y, cv=5):
        """交叉验证"""
        scores = cross_val_score(self.model, X, y, cv=cv, scoring='accuracy')
        return {
            'cv_scores': scores.tolist(),
            'cv_mean': scores.mean(),
            'cv_std': scores.std()
        }

    def get_feature_importance(self, feature_names=None):
        """获取特征重要性"""
        if self.feature_importances_ is None:
            return None

        importance = self.feature_importances_
        if feature_names is not None:
            importance_dict = dict(zip(feature_names, importance))
        else:
            importance_dict = {f'feature_{i}': v for i, v in enumerate(importance)}

        # 排序
        importance_dict = dict(
            sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        )

        return importance_dict

    def save_model(self, path):
        """保存模型"""
        model_data = {
            'model': self.model,
            'fault_to_idx': self.fault_to_idx,
            'idx_to_fault': self.idx_to_fault,
            'feature_importances': self.feature_importances_,
            'config': self.model_config
        }

        joblib.dump(model_data, path)
        print(f"[INFO] 模型已保存至: {path}")

    def load_model(self, path):
        """加载模型"""
        model_data = joblib.load(path)
        self.model = model_data['model']
        self.fault_to_idx = model_data['fault_to_idx']
        self.idx_to_fault = model_data['idx_to_fault']
        self.feature_importances_ = model_data['feature_importances']
        print(f"[INFO] 模型已从: {path} 加载")


class GBModel:
    """梯度提升故障预测模型（对比用）"""

    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.model_config = self.config['models']['rf']
        self.n_estimators = self.model_config['n_estimators']
        self.max_depth = self.model_config['max_depth']
        self.learning_rate = self.model_config['learning_rate']

        self.model = None
        self.fault_to_idx = {}
        self.idx_to_fault = {}

    def build_model(self, n_features, n_classes):
        """构建梯度提升模型"""
        self.model = GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=42
        )
        return self.model

    def fit(self, X, y):
        """训练模型"""
        if self.model is None:
            self.build_model(X.shape[1], len(np.unique(y)))
        self.model.fit(X, y)
        return self

    def predict(self, X):
        """预测"""
        return self.model.predict(X)

    def predict_proba(self, X):
        """预测概率"""
        return self.model.predict_proba(X)

    def evaluate(self, X, y):
        """评估模型"""
        y_pred = self.predict(X)
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y, y_pred, average='weighted', zero_division=0)
        }
        return metrics, y_pred

    def save_model(self, path):
        """保存模型"""
        model_data = {
            'model': self.model,
            'fault_to_idx': self.fault_to_idx,
            'idx_to_fault': self.idx_to_fault
        }
        joblib.dump(model_data, path)
        print(f"[INFO] 模型已保存至: {path}")

    def load_model(self, path):
        """加载模型"""
        model_data = joblib.load(path)
        self.model = model_data['model']
        self.fault_to_idx = model_data['fault_to_idx']
        self.idx_to_fault = model_data['idx_to_fault']
        print(f"[INFO] 模型已从: {path} 加载")


def train_rf_model(X_train, y_train, X_val, y_val, config_path='config.yaml'):
    """训练随机森林模型"""
    model = RFModel(config_path=config_path)
    model.build_model(X_train.shape[1], len(np.unique(y_train)))

    # 训练
    model.fit(X_train, y_train)

    # 评估
    train_metrics, _ = model.evaluate(X_train, y_train)
    val_metrics, _ = model.evaluate(X_val, y_val)

    print("\n" + "=" * 50)
    print("随机森林训练结果")
    print("=" * 50)
    print(f"训练集 - 准确率: {train_metrics['accuracy']:.4f}, F1: {train_metrics['f1']:.4f}")
    print(f"验证集 - 准确率: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}")

    return model
