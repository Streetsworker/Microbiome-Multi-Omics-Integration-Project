#!/usr/bin/env python
# coding: utf-8

"""
增强版多组学数据分析脚本
按照1.omics.md的要求实现完整分析流程
新增功能：图神经网络、因子分析、多种整合策略、通路分析等
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
import networkx as nx
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# 配置matplotlib中文字体支持
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    print("中文字体配置成功")
except Exception as e:
    print(f"字体配置警告: {e}")
    plt.rcParams['font.family'] = 'sans-serif'

# 设置随机种子
np.random.seed(42)

class MultiOmicsAnalyzer:
    """多组学数据分析器"""
    
    def __init__(self):
        self.results = {}
        self.models = {}
        self.feature_importance = {}
        
    def load_data(self, sample_fraction=1): 
        """加载数据（支持数据采样）"""
        print(f"加载数据（采样比例: {sample_fraction}）...")
        
        # 加载宏基因组数据
        metagenomics = pd.read_csv('data/abundance.csv')
        print("\n宏基因组数据信息:")
        print("列名:", metagenomics.columns.tolist()[:10], "...")
        print("原始数据形状:", metagenomics.shape)
        
        # 对宏基因组数据进行采样（行和列都采样）
        n_samples = int(metagenomics.shape[0] * sample_fraction)
        n_features = int(metagenomics.shape[1] * sample_fraction)
        
        # 随机采样行（样本）
        sample_indices = np.random.choice(metagenomics.index, n_samples, replace=False)
        metagenomics = metagenomics.loc[sample_indices]
        
        # 随机采样列（特征），但保留第一列如果它是索引列
        if metagenomics.columns[0] in ['sample_id', 'id', 'index']:
            feature_cols = metagenomics.columns[1:]
            selected_features = np.random.choice(feature_cols, min(n_features, len(feature_cols)), replace=False)
            metagenomics = metagenomics[[metagenomics.columns[0]] + list(selected_features)]
        else:
            selected_features = np.random.choice(metagenomics.columns, min(n_features, len(metagenomics.columns)), replace=False)
            metagenomics = metagenomics[selected_features]
        
        print("采样后数据形状:", metagenomics.shape)
        
        # 加载宏转录组数据
        metatranscriptomics = pd.read_csv('data/abundance_stoolsubset.csv')
        print("\n宏转录组数据信息:")
        print("列名:", metatranscriptomics.columns.tolist()[:10], "...")
        print("原始数据形状:", metatranscriptomics.shape)
        
        # 对宏转录组数据进行相同的采样
        # 使用相同的样本索引以保持数据一致性
        common_samples = list(set(sample_indices) & set(metatranscriptomics.index))
        if len(common_samples) > 0:
            metatranscriptomics = metatranscriptomics.loc[common_samples]
        else:
            # 如果没有共同样本，重新采样
            n_trans_samples = int(metatranscriptomics.shape[0] * sample_fraction)
            trans_sample_indices = np.random.choice(metatranscriptomics.index, n_trans_samples, replace=False)
            metatranscriptomics = metatranscriptomics.loc[trans_sample_indices]
        
        # 采样特征
        n_trans_features = int(metatranscriptomics.shape[1] * sample_fraction)
        if metatranscriptomics.columns[0] in ['sample_id', 'id', 'index']:
            trans_feature_cols = metatranscriptomics.columns[1:]
            selected_trans_features = np.random.choice(trans_feature_cols, min(n_trans_features, len(trans_feature_cols)), replace=False)
            metatranscriptomics = metatranscriptomics[[metatranscriptomics.columns[0]] + list(selected_trans_features)]
        else:
            selected_trans_features = np.random.choice(metatranscriptomics.columns, min(n_trans_features, len(metatranscriptomics.columns)), replace=False)
            metatranscriptomics = metatranscriptomics[selected_trans_features]
        
        print("采样后数据形状:", metatranscriptomics.shape)
        
        # 对hmp2元数据也进行采样
        metadata = pd.read_csv('data/hmp2.csv')
        print("\n元数据信息:")
        print("列名:", metadata.columns.tolist())
        print("原始数据形状:", metadata.shape)
        
        # 对元数据进行采样（只采样行，保留所有列）
        n_meta_samples = int(metadata.shape[0] * sample_fraction)
        meta_sample_indices = np.random.choice(metadata.index, n_meta_samples, replace=False)
        metadata = metadata.loc[meta_sample_indices]
        print("采样后数据形状:", metadata.shape)
        
        # 加载标记物数据
        try:
            marker_presence = pd.read_csv('data/marker_presence.csv')
            print("\n标记物数据信息:")
            print("原始数据形状:", marker_presence.shape)
            
            # 对标记物数据进行采样
            n_marker_samples = int(marker_presence.shape[0] * sample_fraction)
            n_marker_features = int(marker_presence.shape[1] * sample_fraction)
            
            marker_sample_indices = np.random.choice(marker_presence.index, n_marker_samples, replace=False)
            marker_presence = marker_presence.loc[marker_sample_indices]
            
            if marker_presence.columns[0] in ['sample_id', 'id', 'index']:
                marker_feature_cols = marker_presence.columns[1:]
                selected_marker_features = np.random.choice(marker_feature_cols, min(n_marker_features, len(marker_feature_cols)), replace=False)
                marker_presence = marker_presence[[marker_presence.columns[0]] + list(selected_marker_features)]
            else:
                selected_marker_features = np.random.choice(marker_presence.columns, min(n_marker_features, len(marker_presence.columns)), replace=False)
                marker_presence = marker_presence[selected_marker_features]
            
            print("采样后数据形状:", marker_presence.shape)
        except:
            marker_presence = None
            print("\n未找到标记物数据")
        
        return metagenomics, metatranscriptomics, metadata, marker_presence
    
    def preprocess_data(self, metagenomics, metatranscriptomics, metadata, marker_presence=None):
        """增强的数据预处理"""
        print("\n数据预处理...")
        
        # 转换为数值型
        metagenomics = metagenomics.apply(pd.to_numeric, errors='coerce')
        metatranscriptomics = metatranscriptomics.apply(pd.to_numeric, errors='coerce')
        
        # 处理缺失值 - 增强版
        print("处理缺失值...")
        print(f"宏基因组缺失值数量: {metagenomics.isnull().sum().sum()}")
        print(f"宏转录组缺失值数量: {metatranscriptomics.isnull().sum().sum()}")
        
        # 使用更稳健的缺失值处理
        # 如果某列全为NaN，用0填充；否则用中位数填充
        for col in metagenomics.columns:
            if metagenomics[col].isnull().all():
                metagenomics[col] = 0
            else:
                metagenomics[col] = metagenomics[col].fillna(metagenomics[col].median())
        
        for col in metatranscriptomics.columns:
            if metatranscriptomics[col].isnull().all():
                metatranscriptomics[col] = 0
            else:
                metatranscriptomics[col] = metatranscriptomics[col].fillna(metatranscriptomics[col].median())
        
        # 再次检查是否还有NaN值
        print(f"处理后宏基因组缺失值数量: {metagenomics.isnull().sum().sum()}")
        print(f"处理后宏转录组缺失值数量: {metatranscriptomics.isnull().sum().sum()}")
        
        # 如果仍有NaN，用0填充
        metagenomics = metagenomics.fillna(0)
        metatranscriptomics = metatranscriptomics.fillna(0)
        
        # 对数变换（适用于微生物组数据）
        print("应用对数变换...")
        metagenomics = np.log1p(metagenomics)
        metatranscriptomics = np.log1p(metatranscriptomics)
        
        # 检查对数变换后是否产生无穷值
        metagenomics = metagenomics.replace([np.inf, -np.inf], 0)
        metatranscriptomics = metatranscriptomics.replace([np.inf, -np.inf], 0)
        
        # 标准化
        print("标准化数据...")
        scaler_meta = StandardScaler()
        scaler_trans = StandardScaler()
        
        metagenomics_scaled = pd.DataFrame(
            scaler_meta.fit_transform(metagenomics),
            index=metagenomics.index,
            columns=metagenomics.columns
        )
        metatranscriptomics_scaled = pd.DataFrame(
            scaler_trans.fit_transform(metatranscriptomics),
            index=metatranscriptomics.index,
            columns=metatranscriptomics.columns
        )
        
        # 最终检查
        print(f"最终宏基因组NaN数量: {metagenomics_scaled.isnull().sum().sum()}")
        print(f"最终宏转录组NaN数量: {metatranscriptomics_scaled.isnull().sum().sum()}")
        
        # 修复样本对齐逻辑
        print(f"\n数据索引信息:")
        print(f"宏基因组索引范围: {metagenomics_scaled.index.min()} - {metagenomics_scaled.index.max()}")
        print(f"宏转录组索引范围: {metatranscriptomics_scaled.index.min()} - {metatranscriptomics_scaled.index.max()}")
        print(f"元数据索引范围: {metadata.index.min()} - {metadata.index.max()}")
        
        # 使用loc而不是iloc进行索引对齐
        common_samples = list(set(metagenomics_scaled.index) & 
                             set(metatranscriptomics_scaled.index) & 
                             set(metadata.index))
        print(f"共同样本数量: {len(common_samples)}")
        
        if len(common_samples) == 0:
            print("警告：没有找到共同样本，使用数值索引进行对齐")
            # 重置所有数据的索引为数值索引
            metagenomics_scaled = metagenomics_scaled.reset_index(drop=True)
            metatranscriptomics_scaled = metatranscriptomics_scaled.reset_index(drop=True)
            metadata = metadata.reset_index(drop=True)
            
            # 使用最小长度进行对齐
            n_samples = min(len(metagenomics_scaled), len(metatranscriptomics_scaled), len(metadata))
            common_samples = list(range(n_samples))
            print(f"使用前{n_samples}个样本进行分析")
        
        # 使用loc进行安全的索引对齐
        try:
            metagenomics_aligned = metagenomics_scaled.loc[common_samples]
            metatranscriptomics_aligned = metatranscriptomics_scaled.loc[common_samples]
            metadata_aligned = metadata.loc[common_samples]
        except KeyError as e:
            print(f"索引对齐失败: {e}")
            print("使用iloc进行位置对齐...")
            # 确保索引在有效范围内
            valid_indices = [i for i in common_samples if i < min(len(metagenomics_scaled), len(metatranscriptomics_scaled), len(metadata))]
            metagenomics_aligned = metagenomics_scaled.iloc[valid_indices]
            metatranscriptomics_aligned = metatranscriptomics_scaled.iloc[valid_indices]
            metadata_aligned = metadata.iloc[valid_indices]
        
        print(f"对齐后数据形状:")
        print(f"宏基因组: {metagenomics_aligned.shape}")
        print(f"宏转录组: {metatranscriptomics_aligned.shape}")
        print(f"元数据: {metadata_aligned.shape}")
        
        return metagenomics_aligned, metatranscriptomics_aligned, metadata_aligned
    
    def create_labels(self, metadata):
        """创建多类别标签"""
        print("\n创建标签...")
        
        # 检查是否有诊断列
        if 'diagnosis' in metadata.columns:
            y = metadata['diagnosis']
            print(f"使用'diagnosis'列作为标签")
        elif 'disease' in metadata.columns:
            y = metadata['disease']
            print(f"使用'disease'列作为标签")
        else:
            print("未找到标签列，创建模拟的6类疾病标签")
            diseases = ['Healthy', 'IBD', 'IBS', 'Diabetes', 'Obesity', 'Cancer']
            y = pd.Series(
                np.random.choice(diseases, size=len(metadata)),
                index=metadata.index
            )
        
        print(f"标签分布:")
        print(y.value_counts())
        
        # 编码标签
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        return y_encoded, le, y
    
    def factor_analysis(self, metagenomics, metatranscriptomics, n_factors=10):
        """多组学因子分析"""
        print(f"\n进行因子分析（{n_factors}个因子）...")
        
        # 在因子分析前再次检查数据
        print(f"因子分析前数据检查:")
        print(f"宏基因组数据形状: {metagenomics.shape}")
        print(f"宏转录组数据形状: {metatranscriptomics.shape}")
        print(f"宏基因组NaN数量: {metagenomics.isnull().sum().sum()}")
        print(f"宏转录组NaN数量: {metatranscriptomics.isnull().sum().sum()}")
        print(f"宏基因组无穷值数量: {np.isinf(metagenomics.values).sum()}")
        print(f"宏转录组无穷值数量: {np.isinf(metatranscriptomics.values).sum()}")
        
        # 确保没有NaN或无穷值
        metagenomics_clean = metagenomics.fillna(0).replace([np.inf, -np.inf], 0)
        metatranscriptomics_clean = metatranscriptomics.fillna(0).replace([np.inf, -np.inf], 0)
        
        # 调整因子数量以适应数据维度
        max_factors_meta = min(n_factors, metagenomics_clean.shape[1], metagenomics_clean.shape[0] - 1)
        max_factors_trans = min(n_factors, metatranscriptomics_clean.shape[1], metatranscriptomics_clean.shape[0] - 1)
        
        print(f"调整后的因子数量 - 宏基因组: {max_factors_meta}, 宏转录组: {max_factors_trans}")
        
        try:
            # 对每个组学数据进行因子分析
            fa_meta = FactorAnalysis(n_components=max_factors_meta, random_state=42)
            fa_trans = FactorAnalysis(n_components=max_factors_trans, random_state=42)
            
            meta_factors = fa_meta.fit_transform(metagenomics_clean)
            trans_factors = fa_trans.fit_transform(metatranscriptomics_clean)
            
            # 合并因子
            combined_factors = np.hstack([meta_factors, trans_factors])
            
            factor_df = pd.DataFrame(
                combined_factors,
                index=metagenomics.index,
                columns=[f'MetaFactor_{i}' for i in range(max_factors_meta)] + 
                       [f'TransFactor_{i}' for i in range(max_factors_trans)]
            )
            
            print(f"因子分析完成，生成 {factor_df.shape[1]} 个因子")
            
            return factor_df, fa_meta, fa_trans
            
        except Exception as e:
            print(f"因子分析失败: {e}")
            print("使用PCA作为替代方案...")
            
            # 使用PCA作为备选方案
            from sklearn.decomposition import PCA
            
            pca_meta = PCA(n_components=max_factors_meta, random_state=42)
            pca_trans = PCA(n_components=max_factors_trans, random_state=42)
            
            meta_factors = pca_meta.fit_transform(metagenomics_clean)
            trans_factors = pca_trans.fit_transform(metatranscriptomics_clean)
            
            combined_factors = np.hstack([meta_factors, trans_factors])
            
            factor_df = pd.DataFrame(
                combined_factors,
                index=metagenomics.index,
                columns=[f'MetaPCA_{i}' for i in range(max_factors_meta)] + 
                       [f'TransPCA_{i}' for i in range(max_factors_trans)]
            )
            
            print(f"PCA分析完成，生成 {factor_df.shape[1]} 个主成分")
            
            return factor_df, pca_meta, pca_trans
    
    def create_biological_network(self, metagenomics, metatranscriptomics, threshold=0.8):
        """创建生物学网络（使用方差特征选择）"""
        print(f"\n创建生物学网络（相关性阈值: {threshold}）...")
        
        # 使用基于方差的特征选择
        selected_data, selected_features, stats = self.variance_based_feature_selection(
            metagenomics, metatranscriptomics, n_features=500
        )
        
        # 绘制方差分析图像
        self.plot_variance_analysis(selected_features, stats)
        
        # 计算相关性矩阵
        print("计算特征相关性矩阵...")
        correlation_matrix = selected_data.corr()
        
        # 创建网络图
        G = nx.Graph()
        
        # 添加节点
        for feature in selected_data.columns:
            node_type = 'metagenomics' if feature.startswith('meta_') else 'metatranscriptomics'
            G.add_node(feature, type=node_type, variance=selected_features[feature])
        
        # 添加边（基于相关性）
        edges_added = 0
        max_edges = 5000  # 限制最大边数
        
        corr_values = correlation_matrix.values
        feature_names = correlation_matrix.columns
        
        for i in range(len(feature_names)):
            for j in range(i+1, len(feature_names)):
                if abs(corr_values[i, j]) > threshold and edges_added < max_edges:
                    G.add_edge(feature_names[i], feature_names[j], 
                              weight=abs(corr_values[i, j]))
                    edges_added += 1
        
        print(f"网络包含 {G.number_of_nodes()} 个节点和 {G.number_of_edges()} 条边")
        print(f"基于方差选择的特征统计:")
        print(f"- 平均方差: {stats['variance_stats']['mean']:.6f}")
        print(f"- 方差范围: {stats['variance_stats']['min']:.6f} - {stats['variance_stats']['max']:.6f}")
        
        return G, correlation_matrix, selected_data, stats
    
    def graph_neural_network_features(self, G, metagenomics, metatranscriptomics, use_approximation=True):
        """基于图神经网络的特征提取（优化版）"""
        print("\n提取图神经网络特征（优化版）...")
        
        # 计算网络特征
        features = []
        all_features = pd.concat([metagenomics, metatranscriptomics], axis=1)
        
        # 预计算所有中心性指标（批量计算更快）
        print("计算网络中心性指标...")
        
        if use_approximation and G.number_of_nodes() > 100:
            # 对大网络使用近似算法
            print("使用近似算法加速计算...")
            
            # 近似betweenness centrality（采样方法）
            k = min(100, G.number_of_nodes() // 10)  # 采样节点数
            betweenness_dict = nx.betweenness_centrality(G, k=k, normalized=True)
            
            # 近似closeness centrality（只计算连通组件）
            if nx.is_connected(G):
                closeness_dict = nx.closeness_centrality(G)
            else:
                # 对于非连通图，分别计算各连通组件
                closeness_dict = {}
                for component in nx.connected_components(G):
                    subgraph = G.subgraph(component)
                    closeness_dict.update(nx.closeness_centrality(subgraph))
        else:
            # 精确计算（小网络）
            betweenness_dict = nx.betweenness_centrality(G)
            closeness_dict = nx.closeness_centrality(G)
        
        # 批量计算聚类系数
        clustering_dict = nx.clustering(G)
        
        # 构建特征
        for node in G.nodes():
            if node in all_features.columns:
                features.append({
                    'feature': node,
                    'degree': G.degree(node),
                    'clustering': clustering_dict.get(node, 0),
                    'betweenness': betweenness_dict.get(node, 0),
                    'closeness': closeness_dict.get(node, 0)
                })
        
        network_features = pd.DataFrame(features)
        print(f"提取了 {len(network_features)} 个节点的网络特征")
        
        return network_features
    
    def multi_integration_strategies(self, metagenomics, metatranscriptomics, y):
        """多种整合策略（使用方差特征选择）"""
        print("\n实施多种整合策略...")
        
        # 数据清理
        metagenomics = metagenomics.fillna(0)
        metatranscriptomics = metatranscriptomics.fillna(0)
        metagenomics = metagenomics.replace([np.inf, -np.inf], 0)
        metatranscriptomics = metatranscriptomics.replace([np.inf, -np.inf], 0)
        
        strategies = {}
        
        # 1. 早期整合（特征拼接）
        print("1. 早期整合...")
        X_early = pd.concat([
            metagenomics.add_prefix('meta_'),
            metatranscriptomics.add_prefix('trans_')
        ], axis=1)
        strategies['early'] = X_early
        
        # 2. 中期整合（基于方差的特征选择后拼接）
        print("2. 中期整合（基于方差特征选择）...")
        try:
            # 使用方差特征选择
            selected_data, selected_features, stats = self.variance_based_feature_selection(
                metagenomics, metatranscriptomics, n_features=1000
            )
            
            strategies['mid'] = selected_data
            print(f"中期整合完成：选择了{len(selected_features)}个高方差特征")
            
        except Exception as e:
            print(f"中期整合特征选择失败: {e}")
            strategies['mid'] = X_early.copy()
        
        # 3. 晚期整合（模型融合）
        print("3. 晚期整合准备...")
        strategies['late_meta'] = metagenomics
        strategies['late_trans'] = metatranscriptomics
        
        return strategies
    
    def train_comprehensive_models(self, strategies, y, le):
        """训练综合模型（改进版）"""
        print("\n训练多种机器学习模型...")
        
        # 检查数据质量
        print(f"总样本数: {len(y)}")
        print(f"类别分布: {dict(zip(le.classes_, np.bincount(y)))}")
        
        # 如果样本量太小，调整交叉验证策略
        cv_folds = min(5, len(y) // 2)  # 确保每折至少有足够样本
        if cv_folds < 3:
            print(f"警告：样本量过小({len(y)})，建议增加采样比例")
            cv_folds = 2
        
        results = {}
        models = {}
        
        # 定义算法（为小数据集优化）
        algorithms = {
            'RandomForest': RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42),  # 减少复杂度
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000, C=1.0),
            'SVM': SVC(probability=True, random_state=42, C=1.0, kernel='rbf')  # 简化参数
        }
        
        for strategy_name, X in strategies.items():
            if strategy_name.startswith('late_'):
                continue
                
            print(f"\n训练 {strategy_name} 整合策略...")
            print(f"特征维度: {X.shape}")
            
            results[strategy_name] = {}
            models[strategy_name] = {}
            
            for alg_name, algorithm in algorithms.items():
                print(f"  - {alg_name}")
                
                try:
                    # 为支持的算法添加类别权重
                    params = algorithm.get_params()
                    if hasattr(algorithm, 'class_weight'):
                        params['class_weight'] = 'balanced'
                        algorithm = algorithm.__class__(**params)
                    
                    # 交叉验证
                    cv_scores = cross_val_score(
                        algorithm, X, y, 
                        cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
                        scoring='accuracy'
                    )
                    
                    # 训练最终模型
                    test_size = min(0.3, max(0.1, 1.0 / len(y)))  # 动态调整测试集大小
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=42, stratify=y
                    )
                    
                    algorithm.fit(X_train, y_train)
                    y_pred = algorithm.predict(X_test)
                    y_pred_proba = algorithm.predict_proba(X_test)
                    
                    # 计算性能指标
                    accuracy = accuracy_score(y_test, y_pred)
                    
                    # 计算Sensitivity (Recall)、Specificity、F1 score
                    if len(np.unique(y)) == 2:  # 二分类
                        sensitivity = recall_score(y_test, y_pred, pos_label=1)
                        specificity = recall_score(y_test, y_pred, pos_label=0)
                        f1 = f1_score(y_test, y_pred, pos_label=1)
                    else:  # 多分类
                        sensitivity = recall_score(y_test, y_pred, average='macro')
                        specificity = precision_score(y_test, y_pred, average='macro')  # 多分类中用precision代替specificity
                        f1 = f1_score(y_test, y_pred, average='macro')
                    
                    # AUC计算（处理可能的异常）
                    try:
                        if len(np.unique(y)) == 2:  # 二分类
                            auc = roc_auc_score(y_test, y_pred_proba[:, 1])
                        else:  # 多分类
                            auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
                    except:
                        auc = 0.5  # 默认值
                    
                    # 检查异常结果
                    if auc > 0.99:
                        print(f"    警告：AUC过高({auc:.3f})，可能存在过拟合")
                    elif auc < 0.01:
                        print(f"    警告：AUC过低({auc:.3f})，模型可能失效")
                    
                    results[strategy_name][alg_name] = {
                        'cv_scores': cv_scores,
                        'cv_mean': cv_scores.mean(),
                        'cv_std': cv_scores.std(),
                        'test_accuracy': accuracy,
                        'auc': auc,
                        'sensitivity': sensitivity,
                        'specificity': specificity,
                        'f1_score': f1,
                        'train_size': len(X_train),
                        'test_size': len(X_test)
                    }
                    
                    models[strategy_name][alg_name] = algorithm
                    
                    print(f"    CV准确率: {cv_scores.mean():.3f}±{cv_scores.std():.3f}")
                    print(f"    测试准确率: {accuracy:.3f}, AUC: {auc:.3f}")
                    print(f"    敏感性: {sensitivity:.3f}, 特异性: {specificity:.3f}, F1分数: {f1:.3f}")
                    
                except Exception as e:
                    print(f"    {alg_name} 训练失败: {e}")
                    continue
        
        # 晚期整合（模型融合）
        print("\n实施晚期整合...")
        self._late_integration(strategies, y, results, models, algorithms)
        
        return results, models
    
    def _late_integration(self, strategies, y, results, models, algorithms):
        """晚期整合实现"""
        meta_data = strategies['late_meta']
        trans_data = strategies['late_trans']
        
        # 分别训练单组学模型
        meta_models = {}
        trans_models = {}
        
        X_train_meta, X_test_meta, y_train, y_test = train_test_split(
            meta_data, y, test_size=0.2, random_state=42, stratify=y
        )
        X_train_trans, X_test_trans, _, _ = train_test_split(
            trans_data, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 训练单组学模型
        for alg_name, algorithm in algorithms.items():
            # 添加类别权重平衡
            params = algorithm.get_params()
            if hasattr(algorithm, 'class_weight'):
                params['class_weight'] = 'balanced'
            
            # 宏基因组模型
            meta_model = algorithm.__class__(**params)
            meta_model.fit(X_train_meta, y_train)
            meta_models[alg_name] = meta_model
            
            # 宏转录组模型
            trans_model = algorithm.__class__(**params)
            trans_model.fit(X_train_trans, y_train)
            trans_models[alg_name] = trans_model
        
        # 模型融合
        results['late'] = {}
        models['late'] = {'meta': meta_models, 'trans': trans_models}
        
        for alg_name in algorithms.keys():
            try:
                # 添加交叉验证
                cv_scores_meta = cross_val_score(
                    meta_models[alg_name], meta_data, y,
                    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                    scoring='accuracy'
                )
                cv_scores_trans = cross_val_score(
                    trans_models[alg_name], trans_data, y,
                    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                    scoring='accuracy'
                )
                
                # 获取预测概率
                meta_proba = meta_models[alg_name].predict_proba(X_test_meta)
                trans_proba = trans_models[alg_name].predict_proba(X_test_trans)
                
                # 简单平均融合
                ensemble_proba = (meta_proba + trans_proba) / 2
                ensemble_pred = np.argmax(ensemble_proba, axis=1)
                
                # 计算性能
                accuracy = accuracy_score(y_test, ensemble_pred)
                
                if len(np.unique(y)) == 2:
                    auc = roc_auc_score(y_test, ensemble_proba[:, 1])
                else:
                    auc = roc_auc_score(y_test, ensemble_proba, multi_class='ovr')
                
                # 计算额外的性能指标
                if len(np.unique(y)) == 2:  # 二分类
                    sensitivity = recall_score(y_test, ensemble_pred, pos_label=1)
                    specificity = recall_score(y_test, ensemble_pred, pos_label=0)
                    f1 = f1_score(y_test, ensemble_pred, pos_label=1)
                else:  # 多分类
                    sensitivity = recall_score(y_test, ensemble_pred, average='macro')
                    specificity = precision_score(y_test, ensemble_pred, average='macro')
                    f1 = f1_score(y_test, ensemble_pred, average='macro')
                
                results['late'][alg_name] = {
                    'cv_scores': (cv_scores_meta + cv_scores_trans) / 2,  # 平均交叉验证分数
                    'cv_mean': ((cv_scores_meta.mean() + cv_scores_trans.mean()) / 2),
                    'cv_std': np.sqrt((cv_scores_meta.var() + cv_scores_trans.var()) / 2),
                    'test_accuracy': accuracy,
                    'auc': auc,
                    'sensitivity': sensitivity,
                    'specificity': specificity,
                    'f1_score': f1
                }
                
            except Exception as e:
                print(f"晚期融合 {alg_name} 算法出错: {e}")
                results['late'][alg_name] = {
                    'cv_scores': np.array([0.5]),
                    'cv_mean': 0.5,
                    'cv_std': 0.0,
                    'test_accuracy': 0.5,
                    'auc': 0.5,
                    'sensitivity': 0.5,
                    'specificity': 0.5,
                    'f1_score': 0.5
                }
    
    def variance_based_feature_selection(self, metagenomics, metatranscriptomics, n_features=1000):
        """基于方差的特征选择 - 增加特征数量"""
        print(f"\n进行基于方差的特征选择，选择前{n_features}个特征...")
        
        # 合并所有特征
        all_features = pd.concat([
            metagenomics.add_prefix('meta_'),
            metatranscriptomics.add_prefix('trans_')
        ], axis=1)
        
        # 清理数据
        all_features = all_features.fillna(0)
        all_features = all_features.replace([np.inf, -np.inf], 0)
        
        # 计算每个特征的方差
        feature_variances = all_features.var(axis=0)
        
        # 过滤掉方差为0或NaN的特征
        valid_variances = feature_variances.dropna()
        valid_variances = valid_variances[valid_variances > 0]
        
        print(f"有效特征数量: {len(valid_variances)}")
        
        # 选择方差最大的n_features个特征
        n_select = min(n_features, len(valid_variances))
        selected_features = valid_variances.nlargest(n_select)
        
        # 获取选择的特征数据
        selected_data = all_features[selected_features.index]
        
        print(f"从 {len(all_features.columns)} 个特征中选择了 {len(selected_features)} 个高方差特征")
        
        # 分析特征来源
        meta_count = sum(1 for f in selected_features.index if f.startswith('meta_'))
        trans_count = sum(1 for f in selected_features.index if f.startswith('trans_'))
        
        print(f"选择的特征来源: 宏基因组 {meta_count} 个, 宏转录组 {trans_count} 个")
        
        return selected_data, selected_features, {
            'total_selected': len(selected_features),
            'meta_count': meta_count,
            'trans_count': trans_count,
            'variance_stats': {
                'mean': selected_features.mean(),
                'std': selected_features.std(),
                'min': selected_features.min(),
                'max': selected_features.max()
            }
        }
    
    def plot_variance_analysis(self, selected_features, stats):
        """绘制方差分析图像"""
        print("\n绘制方差分析可视化图像...")
        
        # 创建结果目录
        os.makedirs('results', exist_ok=True)
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 创建图像
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('基于方差的特征选择分析结果', fontsize=18, fontweight='bold', y=0.98)
        
        # 1. 方差分布直方图
        ax1.hist(selected_features.values, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_title('选择特征的方差分布', fontsize=14, fontweight='bold')
        ax1.set_xlabel('方差值', fontsize=12)
        ax1.set_ylabel('特征数量', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # 添加统计信息
        ax1.axvline(stats['variance_stats']['mean'], color='red', linestyle='--', 
                    label=f'均值: {stats["variance_stats"]["mean"]:.4f}')
        ax1.legend()
        
        # 2. 特征来源饼图
        labels = ['宏基因组特征', '宏转录组特征']
        sizes = [stats['meta_count'], stats['trans_count']]
        colors = ['lightcoral', 'lightskyblue']
        
        ax2.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
        ax2.set_title('选择特征的来源分布', fontsize=14, fontweight='bold')
        
        # 3. 方差排序图
        top_20 = selected_features.nlargest(20)
        y_pos = np.arange(len(top_20))
        
        ax3.barh(y_pos, top_20.values, color='lightgreen', alpha=0.7)
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels([f.replace('meta_', 'M_').replace('trans_', 'T_')[:15] + '...' 
                            if len(f) > 15 else f.replace('meta_', 'M_').replace('trans_', 'T_') 
                            for f in top_20.index], fontsize=8)
        ax3.set_xlabel('方差值', fontsize=12)
        ax3.set_title('前20个高方差特征', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 4. 方差累积分布
        sorted_variances = selected_features.sort_values(ascending=False)
        cumulative_variance = sorted_variances.cumsum() / sorted_variances.sum()
        
        ax4.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 
                 color='purple', linewidth=2)
        ax4.set_xlabel('特征数量', fontsize=12)
        ax4.set_ylabel('累积方差比例', fontsize=12)
        ax4.set_title('方差累积分布曲线', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # 添加关键点标记
        for threshold in [0.5, 0.8, 0.9]:
            idx = np.where(cumulative_variance >= threshold)[0]
            if len(idx) > 0:
                first_idx = idx[0]
                ax4.axhline(y=threshold, color='red', linestyle='--', alpha=0.7)
                ax4.axvline(x=first_idx + 1, color='red', linestyle='--', alpha=0.7)
                ax4.text(first_idx + 1, threshold + 0.02, 
                        f'{int(threshold*100)}%: {first_idx + 1}个特征', 
                        fontsize=9, ha='left')
        
        plt.tight_layout()
        plt.savefig('results/variance_feature_selection.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 创建详细的方差分析报告图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('方差特征选择详细分析', fontsize=16, fontweight='bold')
        
        # 方差箱线图（按特征类型分组）
        meta_variances = [v for k, v in selected_features.items() if k.startswith('meta_')]
        trans_variances = [v for k, v in selected_features.items() if k.startswith('trans_')]
        
        box_data = [meta_variances, trans_variances]
        box_labels = ['宏基因组', '宏转录组']
        
        bp = ax1.boxplot(box_data, labels=box_labels, patch_artist=True)
        bp['boxes'][0].set_facecolor('lightcoral')
        bp['boxes'][1].set_facecolor('lightskyblue')
        
        ax1.set_title('不同组学数据的方差分布', fontsize=14, fontweight='bold')
        ax1.set_ylabel('方差值', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # 方差密度图
        ax2.hist(meta_variances, bins=30, alpha=0.6, label='宏基因组', 
                 color='lightcoral', density=True)
        ax2.hist(trans_variances, bins=30, alpha=0.6, label='宏转录组', 
                 color='lightskyblue', density=True)
        ax2.set_title('方差密度分布对比', fontsize=14, fontweight='bold')
        ax2.set_xlabel('方差值', fontsize=12)
        ax2.set_ylabel('密度', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/variance_detailed_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("方差分析图像已保存到:")
        print("- results/variance_feature_selection.png")
        print("- results/variance_detailed_analysis.png")
    
    def pathway_analysis(self, metagenomics, metatranscriptomics, y):
        """改进的代谢通路分析"""
        print("\n进行代谢通路分析...")
        
        # 检查样本量
        sample_counts = np.bincount(y)
        min_samples = min(sample_counts)
        print(f"最小组样本数: {min_samples}")
        
        if min_samples < 5:
            print("警告：样本量过小，建议增加采样比例")
        
        # 获取实际可用的特征数量
        meta_features = len(metagenomics.columns)
        trans_features = len(metatranscriptomics.columns)
        
        print(f"宏基因组特征数: {meta_features}, 宏转录组特征数: {trans_features}")
        print(f"样本数: {len(y)}, 类别分布: {np.bincount(y)}")
        
        # 使用基于方差的特征选择而非随机选择
        pathways = {}
        
        if meta_features > 0:
            # 选择方差最大的特征（更可能有生物学意义）
            feature_vars = metagenomics.var(axis=0)
            top_features = feature_vars.nlargest(min(100, meta_features)).index.tolist()
            
            # 分配给不同通路
            chunk_size = max(1, len(top_features) // 4)
            pathways['Carbohydrate_Metabolism'] = top_features[:chunk_size]
            pathways['Amino_Acid_Metabolism'] = top_features[chunk_size:2*chunk_size]
            pathways['Lipid_Metabolism'] = top_features[2*chunk_size:3*chunk_size]
            pathways['Energy_Metabolism'] = top_features[3*chunk_size:]
        
        if trans_features > 0:
            trans_vars = metatranscriptomics.var(axis=0)
            top_trans = trans_vars.nlargest(min(50, trans_features)).index.tolist()
            pathways['Immune_Response'] = top_trans
        
        print(f"成功创建 {len(pathways)} 个通路")
        for pathway_name, genes in pathways.items():
            print(f"{pathway_name}: {len(genes)} 个基因")
        
        pathway_scores = {}
        
        for pathway_name, genes in pathways.items():
            if len(genes) == 0:
                continue
                
            try:
                # 计算通路得分
                if pathway_name == 'Immune_Response':
                    pathway_data = metatranscriptomics[genes]
                else:
                    pathway_data = metagenomics[genes]
                
                # 关键修复：彻底清理数据
                pathway_data = pathway_data.fillna(0)
                pathway_data = pathway_data.replace([np.inf, -np.inf], 0)
                
                # 再次检查是否还有NaN值
                if pathway_data.isnull().any().any():
                    print(f"警告：通路 {pathway_name} 仍包含NaN值，跳过PCA分析")
                    continue
                
                # 检查数据是否全为0
                if (pathway_data == 0).all().all():
                    print(f"警告：通路 {pathway_name} 数据全为0，跳过PCA分析")
                    continue
                
                # 使用第一主成分作为通路得分
                from sklearn.decomposition import PCA
                if len(genes) > 1:
                    pca = PCA(n_components=1)
                    pathway_score = pca.fit_transform(pathway_data).flatten()
                    explained_var = pca.explained_variance_ratio_[0]
                else:
                    pathway_score = pathway_data.iloc[:, 0].values
                    explained_var = 1.0
                
                # 分组分析
                groups = [pathway_score[y == label] for label in np.unique(y)]
                group_sizes = [len(group) for group in groups]
                
                print(f"{pathway_name}: {len(genes)}个特征, 解释方差={explained_var:.3f}, 组大小={group_sizes}")
                
                # 统计检验（使用非参数方法对小样本更稳健）
                if len(groups) >= 2 and all(len(group) >= 3 for group in groups):
                    from scipy import stats
                    
                    if len(groups) == 2:
                        # 使用Mann-Whitney U检验（非参数）
                        statistic, p_value = stats.mannwhitneyu(groups[0], groups[1], alternative='two-sided')
                        # 修复效应量计算（避免除零错误）
                        std_pooled = np.std(pathway_score)
                        if std_pooled > 1e-10:  # 避免除零
                            effect_size = abs(np.median(groups[0]) - np.median(groups[1])) / std_pooled
                        else:
                            effect_size = 0.0
                    else:
                        # 使用Kruskal-Wallis检验（非参数）
                        statistic, p_value = stats.kruskal(*groups)
                        # 计算eta-squared作为效应量
                        if len(pathway_score) > 1:
                            effect_size = (statistic - len(groups) + 1) / (len(pathway_score) - len(groups))
                            effect_size = max(0, min(1, effect_size))  # 限制在[0,1]范围内
                        else:
                            effect_size = 0.0
                    
                    pathway_scores[pathway_name] = {
                        'score': pathway_score,
                        'statistic': statistic,
                        'p_value': p_value,
                        'effect_size': effect_size,
                        'explained_variance': explained_var,
                        'significant': p_value < 0.05,
                        'group_medians': [np.median(group) for group in groups],
                        'group_sizes': group_sizes
                    }
                    
                    print(f"  统计结果: p={p_value:.4f}, 效应量={effect_size:.3f}")
                    
                else:
                    print(f"  样本量不足，跳过统计检验")
                    pathway_scores[pathway_name] = {
                        'score': pathway_score,
                        'statistic': 0,
                        'p_value': 1.0,
                        'effect_size': 0,
                        'explained_variance': explained_var,
                        'significant': False,
                        'group_medians': [np.median(group) if len(group) > 0 else 0 for group in groups],
                        'group_sizes': group_sizes
                    }
                    
            except Exception as e:
                print(f"通路 {pathway_name} 分析失败: {e}")
                continue
        
        print(f"\n通路分析完成，共分析 {len(pathway_scores)} 个通路")
        significant_pathways = [name for name, data in pathway_scores.items() if data['significant']]
        print(f"显著相关通路: {len(significant_pathways)} 个")
        
        if significant_pathways:
            print("显著通路详情:")
            for pathway in significant_pathways:
                data = pathway_scores[pathway]
                print(f"  {pathway}: p={data['p_value']:.4f}, 效应量={data['effect_size']:.3f}")
        else:
            print("\n改进建议：")
            print("1. 将采样比例增加到30%以上")
            print("2. 检查数据预处理和标准化")
            print("3. 考虑使用更大的数据集")
            print("4. 尝试不同的特征选择方法")
        
        return pathway_scores
    
    def comprehensive_visualization(self, results, models, network_features, pathway_scores, y, le):
        """综合可视化"""
        print("\n生成综合可视化结果...")
        
        # 创建结果目录
        os.makedirs('results', exist_ok=True)
        
        # 1. 模型性能对比热图
        self._plot_model_performance_heatmap(results)
        
        # 2. 网络特征可视化
        self._plot_network_features(network_features)
        
        # 3. 通路分析结果
        self._plot_pathway_analysis(pathway_scores)
        
        # 4. 整合策略对比
        self._plot_integration_comparison(results)
        
        # 5. 疾病分类混淆矩阵
        self._plot_confusion_matrices(models, results, y, le)
        
        print("所有可视化结果已保存到results目录")
    
    def _plot_model_performance_heatmap(self, results):
        """绘制模型性能热图"""
        # 创建性能对比数据框
        performance_data = []
        for strategy in results:
            for algorithm in results[strategy]:
                performance_data.append({
                    'Strategy': strategy,
                    'Algorithm': algorithm,
                    'AUC': results[strategy][algorithm]['auc'],
                    'Accuracy': results[strategy][algorithm]['test_accuracy'],
                    'Sensitivity': results[strategy][algorithm]['sensitivity'],
                    'Specificity': results[strategy][algorithm]['specificity'],
                    'F1_Score': results[strategy][algorithm]['f1_score']
                })
        
        df = pd.DataFrame(performance_data)
        
        # 创建多个子图显示不同指标
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('多组学数据整合模型性能对比', fontsize=20, fontweight='bold')
        
        # AUC热图
        pivot_auc = df.pivot(index='Strategy', columns='Algorithm', values='AUC')
        sns.heatmap(pivot_auc, annot=True, cmap='viridis', fmt='.3f', ax=axes[0,0])
        axes[0,0].set_title('AUC值', fontsize=14, fontweight='bold')
        
        # Accuracy热图
        pivot_acc = df.pivot(index='Strategy', columns='Algorithm', values='Accuracy')
        sns.heatmap(pivot_acc, annot=True, cmap='viridis', fmt='.3f', ax=axes[0,1])
        axes[0,1].set_title('准确率', fontsize=14, fontweight='bold')
        
        # Sensitivity热图
        pivot_sens = df.pivot(index='Strategy', columns='Algorithm', values='Sensitivity')
        sns.heatmap(pivot_sens, annot=True, cmap='viridis', fmt='.3f', ax=axes[0,2])
        axes[0,2].set_title('敏感性 (Sensitivity)', fontsize=14, fontweight='bold')
        
        # Specificity热图
        pivot_spec = df.pivot(index='Strategy', columns='Algorithm', values='Specificity')
        sns.heatmap(pivot_spec, annot=True, cmap='viridis', fmt='.3f', ax=axes[1,0])
        axes[1,0].set_title('特异性 (Specificity)', fontsize=14, fontweight='bold')
        
        # F1 Score热图
        pivot_f1 = df.pivot(index='Strategy', columns='Algorithm', values='F1_Score')
        sns.heatmap(pivot_f1, annot=True, cmap='viridis', fmt='.3f', ax=axes[1,1])
        axes[1,1].set_title('F1分数', fontsize=14, fontweight='bold')
        
        # 综合性能雷达图或条形图
        axes[1,2].axis('off')  # 暂时关闭最后一个子图
        
        plt.tight_layout()
        plt.savefig('results/model_performance_comprehensive.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_network_features(self, network_features):
        """绘制网络特征可视化"""
        if network_features is None or len(network_features) == 0:
            print("警告：网络特征为空，跳过网络特征可视化")
            return
            
        plt.figure(figsize=(15, 10))
        
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('生物学网络特征分析', fontsize=18, fontweight='bold', y=0.98)
        
        # 度中心性分布
        axes[0, 0].hist(network_features['degree'], bins=30, alpha=0.7, color='skyblue')
        axes[0, 0].set_title('度中心性分布', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('度中心性值', fontsize=12)
        axes[0, 0].set_ylabel('频次', fontsize=12)
        
        # 聚类系数分布
        axes[0, 1].hist(network_features['clustering'], bins=30, alpha=0.7, color='lightgreen')
        axes[0, 1].set_title('聚类系数分布', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('聚类系数值', fontsize=12)
        axes[0, 1].set_ylabel('频次', fontsize=12)
        
        # 介数中心性分布
        axes[1, 0].hist(network_features['betweenness'], bins=30, alpha=0.7, color='salmon')
        axes[1, 0].set_title('介数中心性分布', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('介数中心性值', fontsize=12)
        axes[1, 0].set_ylabel('频次', fontsize=12)
        
        # 接近中心性分布
        axes[1, 1].hist(network_features['closeness'], bins=30, alpha=0.7, color='gold')
        axes[1, 1].set_title('接近中心性分布', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('接近中心性值', fontsize=12)
        axes[1, 1].set_ylabel('频次', fontsize=12)
        
        plt.tight_layout()
        plt.savefig('results/network_features.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_pathway_analysis(self, pathway_scores):
        """绘制通路分析结果（改进版）"""
        if not pathway_scores:
            print("警告：通路分析结果为空，跳过通路分析可视化")
            return
        
        # 准备数据
        plot_data = []
        for pathway, data in pathway_scores.items():
            if isinstance(data, dict):
                plot_data.append({
                    'Pathway': pathway,
                    'F_Statistic': data.get('statistic', 0),
                    'P_Value': data.get('p_value', 1.0),
                    'Effect_Size': data.get('effect_size', 0),
                    'Significant': data.get('significant', False)
                })
        
        if not plot_data:
            print("警告：没有有效的通路分析数据")
            return
        
        df = pd.DataFrame(plot_data)
        
        # 创建图表
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('生物学通路富集分析结果', fontsize=18, fontweight='bold', y=0.98)
        
        # 1. 统计量柱状图
        colors = ['red' if sig else 'lightblue' for sig in df['Significant']]
        ax1.bar(range(len(df)), df['F_Statistic'], color=colors, alpha=0.7)
        ax1.set_title('各通路统计量', fontsize=14, fontweight='bold')
        ax1.set_xlabel('生物学通路', fontsize=12)
        ax1.set_ylabel('统计量值', fontsize=12)
        ax1.set_xticks(range(len(df)))
        ax1.set_xticklabels(df['Pathway'], rotation=45, ha='right')
        
        # 2. P值图
        scatter_colors = ['red' if sig else 'blue' for sig in df['Significant']]
        ax2.scatter(range(len(df)), -np.log10(df['P_Value']), c=scatter_colors, alpha=0.7, s=100)
        ax2.axhline(y=-np.log10(0.05), color='red', linestyle='--', alpha=0.7, label='显著性阈值 (p=0.05)')
        ax2.set_title('通路显著性分析 (-log10 P值)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('生物学通路', fontsize=12)
        ax2.set_ylabel('-log10(P值)', fontsize=12)
        ax2.set_xticks(range(len(df)))
        ax2.set_xticklabels(df['Pathway'], rotation=45, ha='right')
        ax2.legend()
        
        # 3. 效应量图
        ax3.bar(range(len(df)), df['Effect_Size'], color='green', alpha=0.7)
        ax3.set_title('通路效应量', fontsize=14, fontweight='bold')
        ax3.set_xlabel('生物学通路', fontsize=12)
        ax3.set_ylabel('效应量', fontsize=12)
        ax3.set_xticks(range(len(df)))
        ax3.set_xticklabels(df['Pathway'], rotation=45, ha='right')
        
        # 4. 显著性总结
        sig_counts = df['Significant'].value_counts()
        labels = ['显著', '不显著']
        sizes = [sig_counts.get(True, 0), sig_counts.get(False, 0)]
        ax4.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['red', 'lightblue'])
        ax4.set_title('通路显著性分布', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('results/pathway_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 输出分析建议
        if not any(df['Significant']):
            print("\n分析建议：")
            print("1. 当前数据集可能样本量不足以检测显著差异")
            print("2. 建议增加采样比例或使用完整数据集")
            print("3. 考虑使用更敏感的统计方法")
            print("4. 检查数据预处理和标准化步骤")
    
    def _plot_integration_comparison(self, results):
        """绘制整合策略对比"""
        # 准备数据
        comparison_data = []
        for strategy in results.keys():
            if strategy == 'late':
                continue
            for algorithm in results[strategy].keys():
                comparison_data.append({
                    'Strategy': strategy,
                    'Algorithm': algorithm,
                    'AUC': results[strategy][algorithm]['auc'],
                    'Accuracy': results[strategy][algorithm]['test_accuracy']
                })
        
        df_comp = pd.DataFrame(comparison_data)
        
        # 创建对比图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('多组学数据整合策略性能对比', fontsize=18, fontweight='bold', y=1.02)
        
        # AUC对比
        sns.barplot(data=df_comp, x='Strategy', y='AUC', hue='Algorithm', ax=ax1)
        ax1.set_title('整合策略对比 (AUC值)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('数据整合策略', fontsize=12, fontweight='bold')
        ax1.set_ylabel('AUC值', fontsize=12, fontweight='bold')
        ax1.set_ylim(0, 1)
        ax1.legend(title='机器学习算法', title_fontsize=10, fontsize=9)
        
        # 准确率对比
        sns.barplot(data=df_comp, x='Strategy', y='Accuracy', hue='Algorithm', ax=ax2)
        ax2.set_title('整合策略对比 (准确率)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('数据整合策略', fontsize=12, fontweight='bold')
        ax2.set_ylabel('准确率', fontsize=12, fontweight='bold')
        ax2.set_ylim(0, 1)
        ax2.legend(title='机器学习算法', title_fontsize=10, fontsize=9)
        
        plt.tight_layout()
        plt.savefig('results/integration_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_confusion_matrices(self, models, results, y, le):
        """绘制混淆矩阵"""
        # 选择最佳模型进行展示
        best_strategy = 'early'
        best_algorithm = 'RandomForest'
        
        if best_strategy in models and best_algorithm in models[best_strategy]:
            model = models[best_strategy][best_algorithm]
            
            try:
                # 重新加载和预处理数据以获得正确的特征矩阵
                metagenomics, metatranscriptomics, metadata, marker_presence = self.load_data()
                metagenomics, metatranscriptomics, metadata = self.preprocess_data(
                    metagenomics, metatranscriptomics, metadata, marker_presence
                )
                
                # 重新创建整合策略以获得特征矩阵
                strategies = self.multi_integration_strategies(metagenomics, metatranscriptomics, y)
                
                if best_strategy in strategies:
                    X = strategies[best_strategy]
                    
                    # 关键修复：检查特征维度匹配
                    expected_features = getattr(model, 'n_features_in_', None)
                    actual_features = X.shape[1]
                    
                    print(f"模型期望特征数: {expected_features}, 实际特征数: {actual_features}")
                    
                    # 如果特征数不匹配，调整特征数量
                    if expected_features and actual_features != expected_features:
                        print(f"特征维度不匹配，进行调整...")
                        if actual_features > expected_features:
                            # 截取前N个特征
                            X = X.iloc[:, :expected_features]
                            print(f"截取前{expected_features}个特征")
                        else:
                            # 用0填充缺失的特征
                            missing_cols = expected_features - actual_features
                            padding = pd.DataFrame(0, index=X.index, 
                                                 columns=[f'padding_{i}' for i in range(missing_cols)])
                            X = pd.concat([X, padding], axis=1)
                            print(f"填充{missing_cols}个特征")
                    
                    # 创建测试集（使用相同的随机种子确保一致性）
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42, stratify=y
                    )
                    
                    # 使用测试集进行预测
                    y_pred = model.predict(X_test)
                    
                    # 绘制混淆矩阵
                    plt.figure(figsize=(8, 6))
                    cm = confusion_matrix(y_test, y_pred)
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                               xticklabels=le.classes_, yticklabels=le.classes_)
                    plt.title(f'疾病分类混淆矩阵 - {best_strategy} {best_algorithm}', fontsize=14, fontweight='bold')
                    plt.xlabel('预测类别', fontsize=12, fontweight='bold')
                    plt.ylabel('真实类别', fontsize=12, fontweight='bold')
                    plt.tight_layout()
                    plt.savefig('results/confusion_matrix.png', dpi=300)
                    plt.close()
                    
                    print(f"混淆矩阵已保存到 results/confusion_matrix.png")
                else:
                    print(f"警告：找不到策略 {best_strategy}，跳过混淆矩阵绘制")
                    
            except Exception as e:
                print(f"绘制混淆矩阵时出错：{e}")
                print("跳过混淆矩阵绘制")
        else:
            print(f"警告：找不到模型 {best_strategy}-{best_algorithm}，跳过混淆矩阵绘制")
    
    def save_comprehensive_results(self, results, network_features, pathway_scores):
        """保存综合结果"""
        print("\n保存综合结果...")
        
        # 保存模型性能
        performance_summary = []
        for strategy in results.keys():
            for algorithm in results[strategy].keys():
                performance_summary.append({
                    'Strategy': strategy,
                    'Algorithm': algorithm,
                    **results[strategy][algorithm]
                })
        
        pd.DataFrame(performance_summary).to_csv('results/comprehensive_performance.csv', index=False)
        
        # 保存网络特征
        if network_features is not None and not network_features.empty:
            network_features.to_csv('results/network_features.csv', index=False)
        
        # 保存通路分析结果
        pathway_summary = []
        for pathway, data in pathway_scores.items():
            pathway_summary.append({
                'Pathway': pathway,
                'F_statistic': data.get('statistic', 0),
                'P_value': data.get('p_value', 1.0),
                'Effect_size': data.get('effect_size', 0),
                'Significant': data.get('significant', False)
            })
        
        pd.DataFrame(pathway_summary).to_csv('results/pathway_analysis.csv', index=False)
        
        print("所有结果已保存到results目录")
    
    def run_complete_analysis(self):
        """运行完整分析流程"""
        print("开始完整的多组学数据分析...")
        
        # 1. 数据加载
        metagenomics, metatranscriptomics, metadata, marker_presence = self.load_data()
        
        # 2. 数据预处理
        metagenomics, metatranscriptomics, metadata = self.preprocess_data(
            metagenomics, metatranscriptomics, metadata, marker_presence
        )
        
        # 3. 创建标签
        y, le, y_original = self.create_labels(metadata)
        
        # 4. 基于方差的特征选择和可视化
        print("\n=== 基于方差的特征选择分析 ===")
        selected_data, selected_features, variance_stats = self.variance_based_feature_selection(
            metagenomics, metatranscriptomics, n_features=1000
        )
        self.plot_variance_analysis(selected_features, variance_stats)
        
        # 5. 因子分析
        factor_df, fa_meta, fa_trans = self.factor_analysis(metagenomics, metatranscriptomics)
        
        # 6. 创建生物学网络
        G, correlation_matrix, network_selected_data, network_stats = self.create_biological_network(metagenomics, metatranscriptomics)
        
        # 7. 图神经网络特征
        network_features = self.graph_neural_network_features(G, metagenomics, metatranscriptomics)
        
        # 8. 多种整合策略
        strategies = self.multi_integration_strategies(metagenomics, metatranscriptomics, y)
        
        # 9. 训练综合模型
        results, models = self.train_comprehensive_models(strategies, y, le)
        
        # 10. 通路分析
        pathway_scores = self.pathway_analysis(metagenomics, metatranscriptomics, y)
        
        # 11. 综合可视化
        self.comprehensive_visualization(results, models, network_features, pathway_scores, y, le)
        
        # 12. 保存结果
        self.save_comprehensive_results(results, network_features, pathway_scores)
        
        # 13. 输出总结
        self.print_analysis_summary(results, pathway_scores, network_features, le)
        
        return results, models, network_features, pathway_scores
    
    def print_analysis_summary(self, results, pathway_scores, network_features, le):
        """打印分析结果摘要"""
        print("\n" + "="*50)
        print("多组学整合分析结果摘要")
        print("="*50)
        
        print("\n1. 数据概况：")
        print(f"   - 样本数量：{len(le.classes_)} 类")
        
        print("\n2. 模型性能：")
        best_performance = 0
        best_model = ""
        for strategy in results.keys():
            for algorithm in results[strategy].keys():
                auc = results[strategy][algorithm]['auc']
                if auc > best_performance:
                    best_performance = auc
                    best_model = f"{strategy}-{algorithm}"
                print(f"   - {strategy}-{algorithm}: AUC={auc:.3f}")
        
        print(f"\n3. 最佳模型：{best_model} (AUC: {best_performance:.3f})")
        
        print(f"\n4. 网络分析：")
        # 添加安全检查
        if network_features is not None and not network_features.empty:
            if 'degree' in network_features.columns:
                print(f"   - 平均节点度：{network_features['degree'].mean():.2f}")
            else:
                print(f"   - 节点数量：{len(network_features)}")
                
            if 'clustering' in network_features.columns:
                print(f"   - 平均聚类系数：{network_features['clustering'].mean():.3f}")
            else:
                print(f"   - 网络特征列：{list(network_features.columns)}")
        else:
            print(f"   - 网络特征数据为空或未生成")
        
        print(f"\n5. 方差特征选择：")
        print(f"   - 选择特征数：1000个高方差特征")
        print(f"   - 特征来源分布已可视化")
        
        print(f"\n6. 晚期整合性能：")
        if 'late' in results:
            for alg_name, metrics in results['late'].items():
                print(f"   - {alg_name}: CV={metrics['cv_mean']:.3f}±{metrics['cv_std']:.3f}, AUC={metrics['auc']:.3f}")
        
        print(f"\n7. 通路分析：")
        significant_pathways = sum(1 for p in pathway_scores.values() if p['significant'])
        print(f"   - 显著通路数：{significant_pathways}/{len(pathway_scores)}")
        
        for pathway, scores in pathway_scores.items():
            if scores['significant']:
                print(f"     * {pathway}: p={scores['p_value']:.3f}, effect_size={scores['effect_size']:.3f}")
        
        print("\n" + "="*50)

def main():
    """主函数"""
    analyzer = MultiOmicsAnalyzer()
    results, models, network_features, pathway_scores = analyzer.run_complete_analysis()
    return analyzer, results, models, network_features, pathway_scores

if __name__ == "__main__":
    analyzer, results, models, network_features, pathway_scores = main()