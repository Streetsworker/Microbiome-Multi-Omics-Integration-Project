# 多组学数据分析项目

## 项目简介

本项目实现了一个增强版多组学数据分析系统，集成了宏基因组学和宏转录组学数据，采用多种机器学习和深度学习方法进行疾病预测和生物标志物发现。

## 主要功能

- **多组学数据整合**：支持宏基因组和宏转录组数据的联合分析
- **高级特征工程**：包括因子分析、PCA降维、基于方差的特征选择
- **图神经网络**：构建生物学网络并提取网络特征
- **机器学习模型**：集成随机森林、逻辑回归、支持向量机等多种算法
- **通路分析**：基于KEGG通路的功能富集分析
- **可视化分析**：生成多种统计图表和网络可视化

## 数据集

项目使用HMP2（Human Microbiome Project 2）数据集：
- `abundance.csv`：宏基因组丰度数据
- `abundance_stoolsubset.csv`：宏转录组丰度数据
- `hmp2.csv`：样本元数据
- `marker_presence.csv`：微生物标记物存在性数据
- `markers2clades_DB.csv`：标记物到分类群的映射数据

## 安装要求

### Python版本
- Python 3.8+

### 依赖包
请参考 `requirements.txt` 或使用conda环境文件 `environment.yml`

## 快速开始

### 1. 克隆仓库
```bash
git clone https://github.com/your-username/multi-omics-analysis.git
cd multi-omics-analysis