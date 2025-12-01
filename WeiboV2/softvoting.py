# 文件名：soft_voting_final.py
# 功能：Soft Voting 融合 Machine + Crowd 预测（绝对不丢 isFake！）

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

# ==================== 1. 终极防丢标签读取 ====================
print("正在加载数据（终极防丢版）...")

# 原始标签（唯一可靠来源）
news_basic = pd.read_csv('news_basic.csv')
folder_to_label = dict(zip(news_basic['news_folder'], news_basic['isFake']))

# 读取 Machine 预测（BERT MC Dropout）
trainval_machine = pd.read_csv('splits/trainval_bert_mc.csv')
test_machine     = pd.read_csv('splits/test_bert_mc.csv')

# 读取 Crowd 预测（Beta 分布参数）
trainval_crowd = pd.read_csv('splits/trainval_crowd.csv')
test_crowd     = pd.read_csv('splits/test_crowd.csv')

# 合并 Machine + Crowd
trainval_df = trainval_machine.merge(trainval_crowd, on='news_folder', how='inner')
test_df     = test_machine.merge(test_crowd,      on='news_folder', how='inner')

# 强制添加真实标签 isFake（永不丢失！）
trainval_df['isFake'] = trainval_df['news_folder'].map(folder_to_label)
test_df['isFake']     = test_df['news_folder'].map(folder_to_label)

# 检查
print(f"训练+验证集合数: {len(trainval_df)}，测试集: {len(test_df)}")
print(f"标签是否完整 → 训练+验证: {trainval_df['isFake'].notna().all()}, 测试: {test_df['isFake'].notna().all()}")

# ==================== 2. Soft Voting 融合 ====================
# Machine 概率：prob_fake_mean（已经是 [0,1] 区间）
# Crowd 概率：Beta(α, β) 的期望值 = α / (α + β)

def beta_expectation(alpha, beta):
    return alpha / (alpha + beta + 1e-12)  # 防止除零

# 计算 Crowd 期望概率
trainval_df['crowd_prob'] = beta_expectation(trainval_df['crowd_alpha'], trainval_df['crowd_beta'])
test_df['crowd_prob']     = beta_expectation(test_df['crowd_alpha'],     test_df['crowd_beta'])

# Soft Voting：平均两个概率
trainval_df['soft_voting_prob'] = (trainval_df['prob_fake_mean'] + trainval_df['crowd_prob']) / 2
test_df['soft_voting_prob']     = (test_df['prob_fake_mean']     + test_df['crowd_prob'])     / 2

# 最终预测（阈值 0.5）
test_df['soft_voting_pred'] = (test_df['soft_voting_prob'] >= 0.5).astype(int)

# ==================== 3. 测试集指标 ====================
y_true = test_df['isFake'].values
y_pred = test_df['soft_voting_pred'].values
y_prob = test_df['soft_voting_prob'].values

acc = accuracy_score(y_true, y_pred)
p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
auc = roc_auc_score(y_true, y_prob)

print("\n" + "="*70)
print("           Soft Voting 融合结果（Machine + Crowd）")
print("="*70)
print(f"Accuracy        : {acc:.4f}")
print(f"Macro Precision : {p:.4f}")
print(f"Macro Recall    : {r:.4f}")
print(f"Macro F1        : {f1:.4f}")
print(f"AUC             : {auc:.4f}")
print("="*70)

# ==================== 4. 保存完整结果 ====================
result = test_df[['news_folder', 'isFake']].copy()
result['machine_prob'] = test_df['prob_fake_mean']
result['crowd_prob']   = test_df['crowd_prob']
result['soft_voting_prob'] = test_df['soft_voting_prob']
result['soft_voting_pred'] = test_df['soft_voting_pred']
result['correct'] = (result['soft_voting_pred'] == result['isFake']).astype(int)

result.to_csv('splits/SOFT_VOTING_RESULT.csv', index=False, encoding='utf-8-sig')
print("Soft Voting 结果已保存 → splits/SOFT_VOTING_RESULT.csv")