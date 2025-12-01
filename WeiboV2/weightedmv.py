# 文件名：weighted_voting_fixed.py  （直接覆盖你原来的 weightedmv.py）
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, roc_auc_score
)
import os

print("=== 加权投票（Weighted Voting）最终修复版 ===\n正在加载数据...")

# 1. 加载所有评论（包含所有新闻）
comments_all = pd.read_csv('news_with_comments.csv', dtype={'user_id': str})

# 2. 加载划分好的 train/val/test（只取 news_folder 和 isFake）
train_df = pd.read_csv('splits/news_basic_train.csv')[['news_folder', 'isFake']]
val_df   = pd.read_csv('splits/news_basic_val.csv')[['news_folder', 'isFake']]
test_df  = pd.read_csv('splits/news_basic_test.csv')[['news_folder', 'isFake']]

# 合并 train + val 作为历史数据
history_news = pd.concat([train_df, val_df], ignore_index=True)
print(f"历史新闻数量（train+val）: {len(history_news)}")
print(f"测试集新闻数量: {len(test_df)}")

# 3. 为所有评论表添加真实标签 isFake（最稳的方式）
# 先删掉可能存在的旧标签列，防止列名冲突
if 'isFake' in comments_all.columns:
    comments_all = comments_all.drop(columns=['isFake'])
if 'isFake_x' in comments_all.columns:
    comments_all = comments_all.drop(columns=['isFake_x'])
if 'isFake_y' in comments_all.columns:
    comments_all = comments_all.drop(columns=['isFake_y'])

# 读取所有新闻的真实标签（从三个划分文件合并成完整标签表）
all_news = pd.concat([train_df, val_df, test_df], ignore_index=True)
comments_all = comments_all.merge(all_news, on='news_folder', how='left')

print(f"成功为所有评论打上真实标签，总评论数: {len(comments_all)}")
print(f"标签分布: 真新闻评论 {comments_all['isFake'].sum()} 条，假新闻评论 {len(comments_all) - comments_all['isFake'].sum()} 条\n")

# 4. 计算用户历史准确率（权重）
history_comments = comments_all[comments_all['news_folder'].isin(history_news['news_folder'])].copy()

# 判断每条评论是否正确
history_comments['correct'] = 0
# A + 真新闻 → 正确   (isFake==0)
history_comments.loc[(history_comments['attitude'] == 'A') & (history_comments['isFake'] == 0), 'correct'] = 1
# B + 假新闻 → 正确   (isFake==1)
history_comments.loc[(history_comments['attitude'] == 'B') & (history_comments['isFake'] == 1), 'correct'] = 1

user_stats = history_comments.groupby('user_id').agg(
    total=('correct', 'size'),
    correct=('correct', 'sum')
).reset_index()

user_stats['accuracy'] = user_stats['correct'] / user_stats['total']
user_stats['weight'] = user_stats['accuracy']

global_acc = user_stats['correct'].sum() / user_stats['total'].sum() if len(user_stats) > 0 else 0.5
print(f"历史用户数: {len(user_stats)}，全局平均准确率: {global_acc:.4f}")

user_weight_dict = dict(zip(user_stats['user_id'].astype(str), user_stats['weight']))

# 5. 测试集加权投票
test_comments = comments_all[comments_all['news_folder'].isin(test_df['news_folder'])].copy()

# 分配权重（没出现过的用历史权重，没出现过的用全局平均）
test_comments['user_id'] = test_comments['user_id'].astype(str)
test_comments['weight'] = test_comments['user_id'].map(user_weight_dict).fillna(global_acc)

# 加权得分：B 投 +w，A 投 -w
test_comments['score'] = test_comments['weight'] * (test_comments['attitude'] == 'B') * 2 - test_comments['weight']   # B→+w, A→-w

news_score = test_comments.groupby('news_folder')['score'].sum().reset_index()
news_score['pred_prob'] = news_score['score']
news_score['pred_label'] = (news_score['score'] > 0).astype(int)

# 合并真实标签
test_result = test_df.merge(news_score[['news_folder', 'pred_label', 'pred_prob']], on='news_folder', how='left')

has_pred = test_result['pred_label'].notna()
y_true = test_result.loc[has_pred, 'isFake'].values
y_pred = test_result.loc[has_pred, 'pred_label'].values
y_prob = test_result.loc[has_pred, 'pred_prob'].values

print(f"\n测试集可预测新闻数: {len(y_true)}，无评论新闻数: {len(test_result) - len(y_true)}")

if len(y_true) == 0:
    print("测试集没有带评论的新闻，无法评估！")
else:
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    auc = roc_auc_score(y_true, y_prob)

    print("\n" + "="*65)
    print("          加权投票（Weighted Voting）最终测试结果")
    print("="*65)
    print(f"Accuracy        : {acc:.4f}")
    print(f"Macro Precision : {p:.4f}")
    print(f"Macro Recall    : {r:.4f}")
    print(f"Macro F1        : {f1:.4f}")
    print(f"AUC             : {auc:.4f}   ← 核心指标！")
    print("="*65)
    print(classification_report(y_true, y_pred, target_names=['Real', 'Fake'], digits=4))

    test_result.to_csv('splits/test_weighted_voting_final.csv', index=False, encoding='utf-8-sig')
    print("\n结果已保存：splits/test_weighted_voting_final.csv")