import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import os

# ==================== 1. 加载数据 ====================
comments_df = pd.read_csv('news_with_comments.csv', dtype={'user_id': str})
test_df = pd.read_csv('splits/news_basic_test.csv')

print(f"测试集新闻数量: {len(test_df)}")
print(f"总评论条数: {len(comments_df)}")

# ==================== 2. 只取测试集评论 ====================
test_comments = comments_df[comments_df['news_folder'].isin(test_df['news_folder'])].copy()
print(f"测试集中有评论的新闻数量: {test_comments['news_folder'].nunique()}")

# ==================== 3. Majority Voting ====================
def majority_voting(group):
    count_A = (group['attitude'] == 'A').sum()   # 相信是真的 → 预测 0
    count_B = (group['attitude'] == 'B').sum()   # 认为是假的 → 预测 1
    # 严格多数：B 必须严格大于 A 才判为假新闻，否则判真（包括平局）
    return 1 if count_B > count_A else 0

# 每条新闻的投票结果
pred_results = test_comments.groupby('news_folder').apply(majority_voting).reset_index()
pred_results.columns = ['news_folder', 'pred']

# ==================== 4. 合并真实标签 ====================
test_merged = test_df[['news_folder', 'isFake']].merge(
    pred_results, on='news_folder', how='left'
)

# 有预测的样本
has_pred = test_merged['pred'].notna()
y_true = test_merged.loc[has_pred, 'isFake'].astype(int).values
y_pred = test_merged.loc[has_pred, 'pred'].astype(int).values

print(f"\n能参与投票的新闻数: {len(y_true)}")
print(f"无评论无法投票的新闻数: {len(test_merged) - len(y_true)}")

if len(y_true) == 0:
    print("测试集中没有带评论的新闻，无法进行投票评估！")
else:
    # ==================== 5. 计算指标（兼容旧版 sklearn）===================
    accuracy = accuracy_score(y_true, y_pred)
    # precision_recall_fscore_support 是最兼容的写法
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')

    print("\n" + "="*60)
    print("          Majority Voting 测试集结果（Macro）")
    print("="*60)
    print(f"Accuracy           : {accuracy:.4f}")
    print(f"Macro Precision    : {precision:.4f}")
    print(f"Macro Recall       : {recall:.4f}")
    print(f"Macro F1           : {f1:.4f}")
    print("="*60)
    print("\n详细分类报告：")
    print(classification_report(y_true, y_pred, target_names=['Real (0)', 'Fake (1)'], digits=4, zero_division=0))

    # 保存预测结果
    test_merged.to_csv('splits/test_majority_voting_results.csv', index=False, encoding='utf-8-sig')
    print("\n预测结果已保存至 splits/test_majority_voting_results.csv")