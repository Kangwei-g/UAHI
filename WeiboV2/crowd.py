# 文件名：rahi_crowd_module_fixed.py
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import defaultdict
import os

# ==================== 1. 加载数据 ====================
print("正在加载评论数据...")
comments_df = pd.read_csv('news_with_comments.csv', dtype={'user_id': str})
train_folders = pd.read_csv('splits/news_basic_train.csv')['news_folder'].tolist()
val_folders   = pd.read_csv('splits/news_basic_val.csv')['news_folder'].tolist()
test_folders  = pd.read_csv('splits/news_basic_test.csv')['news_folder'].tolist()

train_val_folders = train_folders + val_folders

# 真实标签映射
news_basic = pd.read_csv('news_basic.csv')
folder_to_label = dict(zip(news_basic['news_folder'], news_basic['isFake']))

comments_df['true_label'] = comments_df['news_folder'].map(folder_to_label)
comments_df['user_vote'] = comments_df['attitude'].map({'A': 0, 'B': 1})
comments_df['correct'] = (comments_df['user_vote'] == comments_df['true_label'])

# ==================== 2. 计算新闻难度 diff ====================
news_stats = comments_df.groupby('news_folder').agg(
    total_users=('user_id', 'nunique'),
    correct_users=('correct', 'sum')
).reset_index()

news_stats['diff'] = news_stats['correct_users'] / news_stats['total_users']
news_stats['diff'] = news_stats['diff'].replace(0, 1e-8)  # 防止除0

# ==================== 3. 初始化用户可靠性 c_a（只用答对的）===================
correct_comments = comments_df[comments_df['correct']].copy()
correct_comments = correct_comments.merge(news_stats[['news_folder', 'diff']], on='news_folder')

user_ca_init = correct_comments.groupby('user_id').agg(
    sum_inv_diff=('diff', lambda x: (1 / x).sum()),
    correct_count=('correct', 'count')
).reset_index()

user_ca_init['c_a_init'] = user_ca_init['sum_inv_diff'] / user_ca_init['correct_count']
global_avg_ca = user_ca_init['c_a_init'].mean()
print(f"全局平均初始 c_a: {global_avg_ca:.4f}")

user_to_ca_init = dict(zip(user_ca_init['user_id'], user_ca_init['c_a_init']))

# ==================== 4. 可学习 c_a 参数（关键修复！）===================
train_comments = comments_df[comments_df['news_folder'].isin(train_val_folders)].copy()
unique_users = train_comments['user_id'].unique()
user_to_idx = {uid: i for i, uid in enumerate(unique_users)}
n_users = len(unique_users)

# 初始化为可学习参数（必须是 Parameter！）
init_values = torch.tensor(
    [user_to_ca_init.get(uid, global_avg_ca) for uid in unique_users],
    dtype=torch.float32
)
ca_params = torch.nn.Parameter(torch.clamp(init_values, min=1e-6))  # 必须是 Parameter

optimizer = optim.SGD([ca_params], lr=0.05, momentum=0.9)
criterion = nn.BCELoss()

print(f"开始微调 {n_users} 个用户的 c_a 参数...")

best_loss = float('inf')
best_ca = ca_params.clone().detach()

for epoch in range(1, 31):
    optimizer.zero_grad()
    total_loss = 0.0

    for news_id, group in train_comments.groupby('news_folder'):
        true_label = group['true_label'].iloc[0].item()
        target = torch.tensor(1.0 if true_label == 1 else 0.0, dtype=torch.float32)

        # 收集支持假新闻和真新闻的用户
        fake_users = group[group['user_vote'] == 1]['user_id'].tolist()
        true_users = group[group['user_vote'] == 0]['user_id'].tolist()

        weight_fake = 0.0
        weight_true = 0.0

        # 必须通过索引从 ca_params 中取值，保证可导！
        for uid in fake_users:
            if uid in user_to_idx:
                weight_fake += ca_params[user_to_idx[uid]]
        for uid in true_users:
            if uid in user_to_idx:
                weight_true += ca_params[user_to_idx[uid]]

        total_weight = weight_fake + weight_true + 1e-8
        crowd_prob_fake = weight_fake / total_weight  # 这是一个可导的 tensor！

        loss = criterion(crowd_prob_fake, target)
        total_loss += loss.item()
        loss.backward()  # 现在可以反向传播了！

    optimizer.step()

    # 保持正数
    with torch.no_grad():
        ca_params.clamp_(min=1e-6)

    avg_loss = total_loss / len(train_comments['news_folder'].unique())
    print(f"Epoch {epoch:02d} | Loss: {avg_loss:.6f}")

    if avg_loss < best_loss:
        best_loss = avg_loss
        best_ca = ca_params.clone().detach()
        print("  → 新最佳 c_a")

print(f"\n人群智能训练完成！最佳 Loss: {best_loss:.6f}")

# 保存
learned_ca_dict = {uid: best_ca[i].item() for i, uid in enumerate(unique_users)}
torch.save({
    'user_to_idx': user_to_idx,
    'learned_ca': best_ca,
    'learned_ca_dict': learned_ca_dict,
    'global_avg_ca': global_avg_ca
}, 'rahi_crowd_learned_ca.pth')
print("已保存 → rahi_crowd_learned_ca.pth")

# ==================== 5. 测试集预测 ====================
print("\n正在生成测试集 Crowd 预测...")
test_results = []

for news_id in test_folders:
    group = comments_df[comments_df['news_folder'] == news_id]
    if len(group) == 0:
        test_results.append({
            'news_folder': news_id,
            'crowd_prob_fake': 0.5,
            'crowd_alpha': 1.0,
            'crowd_beta': 1.0
        })
        continue

    fake_users = group[group['user_vote'] == 1]['user_id'].tolist()
    true_users = group[group['user_vote'] == 0]['user_id'].tolist()

    w_fake = sum(learned_ca_dict.get(u, global_avg_ca) for u in fake_users)
    w_true = sum(learned_ca_dict.get(u, global_avg_ca) for u in true_users)

    total_w = w_fake + w_true + 1e-8
    prob_fake = w_fake / total_w

    alpha = 1.0 + w_fake
    beta  = 1.0 + w_true

    test_results.append({
        'news_folder': news_id,
        'crowd_prob_fake': prob_fake,
        'crowd_alpha': alpha,
        'crowd_beta': beta
    })

crowd_df = pd.DataFrame(test_results)
crowd_df.to_csv('splits/test_crowd_prediction_rahi.csv', index=False, encoding='utf-8-sig')
print("测试集 Crowd 预测已保存 → splits/test_crowd_prediction_rahi.csv")