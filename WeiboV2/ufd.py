# 文件名：ufd_ultimate.py   ← 请直接覆盖你的 ufd.py
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, roc_auc_score

print("=== UFD[24] 贝叶斯用户可靠性检测模型（最终完美版）===")

# ==================== 1. 加载数据 ====================
comments = pd.read_csv('news_with_comments.csv', dtype={'user_id': str})
train_df = pd.read_csv('splits/news_basic_train.csv')[['news_folder', 'isFake']]
val_df   = pd.read_csv('splits/news_basic_val.csv')[['news_folder', 'isFake']]
test_df  = pd.read_csv('splits/news_basic_test.csv')[['news_folder', 'isFake']]

history = pd.concat([train_df, val_df])
train_comments = comments[comments['news_folder'].isin(history['news_folder'])].merge(history, on='news_folder')
test_comments  = comments[comments['news_folder'].isin(test_df['news_folder'])].merge(test_df, on='news_folder')

print(f"训练集新闻（有评论）: {train_comments['news_folder'].nunique()}")
print(f"测试集新闻（有评论） : {test_comments['news_folder'].nunique()}")

# ==================== 2. EM 算法（字典版，永不报错）===================
users = train_comments['user_id'].unique()
news_list = train_comments['news_folder'].unique()

# 初始化
sens = {u: 0.75 for u in users}
spec = {u: 0.75 for u in users}
p_fake = {n: 0.5 for n in news_list}

for it in range(1, 31):
    df = train_comments.copy()
    df['p_fake_cur'] = df['news_folder'].map(p_fake)
    df['sens_cur'] = df['user_id'].map(sens)
    df['spec_cur'] = df['user_id'].map(spec)

    # 对数似然
    df['log_fake'] = np.where(df['attitude'] == 'B',
                              np.log(df['sens_cur'] + 1e-12),
                              np.log(1 - df['sens_cur'] + 1e-12))
    df['log_real'] = np.where(df['attitude'] == 'B',
                              np.log(1 - df['spec_cur'] + 1e-12),
                              np.log(df['spec_cur'] + 1e-12))

    ll_fake = df.groupby('news_folder')['log_fake'].sum()
    ll_real = df.groupby('news_folder')['log_real'].sum()

    # 更新新闻真实性概率
    odds = np.exp(ll_fake - ll_real)
    p_fake_new = odds / (1 + odds)
    p_fake = p_fake_new.to_dict()

    # 更新用户参数
    df['p_fake_cur'] = df['news_folder'].map(p_fake)
    df['p_real_cur'] = 1 - df['p_fake_cur']

    # Sensitivity P(B|Fake)
    sens_num = df[df['attitude'] == 'B'].groupby('user_id')['p_fake_cur'].sum()
    sens_den = df.groupby('user_id')['p_fake_cur'].sum()
    for u in users:
        num = sens_num.get(u, 0)
        den = sens_den.get(u, 0)
        sens[u] = num / den if den > 0 else 0.75

    # Specificity P(A|Real)
    spec_num = df[df['attitude'] == 'A'].groupby('user_id')['p_real_cur'].sum()
    spec_den = df.groupby('user_id')['p_real_cur'].sum()
    for u in users:
        num = spec_num.get(u, 0)
        den = spec_den.get(u, 0)
        spec[u] = num / den if den > 0 else 0.75

    if it % 5 == 0:
        global_sens = df[df['attitude']=='B']['p_fake_cur'].sum() / (df['p_fake_cur'].sum() + 1e-12)
        global_spec = df[df['attitude']=='A']['p_real_cur'].sum() / (df['p_real_cur'].sum() + 1e-12)
        print(f"Iter {it:2d} | Global Sensitivity: {global_sens:.4f} | Global Specificity: {global_spec:.4f}")

print("EM 训练完成！")

# ==================== 3. 测试集预测 ====================
test_df_merged = test_comments.copy()
test_df_merged['sens'] = test_comments['user_id'].map(sens).fillna(np.mean(list(sens.values())))
test_df_merged['spec'] = test_comments['user_id'].map(spec).fillna(np.mean(list(spec.values())))

test_df_merged['llr_contrib'] = np.where(
    test_df_merged['attitude'] == 'B',
    np.log(test_df_merged['sens'] / (1 - test_df_merged['spec'] + 1e-12)),
    np.log((1 - test_df_merged['sens']) / (test_df_merged['spec'] + 1e-12))
)

news_llr = test_df_merged.groupby('news_folder')['llr_contrib'].sum().to_frame('llr')
news_llr['pred_prob'] = 1 / (1 + np.exp(-news_llr['llr']))
news_llr['pred_label'] = (news_llr['llr'] > 0).astype(int)
news_llr = news_llr.reset_index()

result = test_df.merge(news_llr[['news_folder', 'pred_label', 'pred_prob']], on='news_folder', how='left')
has_pred = result['pred_label'].notna()
y_true = result.loc[has_pred, 'isFake'].values
y_pred = result.loc[has_pred, 'pred_label'].values
y_prob = result.loc[has_pred, 'pred_prob'].values

print(f"\n可参与评估的测试新闻: {len(y_true)} 条")

acc = accuracy_score(y_true, y_pred)
p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
auc = roc_auc_score(y_true, y_prob)

print("\n" + "="*70)
print("           UFD[24] 贝叶斯模型 测试集最终结果")
print("="*70)
print(f"Accuracy        : {acc:.4f}")
print(f"Macro Precision : {p:.4f}")
print(f"Macro Recall    : {r:.4f}")
print(f"Macro F1        : {f1:.4f}")
print(f"AUC             : {auc:.4f}")
print("="*70)
print(classification_report(y_true, y_pred, target_names=['Real', 'Fake'], digits=4))

result.to_csv('splits/test_ufd_ultimate.csv', index=False, encoding='utf-8-sig')
print("\n结果已保存：splits/test_ufd_ultimate.csv")