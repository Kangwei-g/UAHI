# 文件名：svm_word2vec_forward_baseline.py
# 复现经典论文中的 SVM + Word2Vec + 传播特征 baseline

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from gensim.models import Word2Vec
import jieba
from tqdm import tqdm
import os

# ==================== 1. 终极防丢标签加载 ====================
print("正在加载数据（永不丢失 isFake）...")
news_basic = pd.read_csv('news_basic.csv')
comments_df = pd.read_csv('news_with_comments_fixed.csv.csv', dtype={'user_id': str})

# 强制创建标签映射
folder_to_label = dict(zip(news_basic['news_folder'], news_basic['isFake']))

# 划分
train_folders = set(pd.read_csv('splits/news_basic_train.csv')['news_folder'])
val_folders   = set(pd.read_csv('splits/news_basic_val.csv')['news_folder'])
test_folders  = set(pd.read_csv('splits/news_basic_test.csv')['news_folder'])

train_val_folders = train_folders | val_folders

# 合并文本和标签
df = news_basic[['news_folder', 'news_text_raw']].copy()
df['isFake'] = df['news_folder'].map(folder_to_label)

# 划分集合
train_df = df[df['news_folder'].isin(train_folders)].copy()
val_df   = df[df['news_folder'].isin(val_folders)].copy()
test_df  = df[df['news_folder'].isin(test_folders)].copy()

print(f"训练集: {len(train_df)}，验证集: {len(val_df)}，测试集: {len(test_df)}")

# ==================== 2. 计算传播特征：转发/评论比例 ====================
print("正在计算传播特征（评论比例）...")

# 统计每条新闻的评论数
comment_counts = comments_df['news_folder'].value_counts().to_dict()

# 评论比例 = 评论数 / 总新闻数（归一化）
total_news = len(df)
df['comment_ratio'] = df['news_folder'].map(comment_counts).fillna(0) / total_news

# 添加到各集合
train_df['comment_ratio'] = train_df['news_folder'].map(df.set_index('news_folder')['comment_ratio'])
val_df['comment_ratio']   = val_df['news_folder'].map(df.set_index('news_folder')['comment_ratio'])
test_df['comment_ratio']  = test_df['news_folder'].map(df.set_index('news_folder')['comment_ratio'])

# ==================== 3. 训练 Word2Vec ====================
print("正在训练 Word2Vec...")
sentences = [jieba.lcut(text) for text in df['news_text_raw'].astype(str)]
w2v_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4, epochs=10)

# 文本转向量（平均词向量）
def text_to_vec(text, model):
    words = jieba.lcut(text)
    vecs = [model.wv[w] for w in words if w in model.wv]
    return np.mean(vecs, axis=0) if vecs else np.zeros(100)

print("正在生成 Word2Vec 文本向量...")
train_vec = np.array([text_to_vec(t, w2v_model) for t in tqdm(train_df['news_text_raw'], desc="Train")])
val_vec   = np.array([text_to_vec(t, w2v_model) for t in tqdm(val_df['news_text_raw'],   desc="Val")])
test_vec  = np.array([text_to_vec(t, w2v_model) for t in tqdm(test_df['news_text_raw'],  desc="Test")])

# ==================== 4. 拼接特征：Word2Vec + 评论比例 ====================
X_train = np.hstack([train_vec, train_df[['comment_ratio']].values])
X_val   = np.hstack([val_vec,   val_df[['comment_ratio']].values])
X_test  = np.hstack([test_vec,  test_df[['comment_ratio']].values])

y_train = train_df['isFake'].values
y_val   = val_df['isFake'].values
y_test  = test_df['isFake'].values

# ==================== 5. 训练 SVM（带概率输出）===================
print("正在训练 SVM...")
svm = SVC(kernel='rbf', C=10, gamma='scale', probability=True, class_weight='balanced')
svm.fit(np.vstack([X_train, X_val]), np.concatenate([y_train, y_val]))  # 用 train+val 训练

# ==================== 6. 测试集预测与评估 ====================
y_pred = svm.predict(X_test)
y_prob = svm.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')
auc = roc_auc_score(y_test, y_prob)

print("\n" + "="*70)
print("           SVM + Word2Vec + 评论比例 Baseline")
print("="*70)
print(f"Accuracy        : {acc:.4f}")
print(f"Macro Precision : {p:.4f}")
print(f"Macro Recall    : {r:.4f}")
print(f"Macro F1        : {f1:.4f}")
print(f"AUC             : {auc:.4f}")
print("="*70)

# 保存结果
result = test_df[['news_folder', 'isFake']].copy().reset_index(drop=True)
result['svm_prob'] = y_prob
result['svm_pred'] = y_pred
result['correct']  = (y_pred == y_test).astype(int)
result.to_csv('splits/SVM_WORD2VEC_BASELINE_RESULT.csv', index=False, encoding='utf-8-sig')
print("SVM 结果已保存 → splits/SVM_WORD2VEC_BASELINE_RESULT.csv")