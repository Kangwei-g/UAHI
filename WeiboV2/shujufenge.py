import json
import pandas as pd
from sklearn.model_selection import train_test_split
import os  # 新增这行

# 1. 读取原始数据（假设文件名为 a.json，和脚本放同一目录）
with open('c-0-c-picture删去4个-5-200-1.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 2. 提取数据
news_list = []
comment_list = []

for item in data:
    folder = item['news_folder']
    text = item['news_text_raw'].replace('\n', ' ').strip()
    isFake = item['isFake']

    news_list.append({
        'news_folder': folder,
        'news_text_raw': text,
        'isFake': isFake
    })

    if 'comments' in item and item['comments']:
        for comment in item['comments']:
            comment_list.append({
                'news_folder': folder,
                'news_text_raw': text,
                'isFake': isFake,
                'user_id': comment['user_id'],
                'attitude': comment['attitude']
            })

news_df = pd.DataFrame(news_list)
comment_df = pd.DataFrame(comment_list)

# 3. 按 news_folder 排序 + 7:2:1 划分（带分层）
news_df = news_df.sort_values(by='news_folder').reset_index(drop=True)

train_df, remain_df = train_test_split(
    news_df, train_size=0.7, random_state=42, stratify=news_df['isFake'])
val_df, test_df = train_test_split(
    remain_df, train_size=0.2 / 0.3, random_state=42, stratify=remain_df['isFake'])

# 4. 创建需要的文件夹（关键修复）
os.makedirs('splits', exist_ok=True)  # 这行搞定一切

# 5. 保存划分好的文件夹 id（方便以后复现）
split_dict = {
    'train': train_df['news_folder'].tolist(),
    'val': val_df['news_folder'].tolist(),
    'test': test_df['news_folder'].tolist()
}
with open('split_folders.json', 'w', encoding='utf-8') as f:
    json.dump(split_dict, f, ensure_ascii=False, indent=2)

# 6. 输出两个主文件
news_df[['news_folder', 'news_text_raw', 'isFake']].to_csv(
    'news_basic.csv', index=False, encoding='utf-8-sig')

if not comment_df.empty:
    comment_df = comment_df.sort_values(by=['news_folder', 'user_id'])
    comment_df.to_csv('news_with_comments.csv', index=False, encoding='utf-8-sig')
else:
    pd.DataFrame(columns=['news_folder', 'news_text_raw', 'isFake', 'user_id', 'attitude']).to_csv(
        'news_with_comments.csv', index=False, encoding='utf-8-sig')

# 7. 保存三个划分的子集（方便直接加载）
train_df.to_csv('splits/news_basic_train.csv', index=False, encoding='utf-8-sig')
val_df.to_csv('splits/news_basic_val.csv', index=False, encoding='utf-8-sig')
test_df.to_csv('splits/news_basic_test.csv', index=False, encoding='utf-8-sig')

print("所有文件生成完毕！")
print(f"总新闻条数: {len(news_df)}  (假新闻 {news_df['isFake'].sum()} 条)")
print(f"训练集 {len(train_df)}  验证集 {len(val_df)}  测试集 {len(test_df)}")
print(f"评论总数: {len(comment_df)} 条")
print("生成的文件如下：")
print("   news_basic.csv")
print("   news_with_comments.csv")
print("   split_folders.json")
print("   splits/news_basic_train.csv")
print("   splits/news_basic_val.csv")
print("   splits/news_basic_test.csv")