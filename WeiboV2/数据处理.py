# 文件名：add_comment_text_to_csv.py
# 功能：从 c-0-c-picture删去4个-5-200.json 文件中提取 comment_text_raw，补全到 news_with_comments.csv

import pandas as pd
import json
import os
from tqdm import tqdm

# ==================== 配置 ====================
json_folder = "."  # json 文件所在目录（当前目录）
json_pattern = "/Users/Zhuanz1/Downloads/zhoulujuan/PythonProject5/c-0-c-picture删去4个-5-200.json"  # 你的 json 文件名
csv_input = "news_with_comments.csv"
csv_output = "news_with_comments_fixed.csv"

# ==================== 1. 读取所有 json 文件，构建 (news_folder, user_id) → comment_text_raw 映射 ====================
print(f"正在从 {json_pattern} 加载评论文本...")
comment_dict = {}  # (news_folder, user_id) → comment_text_raw

with open(json_pattern, 'r', encoding='utf-8') as f:
    data = json.load(f)  # 整个文件是一个大列表

    for item in tqdm(data, desc="解析 JSON"):
        news_folder = item.get("news_folder")
        if not news_folder:
            continue
        for comment in item.get("comments", []):
            user_id = comment.get("user_id")
            text = comment.get("comment_text_raw", "")
            if user_id is not None:
                key = (news_folder, str(user_id))  # user_id 转字符串统一
                comment_dict[key] = text

print(f"共加载 {len(comment_dict)} 条评论文本")

# ==================== 2. 读取原始 csv 并补全 comment_text_raw ====================
print(f"正在加载并补全 {csv_input} ...")
df = pd.read_csv(csv_input, dtype={'user_id': str})

# 确保 user_id 是字符串
df['user_id'] = df['user_id'].astype(str)

# 创建映射键
df['key'] = list(zip(df['news_folder'], df['user_id']))

# 映射 comment_text_raw
df['comment_text_raw'] = df['key'].map(comment_dict)

# 删除辅助列
df = df.drop(columns=['key'])

# 检查缺失情况
missing = df['comment_text_raw'].isna().sum()
print(f"补全完成！共 {len(df)} 条记录，其中 {missing} 条未找到对应文本（已填 NaN）")

# ==================== 3. 保存新文件 ====================
df.to_csv(csv_output, index=False, encoding='utf-8-sig')
print(f"已保存完整版文件 → {csv_output}")

# 顺便打印前几行看看效果
print("\n前10条示例：")
print(df[['news_folder', 'user_id', 'attitude', 'comment_text_raw']].head(10))