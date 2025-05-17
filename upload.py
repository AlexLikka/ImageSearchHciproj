# -*- coding: utf-8 -*-
"""
上传 GroceryStoreDataset 图像特征向量到 Upstash Vector

运行前请确保：
1. 安装依赖：pip install torch numpy pillow requests clip-by-openai upstash-vector
2. 设置以下环境变量（可通过输入命令提供）：
   - UPSTASH_VECTOR_REST_URL
   - UPSTASH_VECTOR_REST_TOKEN
"""

import os
import json
import torch
from PIL import Image
import numpy as np
import clip
from upstash_vector import Index, Vector

# ------------------ Step 1: 加载模型 ------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# ------------------ Step 2: Upstash 凭据 ------------------
UPSTASH_VECTOR_URL = "https://evolved-parakeet-76384-us1-vector.upstash.io"
UPSTASH_VECTOR_TOKEN = "ABoFMGV2b2x2ZWQtcGFyYWtlZXQtNzYzODQtdXMxYWRtaW5OelZqWW1GaU1Ua3Raak14WVMwME0ySXpMVGhqTURjdFl6Vm1aVGMxWXpsbU5tSTQ="


# ------------------ Step 3: 提取图片路径 ------------------
def load_image_paths(txt_files, base_dir=""):
    image_paths = []
    for file in txt_files:
        with open(file, 'r') as f:
            for line in f:
                parts = line.strip().split(",")  # 这里按逗号分割，避免路径被截断
                if len(parts) >= 1:
                    img_rel_path = parts[0].strip()
                    full_path = os.path.join(base_dir, img_rel_path)  # 拼接完整路径
                    image_paths.append(full_path)
    return image_paths


# ------------------ Step 4: 特征提取 ------------------
def embed_image(img: Image.Image) -> np.ndarray:
    img_input = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.encode_image(img_input)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy().flatten()


# ------------------ Step 5: 上传向量 ------------------
def upload_vector(index, path: str, vector: np.ndarray):
    metadata = {
        "path": path
    }
    vector_data = Vector(
        id=path,
        vector=vector.tolist(),
        metadata=metadata
    )
    index.upsert(vectors=[vector_data])
    print(f"[成功] 上传 {path}")


# ------------------ Step 6: 批量上传 ------------------
def upload_all(image_paths, index):
    for path in image_paths:
        try:
            img = Image.open(path).convert("RGB")
            vec = embed_image(img)
            upload_vector(index, path, vec)
        except Exception as e:
            print(f"[跳过] {path}：{e}")


# ------------------ 主程序入口 ------------------
if __name__ == "__main__":
    # 创建Upstash向量索引
    index = Index(url=UPSTASH_VECTOR_URL, token=UPSTASH_VECTOR_TOKEN)

    # 加载图片路径
    txts = [
        "train.txt",
        "val.txt",
        "test.txt"
    ]
    paths = load_image_paths(txts)
    print(f"共加载 {len(paths)} 张图片，开始上传...")

    # 批量上传图片特征向量
    upload_all(paths, index)

    print("全部上传完成。")
