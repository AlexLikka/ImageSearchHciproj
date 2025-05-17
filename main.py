import gradio as gr
from PIL import Image
import requests
import io
import numpy as np
import clip
import torch
from json import JSONEncoder
from upstash_vector import Index


# 初始化带重试机制的requests session
def create_session():
    session = requests.Session()
    retry = requests.adapters.HTTPAdapter(max_retries=3)
    session.mount('http://', retry)
    session.mount('https://', retry)
    return session


# 全局session对象
session = create_session()

# 初始化Index
index = Index(
    url="https://evolved-parakeet-76384-us1-vector.upstash.io",
    token="ABoFMGV2b2x2ZWQtcGFyYWtlZXQtNzYzODQtdXMxYWRtaW5OelZqWW1GaU1Ua3Raak14WVMwME0ySXpMVGhqTURjdFl6Vm1aVGMxWXpsbU5tSTQ="
)


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.float32):
            return float(obj)
        return JSONEncoder.default(self, obj)


# 初始化CLIP模型
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


def embed_image(img: Image.Image) -> np.ndarray:
    """提取图片特征向量"""
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)

    img_input = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.encode_image(img_input)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy().squeeze()


def embed_text(text: str) -> np.ndarray:
    """提取文本特征向量"""
    text_tokens = clip.tokenize([text]).to(device)
    with torch.no_grad():
        emb = model.encode_text(text_tokens)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy().squeeze()


def search_vector(query_emb: np.ndarray, top_k: int = 12):
    """向量检索"""
    try:
        vec_list = [round(float(x), 6) for x in query_emb]
        results = index.query(
            vector=vec_list,
            top_k=top_k,
            include_metadata=True
        )
        return [(r.score, r.metadata) for r in results]
    except Exception as e:
        print(f"查询失败: {str(e)}")
        return []


def text_to_image(query: str):
    """文本搜图"""
    emb = embed_text(query)
    hits = search_vector(emb)
    imgs = []
    for score, meta in hits:
        path = meta.get('path', '')
        try:
            if path.startswith(('http://', 'https://')):
                response = session.get(path, timeout=10)
                img = Image.open(io.BytesIO(response.content))
            else:
                img = Image.open(path)
            imgs.append(img.convert('RGB'))
        except Exception as e:
            print(f"加载图像失败: {path}, 错误: {str(e)}")
    return imgs if imgs else []


def image_to_image(input_img):
    """图搜图"""
    if isinstance(input_img, np.ndarray):
        input_img = Image.fromarray(input_img)

    emb = embed_image(input_img)
    hits = search_vector(emb)
    imgs = []
    for score, meta in hits:
        path = meta.get('path', '')
        try:
            if path.startswith(('http://', 'https://')):
                response = session.get(path, timeout=10)
                img = Image.open(io.BytesIO(response.content))
            else:
                img = Image.open(path)
            imgs.append(img.convert('RGB'))
        except Exception as e:
            print(f"加载图像失败: {path}, 错误: {str(e)}")
    return imgs if imgs else []


# 创建Gradio界面
with gr.Blocks() as demo:
    gr.Markdown("# 图像检索 Demo")
    with gr.Tab("文本检索"):
        txt = gr.Textbox(label="输入文字检索图像")
        btn1 = gr.Button("搜索")
        gallery1 = gr.Gallery(label="检索结果", columns=4, height="auto")
        btn1.click(fn=text_to_image, inputs=txt, outputs=gallery1)
    with gr.Tab("图像检索"):
        img_input = gr.Image(label="输入示例图像", type="pil")
        btn2 = gr.Button("搜索相似图像")
        gallery2 = gr.Gallery(label="检索结果", columns=4, height="auto")
        btn2.click(fn=image_to_image, inputs=img_input, outputs=gallery2)
    gr.Markdown("**提示：在检索结果上右键可下载图片；可将检索到的图片地址收藏以便后续使用。**")

demo.launch()