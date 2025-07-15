import os
import shutil
import json
import re
from typing import List

import pdfplumber
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from transformers import MarianMTModel, MarianTokenizer

# FastAPI 应用初始化
app = FastAPI()

# 允许跨域（建议上线时替换为具体前端域名）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 路径常量
UPLOAD_DIR = "uploaded_files"
CACHE_DIR = "cache"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# 加载 MarianMT 模型（中文 → 英文）
model_name = "Helsinki-NLP/opus-mt-zh-en"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# ✅ 智能提取段落（按中文句末标点断句）
def extract_paragraphs_from_text(text: str) -> List[str]:
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    blocks = []
    buffer = []
    for line in lines:
        if line == "":
            if buffer:
                blocks.append(" ".join(buffer))
                buffer = []
        else:
            buffer.append(line)
    if buffer:
        blocks.append(" ".join(buffer))

    final_paragraphs = []
    for block in blocks:
        sentences = re.split(r"(?<=[。！？!?])", block)
        para = ""
        for sentence in sentences:
            para += sentence.strip()
            if re.search(r"[。！？!?]$", sentence):
                final_paragraphs.append(para.strip())
                para = ""
        if para:
            final_paragraphs.append(para.strip())
    return final_paragraphs

# ✅ 从 PDF 提取每页段落
def extract_pages_from_pdf(file_path: str) -> List[List[str]]:
    pages = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                paragraphs = extract_paragraphs_from_text(text)
                pages.append(paragraphs)
            else:
                pages.append([])
    return pages

# ✅ 高效批量翻译函数（支持 GPU/CPU 批处理）
def batch_translate(texts: List[str], batch_size: int = 8) -> List[str]:
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
        outputs = model.generate(**inputs)
        translations = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        results.extend(translations)
    return results

# ✅ 生成缓存路径
def get_cache_path(filename: str, page: int) -> str:
    safe_name = filename.replace("/", "_").replace("\\", "_")
    return os.path.join(CACHE_DIR, f"{safe_name}_page_{page}.json")

# ✅ 读取缓存
def load_cache(filename: str, page: int):
    path = get_cache_path(filename, page)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

# ✅ 写入缓存
def save_cache(filename: str, page: int, data):
    path = get_cache_path(filename, page)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# ✅ 上传 PDF 接口（保存文件 + 异步预翻译前3页）
@app.post("/upload/")
async def upload_pdf(file: UploadFile = File(...)):
    temp_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    pages = extract_pages_from_pdf(temp_path)
    total_pages = len(pages)

    # 预翻译前 3 页（同步执行，为了前端首屏体验）
    for page_idx in range(min(3, total_pages)):
        if load_cache(file.filename, page_idx) is None:
            original = pages[page_idx]
            translation = batch_translate(original)
            save_cache(file.filename, page_idx, {
                "original": original,
                "translation": translation
            })

    return {"filename": file.filename, "total_pages": total_pages}

# ✅ 获取某页翻译内容（使用缓存，否则重新翻译）
@app.get("/get_page/")
async def get_page(filename: str = Query(...), page: int = Query(...)):
    path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(path):
        return {"error": "文件未找到"}

    # 优先读取缓存
    cached = load_cache(filename, page)
    if cached is not None:
        return cached

    # 无缓存则重新翻译
    pages = extract_pages_from_pdf(path)
    if page >= len(pages):
        return {"error": "页码超出范围"}

    original = pages[page]
    translation = batch_translate(original)
    result = {"original": original, "translation": translation}
    save_cache(filename, page, result)
    return result
