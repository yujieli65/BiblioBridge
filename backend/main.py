import os
import shutil
import json
import asyncio
import re
from typing import List, Dict

import pdfplumber
from fastapi import FastAPI, UploadFile, File, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from transformers import MarianMTModel, MarianTokenizer

app = FastAPI()

# CORS 设置（可更改为你的前端地址）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploaded_books"
CACHE_DIR = "cache"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# 翻译模型（中译英）
model_name = "Helsinki-NLP/opus-mt-zh-en"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# ========================
# ✅ 实用函数
# ========================

def extract_paragraphs(text: str) -> List[str]:
    """智能段落提取：按中文标点拆分"""
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    buffer = []
    for line in lines:
        if not buffer or not re.search(r"[。！？!?]$", buffer[-1]):
            buffer.append(line)
        else:
            buffer.append("§")  # 用特殊分隔符标识新段
            buffer.append(line)
    raw = " ".join(buffer)
    parts = raw.split("§")
    result = []
    for part in parts:
        sentences = re.split(r"(?<=[。！？!?])", part)
        para = ""
        for sentence in sentences:
            para += sentence.strip()
            if re.search(r"[。！？!?]$", sentence):
                result.append(para.strip())
                para = ""
        if para:
            result.append(para.strip())
    return [p for p in result if p]

def extract_pages_from_pdf(path: str) -> List[List[str]]:
    """提取 PDF 每页的段落"""
    pages = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                paras = extract_paragraphs(text)
                pages.append(paras)
            else:
                pages.append([])
    return pages

async def translate_paragraph(paragraph: str) -> str:
    inputs = tokenizer(paragraph, return_tensors="pt", truncation=True, padding=True)
    outputs = model.generate(**inputs)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

async def translate_page(paragraphs: List[str]) -> List[str]:
    tasks = [translate_paragraph(p) for p in paragraphs]
    return await asyncio.gather(*tasks)

def get_pdf_path(book: str) -> str:
    return os.path.join(UPLOAD_DIR, book)

def get_cache_path(book: str, page: int) -> str:
    return os.path.join(CACHE_DIR, f"{book}_page_{page}.json")

def load_page_cache(book: str, page: int) -> Dict:
    path = get_cache_path(book, page)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_page_cache(book: str, page: int, data: Dict):
    with open(get_cache_path(book, page), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def get_status_path(book: str) -> str:
    return os.path.join(CACHE_DIR, f"{book}_status.json")

def update_translation_status(book: str, total_pages: int, done: int):
    path = get_status_path(book)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"total": total_pages, "done": done}, f)

# ========================
# ✅ 接口
# ========================

@app.post("/upload/")
async def upload_pdf(file: UploadFile = File(...)):
    path = get_pdf_path(file.filename)
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    pages = extract_pages_from_pdf(path)
    total_pages = len(pages)

    # 缓存页码结构，供 get_page 使用
    for i, para_list in enumerate(pages[:3]):  # 预翻译前3页
        if not load_page_cache(file.filename, i):
            translated = await translate_page(para_list)
            save_page_cache(file.filename, i, {
                "original": para_list,
                "translation": translated
            })

    update_translation_status(file.filename, total_pages, min(3, total_pages))
    return {"filename": file.filename, "total_pages": total_pages}

@app.get("/get_page/")
async def get_page(filename: str = Query(...), page: int = Query(...)):
    path = get_pdf_path(filename)
    if not os.path.exists(path):
        return {"error": "文件不存在"}

    cached = load_page_cache(filename, page)
    if cached:
        return cached

    all_pages = extract_pages_from_pdf(path)
    if page >= len(all_pages):
        return {"error": "页码超出范围"}

    original = all_pages[page]
    translation = await translate_page(original)
    data = {"original": original, "translation": translation}
    save_page_cache(filename, page, data)

    status = load_translation_status(filename)
    done_pages = status.get("done", 0)
    update_translation_status(filename, len(all_pages), max(done_pages, page + 1))

    return data

@app.get("/list_books/")
def list_books():
    return [f for f in os.listdir(UPLOAD_DIR) if f.lower().endswith(".pdf")]

@app.post("/translate_all/")
async def translate_all(book: str):
    path = get_pdf_path(book)
    if not os.path.exists(path):
        return {"error": "文件不存在"}

    all_pages = extract_pages_from_pdf(path)
    total_pages = len(all_pages)
    done_pages = 0

    for i, para_list in enumerate(all_pages):
        if not load_page_cache(book, i):
            translated = await translate_page(para_list)
            save_page_cache(book, i, {
                "original": para_list,
                "translation": translated
            })
        done_pages += 1
        update_translation_status(book, total_pages, done_pages)

    return {"message": f"{book} 翻译完成", "pages": total_pages}

@app.get("/translation_status/")
def translation_status(book: str):
    path = get_status_path(book)
    if not os.path.exists(path):
        return {"error": "尚未开始翻译"}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_translation_status(book: str):
    path = get_status_path(book)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"done": 0, "total": 0}
