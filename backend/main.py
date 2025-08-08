import os, shutil, json, asyncio, re
from typing import List, Dict, Tuple
from fastapi import FastAPI, UploadFile, File, Query, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from fastapi.middleware.cors import CORSMiddleware
from transformers import MarianMTModel, MarianTokenizer
import pdfplumber
import torch

app = FastAPI()

# Mount static files (CSS, JS, etc.)
# app.mount("/static", StaticFiles(directory="backend/static"), name="static")

# Set up Jinja2 templates
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 项目根目录
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "frontend"))
#templates = Jinja2Templates(directory="frontend")

UPLOAD_DIR = "uploaded_files"
CACHE_DIR = "cache"
STATUS: Dict[str, Dict] = {}

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

LANG_MODEL_MAP: Dict[Tuple[str, str], str] = {
    ("zh", "en"): "Helsinki-NLP/opus-mt-zh-en",
    ("en", "zh"): "Helsinki-NLP/opus-mt-en-zh",
    ("en", "fr"): "Helsinki-NLP/opus-mt-en-fr",
    ("fr", "en"): "Helsinki-NLP/opus-mt-fr-en",
    # 你可以根据需要加更多语言对
}

MODEL_CACHE: Dict[Tuple[str, str], Tuple[MarianTokenizer, MarianMTModel]] = {}

def get_model_tokenizer(src_lang: str, tgt_lang: str):
    key = (src_lang, tgt_lang)
    if key in MODEL_CACHE:
        return MODEL_CACHE[key]
    model_name = LANG_MODEL_MAP.get(key)
    if not model_name:
        raise ValueError(f"Unsupported language pair: {src_lang} → {tgt_lang}")
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name).to(device)
    MODEL_CACHE[key] = (tokenizer, model)
    return tokenizer, model

def extract_paragraphs_from_text(text: str) -> List[str]:
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    blocks, buffer = [], []
    for line in lines:
        if not line:
            if buffer:
                blocks.append(" ".join(buffer))
                buffer = []
        else:
            buffer.append(line)
    if buffer:
        blocks.append(" ".join(buffer))

    paragraphs = []
    for block in blocks:
        sentences = re.split(r"(?<=[。！？!?])", block)
        para = ""
        for sentence in sentences:
            para += sentence.strip()
            if re.search(r"[。！？!?]$", sentence):
                paragraphs.append(para.strip())
                para = ""
        if para:
            paragraphs.append(para.strip())
    return paragraphs

def extract_pages_from_pdf(file_path: str) -> List[List[str]]:
    pages = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            paragraphs = extract_paragraphs_from_text(text or "")
            pages.append(paragraphs)
    return pages

async def translate_paragraph(paragraph: str, tokenizer: MarianTokenizer, model: MarianMTModel) -> str:
    inputs = tokenizer(paragraph, return_tensors="pt", truncation=True, padding=True).to(device)
    outputs = model.generate(**inputs)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

async def translate_page(page: List[str], tokenizer: MarianTokenizer, model: MarianMTModel) -> List[str]:
    return await asyncio.gather(*[translate_paragraph(p, tokenizer, model) for p in page])

def get_cache_path(filename: str, page: int, src_lang: str, tgt_lang: str) -> str:
    return os.path.join(CACHE_DIR, f"{filename}_page_{page}_{src_lang}_{tgt_lang}.json")

def save_cache(filename: str, page: int, src_lang: str, tgt_lang: str, data):
    with open(get_cache_path(filename, page, src_lang, tgt_lang), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

def load_cache(filename: str, page: int, src_lang: str, tgt_lang: str):
    path = get_cache_path(filename, page, src_lang, tgt_lang)
    if os.path.exists(path):
        print(f"[CACHE HIT] Loading cache from {path}")
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    print(f"[CACHE MISS] No cache file at {path}")
    return None


@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# 文本输入页面
@app.get("/text_input", response_class=HTMLResponse)
async def text_input_page(request: Request):
    return templates.TemplateResponse("text_input.html", {"request": request})

@app.get("/pdf_upload", response_class=HTMLResponse)
async def pdf_upload_page(request: Request):
    return templates.TemplateResponse("pdf_upload.html", {"request": request})


@app.get("/library_page", response_class=HTMLResponse)
async def library_page(request: Request):
    return templates.TemplateResponse("library.html", {"request": request})

@app.get("/library/")
def list_library():
    # 这里返回缓存目录里所有上传过的文件及页数
    books = []
    for filename in os.listdir(UPLOAD_DIR):
        if not filename.lower().endswith(".pdf"):
            continue
        # 简单统计页数：缓存里有多少页
        pages = len([f for f in os.listdir(CACHE_DIR) if f.startswith(filename)])
        books.append({
            "filename": filename,
            "pages_cached": pages,
        })
    return {"books": books}

@app.get("/reader", response_class=HTMLResponse)
async def reader_page(request: Request, filename: str = Query(...)):
    return templates.TemplateResponse("reader.html", {"request": request, "filename": filename})




# --- 新增文本翻译API ---

@app.post("/translate_text/")
async def translate_text(
    text: str = Form(...),
    src_lang: str = Form("zh"),
    tgt_lang: str = Form("en"),
):
    try:
        tokenizer, model = get_model_tokenizer(src_lang, tgt_lang)
    except ValueError as e:
        return {"error": str(e)}

    paragraphs = extract_paragraphs_from_text(text)
    translations = await translate_page(paragraphs, tokenizer, model)
    return {"original": paragraphs, "translation": translations}

@app.post("/upload/")
async def upload_pdf(file: UploadFile = File(...)):
    path = os.path.join(UPLOAD_DIR, file.filename)
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    pages = extract_pages_from_pdf(path)
    STATUS[file.filename] = {
        "total": len(pages), "done": 0, "started": False
    }
    return {"filename": file.filename, "total_pages": len(pages)}

@app.post("/translate_all/")
async def translate_all(
    filename: str = Query(...),
    src_lang: str = Query("zh"),
    tgt_lang: str = Query("en"),
):
    if STATUS.get(filename, {}).get("started"):
        return {"status": "already_running"}

    try:
        tokenizer, model = get_model_tokenizer(src_lang, tgt_lang)
    except ValueError as e:
        return {"error": str(e)}

    STATUS[filename]["started"] = True
    pages = extract_pages_from_pdf(os.path.join(UPLOAD_DIR, filename))

    async def process():
        for i, page in enumerate(pages):
            if load_cache(filename, i, src_lang, tgt_lang):
                STATUS[filename]["done"] += 1
                continue
            translated = await translate_page(page, tokenizer, model)
            save_cache(filename, i, src_lang, tgt_lang, {"original": page, "translation": translated})
            STATUS[filename]["done"] += 1

    asyncio.create_task(process())
    return {"status": "started"}

@app.get("/translation_status/")
def translation_status(filename: str = Query(...)):
    status = STATUS.get(filename)
    if not status:
        return {"error": "not_found"}
    return status

@app.get("/get_page/")
async def get_page(
    filename: str = Query(...),
    page: int = Query(...),
    src_lang: str = Query("zh"),
    tgt_lang: str = Query("en"),
):
    path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(path):
        return {"error": "file_not_found"}

    cached = load_cache(filename, page, src_lang, tgt_lang)
    pages = extract_pages_from_pdf(path)
    total_pages = len(pages)

    if cached:
        cached["total_pages"] = total_pages
        return cached

    if page >= total_pages:
        return {"error": "page_out_of_range"}

    try:
        tokenizer, model = get_model_tokenizer(src_lang, tgt_lang)
    except ValueError as e:
        return {"error": str(e)}

    original = pages[page]
    translation = await translate_page(original, tokenizer, model)
    data = {"original": original, "translation": translation, "total_pages": total_pages}
    save_cache(filename, page, src_lang, tgt_lang, data)
    if STATUS.get(filename):
        STATUS[filename]["done"] += 1
    return data
