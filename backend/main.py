import os, shutil, json, asyncio, re
from typing import List, Dict
from fastapi import FastAPI, UploadFile, File, Query, Request
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
templates = Jinja2Templates(directory="frontend")


# Serve index.html at "/"
@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "/tmp/uploaded_files"
CACHE_DIR = "/tmp/cache"
STATUS: Dict[str, Dict] = {}

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "Helsinki-NLP/opus-mt-zh-en"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name).to(device)


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


async def translate_paragraph(paragraph: str) -> str:
    inputs = tokenizer(
        paragraph, return_tensors="pt", truncation=True, padding=True
    ).to(device)
    outputs = model.generate(**inputs)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]


async def translate_page(page: List[str]) -> List[str]:
    return await asyncio.gather(*[translate_paragraph(p) for p in page])


def get_cache_path(filename: str, page: int) -> str:
    return os.path.join(CACHE_DIR, f"{filename}_page_{page}.json")


def save_cache(filename: str, page: int, data):
    with open(get_cache_path(filename, page), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def load_cache(filename: str, page: int):
    path = get_cache_path(filename, page)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


@app.post("/upload/")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        path = os.path.join(UPLOAD_DIR, file.filename)
        with open(path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        pages = extract_pages_from_pdf(path)
        STATUS[file.filename] = {"total": len(pages), "done": 0, "started": False}
        return {"filename": file.filename, "total_pages": len(pages)}
    except Exception as e:
        print(f"Upload failed: {e}")
        return {"error": str(e)}


@app.post("/translate_all/")
async def translate_all(filename: str = Query(...)):
    if STATUS.get(filename, {}).get("started"):
        return {"status": "already_running"}

    STATUS[filename]["started"] = True
    pages = extract_pages_from_pdf(os.path.join(UPLOAD_DIR, filename))

    async def process():
        for i, page in enumerate(pages):
            if load_cache(filename, i):
                STATUS[filename]["done"] += 1
                continue
            translated = await translate_page(page)
            save_cache(filename, i, {"original": page, "translation": translated})
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
async def get_page(filename: str = Query(...), page: int = Query(...)):
    path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(path):
        return {"error": "file_not_found"}

    cached = load_cache(filename, page)
    if cached:
        return cached

    pages = extract_pages_from_pdf(path)
    if page >= len(pages):
        return {"error": "page_out_of_range"}
    original = pages[page]
    translation = await translate_page(original)
    data = {"original": original, "translation": translation}
    save_cache(filename, page, data)
    if STATUS.get(filename):
        STATUS[filename]["done"] += 1
    return data
