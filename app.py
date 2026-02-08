"""
AquaHumanizer Pro - AI Text Enhancement API
Copyright (c) 2026 AquilaStudios
Licensed under the MIT License

Production-ready paraphrasing service using T5 model with structural preservation.
"""

import os
import logging
import io
import asyncio
import json
import time
import re
import math
from typing import Optional, List, Dict
from enum import Enum
from contextlib import asynccontextmanager
import torch
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel, Field, field_validator
from transformers import T5ForConditionalGeneration, T5Tokenizer
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.enums import TA_LEFT, TA_JUSTIFY
from reportlab.pdfgen import canvas
import uuid
from datetime import datetime

# --- LOGGING CONFIGURATION ---

class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured logging"""
    def format(self, record):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "component": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, 'request_id'):
            log_entry["request_id"] = record.request_id
        return json.dumps(log_entry)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("aqua_humanizer")
handler = logging.StreamHandler()
handler.setFormatter(StructuredFormatter())
logger.addHandler(handler)
logger.propagate = False

# --- CONFIGURATION CONSTANTS ---

# Model Configuration
MODEL_NAME = "Vamsi/T5_Paraphrase_Paws"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Text Processing Constraints
MIN_WORDS = 3
# T5 works best on sentence/short paragraph level. 
# We split long text into smaller logical chunks to prevent quality loss.
MAX_CHUNK_TOKENS = 64  
MAX_INPUT_LENGTH = 512

# Generation Parameters (Critical for fixing "shortening" issue)
NUM_BEAMS = 5
NO_REPEAT_NGRAM_SIZE = 3
LENGTH_PENALTY = 1.5 # Forces model to generate longer output (prevents shortening)
TEMPERATURE = 0.9    # Slight creativity to avoid robotic feel

class ParaphraseStyle(str, Enum):
    neutral = "neutral"
    formal = "formal"
    ats = "ats"
    bullets = "bullets"

# --- GLOBAL VARIABLES ---

model: Optional[T5ForConditionalGeneration] = None
tokenizer: Optional[T5Tokenizer] = None

# --- LIFESPAN MANAGER ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, clean up on shutdown"""
    global model, tokenizer
    logger.info(f"🚀 Starting up... Loading model on {DEVICE}")
    
    try:
        tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
        model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME).to(DEVICE)
        model.eval()
        logger.info("✅ Model loaded successfully")
    except Exception as e:
        logger.critical(f"❌ Failed to load model: {str(e)}")
        raise RuntimeError("Model loading failed")
    
    yield
    
    logger.info("🛑 Shutting down...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

app = FastAPI(
    title="AquaHumanizer Pro",
    version="2.1.0",
    lifespan=lifespan
)

# --- CORE LOGIC (THE FIXES) ---

def build_prompt(text: str, style: ParaphraseStyle) -> str:
    """
    Constructs the prompt based on style.
    Refined specifically for T5 Paws model capabilities.
    """
    clean_text = text.strip()
    
    if style == ParaphraseStyle.formal:
        return f"paraphrase formally: {clean_text}"
    elif style == ParaphraseStyle.ats:
        # T5 Paws isn't a Resume expert, so we ask for professional rewriting
        return f"paraphrase professionally: {clean_text}"
    elif style == ParaphraseStyle.bullets:
        return f"paraphrase concisely: {clean_text}"
    
    return f"paraphrase: {clean_text}"

def split_into_sentences(text: str) -> List[str]:
    """
    Robust sentence splitting using Regex to handle abbreviations properly.
    """
    # Split by periods, questions, exclamations followed by space and uppercase/end
    # This prevents splitting "Mr. Smith" or "e.g." incorrectly
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text)
    return [s.strip() for s in sentences if s.strip()]

def get_dynamic_max_length(input_len: int) -> int:
    """
    Calculates max_length dynamically.
    If input is 20 tokens, we shouldn't limit output to 20, nor fix it at 128.
    """
    # Allow expansion up to 2x or at least 60 tokens for short sentences
    return max(60, int(input_len * 2))

async def process_sentence(sentence: str, style: ParaphraseStyle) -> str:
    """
    Process a single semantic unit (sentence) with optimized generation params.
    """
    if not sentence or len(sentence.split()) < 2:
        return sentence

    prompt = build_prompt(sentence, style)
    
    input_ids = tokenizer.encode(
        prompt, 
        return_tensors="pt", 
        max_length=MAX_INPUT_LENGTH, 
        truncation=True
    ).to(DEVICE)

    # Dynamic length calculation to prevent truncation
    input_length = len(input_ids[0])
    dynamic_max = get_dynamic_max_length(input_length)

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_length=dynamic_max,
            num_beams=NUM_BEAMS,
            no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
            length_penalty=LENGTH_PENALTY, # Critical fix for "text getting short"
            early_stopping=True
        )

    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Post-processing fallback: if model outputs garbage or too short, keep original
    if len(output_text) < len(sentence) * 0.4: 
        return sentence
        
    return output_text

async def process_structured_text(text: str, style: ParaphraseStyle) -> str:
    """
    THE MASTER FIX: Preserves formatting (paragraphs, lists).
    1. Splits by Paragraphs (Newlines).
    2. Identifies Bullets.
    3. Splits Paragraphs into Sentences.
    4. Reconstructs exactly.
    """
    # 1. Split into paragraphs to preserve structure
    paragraphs = text.split('\n')
    processed_paragraphs = []

    for para in paragraphs:
        if not para.strip():
            processed_paragraphs.append("") # Preserve empty lines
            continue

        # Check if it's a list item
        is_list_item = bool(re.match(r'^[\s]*[•\-\*1-9]\.?\s', para))
        
        # If user specifically asked for Bullet style, force bullet formatting
        if style == ParaphraseStyle.bullets and not is_list_item:
            prefix = "• "
        else:
            prefix = ""

        # Remove existing bullets for processing (clean text)
        clean_para = re.sub(r'^[\s]*[•\-\*]\s?', '', para).strip()

        # 2. Split paragraph into sentences to feed model correctly
        sentences = split_into_sentences(clean_para)
        processed_sentences = []
        
        for sent in sentences:
            # Process strictly
            p_sent = await process_sentence(sent, style)
            processed_sentences.append(p_sent)
            
            # Small yield to prevent blocking event loop on large docs
            await asyncio.sleep(0.01)

        # Rejoin sentences into paragraph
        rejoined_para = " ".join(processed_sentences)
        
        # Add prefix back (or add new bullet if requested)
        if is_list_item:
            # Try to keep original bullet char
            original_bullet = para.strip()[0]
            processed_paragraphs.append(f"{original_bullet} {rejoined_para}")
        elif style == ParaphraseStyle.bullets:
            processed_paragraphs.append(f"• {rejoined_para}")
        else:
            processed_paragraphs.append(rejoined_para)

    # Join back with original newlines
    return "\n".join(processed_paragraphs)

# --- FILE GENERATION UTILS (Retained and Cleaned) ---

def create_docx(text: str) -> bytes:
    try:
        from docx import Document
        doc = Document()
        for line in text.split('\n'):
            if not line.strip():
                continue
            if line.strip().startswith(('•', '-', '*')):
                doc.add_paragraph(line.strip().lstrip('•-* '), style='List Bullet')
            else:
                doc.add_paragraph(line)
        buffer = io.BytesIO()
        doc.save(buffer)
        buffer.seek(0)
        return buffer.read()
    except Exception as e:
        logger.error(f"DOCX Error: {e}")
        raise ValueError("DOCX generation failed")

def create_pdf(text: str) -> bytes:
    try:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        elements = []
        styles = getSampleStyleSheet()
        
        for line in text.split('\n'):
            if not line.strip():
                elements.append(Spacer(1, 12))
                continue
            
            # Sanitize XML
            clean_line = line.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            
            if clean_line.strip().startswith(('•', '-', '*')):
                # Remove bullet char for ReportLab style
                content = clean_line.strip().lstrip('•-* ')
                elements.append(Paragraph(content, styles['Bullet'], bulletText='•'))
            else:
                elements.append(Paragraph(clean_line, styles['Normal']))
                
        doc.build(elements)
        buffer.seek(0)
        return buffer.read()
    except Exception as e:
        logger.error(f"PDF Error: {e}")
        raise ValueError("PDF generation failed")

# --- API MODELS ---

class ParaphraseRequest(BaseModel):
    text: str = Field(..., min_length=1)
    style: ParaphraseStyle = Field(default=ParaphraseStyle.neutral)

    @field_validator('text')
    def validate_text(cls, v):
        if not v.strip():
            raise ValueError("Text cannot be empty")
        return v

class ParaphraseResponse(BaseModel):
    output: str
    word_count_in: int
    word_count_out: int

# --- ENDPOINTS ---

@app.get("/health")
async def health():
    return {
        "status": "healthy", 
        "model_device": str(model.device) if model else "offline",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/humanize", response_model=ParaphraseResponse)
async def humanize(request: ParaphraseRequest):
    """
    Main endpoint for raw text.
    Handles formatting and length issues via `process_structured_text`.
    """
    if not model:
        raise HTTPException(status_code=503, detail="Model is loading")
    
    start_time = time.time()
    
    try:
        output_text = await process_structured_text(request.text, request.style)
        
        return ParaphraseResponse(
            output=output_text,
            word_count_in=len(request.text.split()),
            word_count_out=len(output_text.split())
        )
    except Exception as e:
        logger.error(f"Processing error: {str(e)}", exc_info=True)
        # Fail safe: return original text if critical error
        return ParaphraseResponse(
            output=request.text,
            word_count_in=len(request.text.split()),
            word_count_out=len(request.text.split())
        )

@app.post("/humanize-document")
async def humanize_document(
    file: UploadFile = File(...),
    style: ParaphraseStyle = Form(ParaphraseStyle.neutral),
    output_format: str = Form("json")
):
    """
    Handles file uploads, processes text, and returns formatted file.
    """
    # 1. Extraction
    content = await file.read()
    text = ""
    filename = file.filename.lower()

    try:
        if filename.endswith('.docx'):
            from docx import Document
            doc = Document(io.BytesIO(content))
            text = "\n".join([p.text for p in doc.paragraphs])
        elif filename.endswith('.pdf'):
            # Basic PDF extraction (requires pypdf or similar installed, using simpler txt approach here)
            # For production, consider using PyPDF2 or pdfplumber
            raise HTTPException(status_code=400, detail="Direct PDF upload not fully supported in this snippet. Convert to TXT or DOCX.")
        else:
            text = content.decode('utf-8')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"File parsing failed: {str(e)}")

    # 2. Processing
    processed_text = await process_structured_text(text, style)

    # 3. Formatting
    if output_format == "json":
        return {"output": processed_text}
    
    elif output_format == "docx":
        file_bytes = create_docx(processed_text)
        media_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        fname = "humanized.docx"
    
    elif output_format == "pdf":
        file_bytes = create_pdf(processed_text)
        media_type = "application/pdf"
        fname = "humanized.pdf"
    
    else:
        raise HTTPException(status_code=400, detail="Invalid output format")

    return Response(
        content=file_bytes,
        media_type=media_type,
        headers={"Content-Disposition": f"attachment; filename={fname}"}
    )

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)