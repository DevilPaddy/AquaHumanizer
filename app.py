"""
AquaHumanizer Pro - AI Text Enhancement API
Copyright (c) 2026 AquilaStudios
Licensed under the MIT License

Production-ready paraphrasing service using T5 model with document processing capabilities.
"""

import os
import logging
import io
import asyncio
import json
from typing import Optional, List
from enum import Enum
from contextlib import asynccontextmanager
import torch
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator
from transformers import T5ForConditionalGeneration, T5Tokenizer
from docx import Document
from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global model and tokenizer (loaded once on startup)
model: Optional[T5ForConditionalGeneration] = None
tokenizer: Optional[T5Tokenizer] = None

# Configuration constants
MIN_WORDS = 5
MAX_INPUT_LENGTH = 512  # Protect CPU from very long inputs
MAX_OUTPUT_LENGTH = 128
NUM_BEAMS = 4
NO_REPEAT_NGRAM_SIZE = 3
CHUNK_TOKEN_LIMIT = 400  # Safe chunking for long documents
MAX_DOC_CHUNKS = 50  # Free-tier protection


class ParaphraseStyle(str, Enum):
    """Paraphrasing styles for different use cases"""
    neutral = "neutral"
    formal = "formal"
    ats = "ats"
    bullets = "bullets"


async def load_model():
    """Load model and tokenizer once on startup"""
    global model, tokenizer
    
    model_name = "Vamsi/T5_Paraphrase_Paws"
    logger.info(f"Loading model: {model_name}")
    
    try:
        # Load tokenizer
        logger.info("Loading tokenizer...")
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        
        # Load model
        logger.info("Loading model...")
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        
        # Set to evaluation mode
        model.eval()
        
        # Move to CPU (explicit for clarity)
        device = torch.device("cpu")
        model.to(device)
        
        logger.info("Model loaded successfully on CPU")
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for FastAPI"""
    # Startup
    logger.info("Starting up...")
    await load_model()
    logger.info("Startup complete")
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")
    # Clean up resources if needed
    logger.info("Shutdown complete")


# FastAPI app with lifespan
app = FastAPI(
    title="T5 Paraphrasing API",
    description="Production-ready paraphrasing service using T5 model",
    version="2.0.0",
    lifespan=lifespan
)


def build_prompt(text: str, style: ParaphraseStyle) -> str:
    """Build style-specific prompts for T5 model"""
    if style == ParaphraseStyle.formal:
        return f"paraphrase formally: {text}"
    elif style == ParaphraseStyle.ats:
        return (
            "paraphrase professionally for a resume, "
            "use clear action verbs, concise language, "
            f"and ATS-friendly wording: {text}"
        )
    elif style == ParaphraseStyle.bullets:
        return (
            "Rewrite the following text into concise, professional resume bullet points. "
            "Use strong action verbs, quantify impact where possible, "
            f"and keep each bullet ATS-friendly: {text}"
        )
    return f"paraphrase: {text}"


def chunk_text(text: str, tokenizer, max_tokens: int) -> List[str]:
    """Chunk text by sentences to stay within token limits"""
    sentences = text.split('. ')
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # Add period back if it was removed
        if not sentence.endswith('.'):
            sentence += '.'
            
        tokens = tokenizer.encode(sentence, add_special_tokens=False)
        
        if current_tokens + len(tokens) > max_tokens:
            if current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_tokens = len(tokens)
            else:
                # Single sentence too long, truncate
                chunks.append(sentence[:max_tokens])
        else:
            current_chunk.append(sentence)
            current_tokens += len(tokens)
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks


def paragraph_chunks(text: str, tokenizer, max_tokens: int) -> List[str]:
    """Chunk text by paragraphs for bullet-point restructuring"""
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []
    current = []
    tokens_count = 0
    
    for p in paragraphs:
        tokens = tokenizer.encode(p, add_special_tokens=False)
        if tokens_count + len(tokens) > max_tokens:
            if current:
                chunks.append("\n".join(current))
                current = [p]
                tokens_count = len(tokens)
            else:
                # Single paragraph too long, use sentence chunking
                sentence_chunks = chunk_text(p, tokenizer, max_tokens)
                chunks.extend(sentence_chunks)
        else:
            current.append(p)
            tokens_count += len(tokens)
    
    if current:
        chunks.append("\n".join(current))
    
    return chunks


def create_docx(text: str) -> bytes:
    """Create DOCX file from text"""
    doc = Document()
    
    for line in text.split("\n"):
        if line.strip():
            if line.strip().startswith("•") or line.strip().startswith("-"):
                # Bullet point
                doc.add_paragraph(line.strip(), style='List Bullet')
            else:
                # Regular paragraph
                doc.add_paragraph(line.strip())
        else:
            # Empty line for spacing
            doc.add_paragraph("")
    
    buffer = io.BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer.read()


def create_pdf(text: str) -> bytes:
    """Create PDF file from text"""
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=LETTER)
    width, height = LETTER
    y = height - 40
    
    for line in text.split("\n"):
        if y < 40:  # New page
            c.showPage()
            y = height - 40
        
        # Handle long lines
        line = line[:100] + "..." if len(line) > 100 else line
        c.drawString(40, y, line)
        y -= 14
    
    c.save()
    buffer.seek(0)
    return buffer.read()


class ParaphraseRequest(BaseModel):
    """Input schema for paraphrasing request"""
    text: str = Field(..., min_length=1, description="Text to paraphrase")
    style: ParaphraseStyle = Field(default=ParaphraseStyle.neutral, description="Paraphrasing style")

    @field_validator('text')
    @classmethod
    def validate_text(cls, v: str) -> str:
        """Validate and clean input text"""
        if not v or not v.strip():
            raise ValueError("Text cannot be empty")
        
        # Clean text lightly (remove extra whitespace, preserve punctuation)
        cleaned = ' '.join(v.strip().split())
        
        # Check word count
        word_count = len(cleaned.split())
        if word_count < MIN_WORDS:
            raise ValueError(f"Text must contain at least {MIN_WORDS} words")
        
        # For long documents, we'll chunk them
        if len(cleaned) > MAX_INPUT_LENGTH * 3:  # Allow longer text for chunking
            logger.info(f"Long document detected: {len(cleaned)} characters")
        
        return cleaned


class DocumentRequest(BaseModel):
    """Input schema for document processing"""
    style: ParaphraseStyle = Field(default=ParaphraseStyle.neutral, description="Paraphrasing style")
    output_format: str = Field(default="json", description="Output format: json, docx, or pdf")


class ParaphraseResponse(BaseModel):
    """Output schema for paraphrasing response"""
    output: str = Field(..., description="Paraphrased text")


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None and tokenizer is not None
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None and tokenizer is not None
    }


@app.post("/humanize", response_model=ParaphraseResponse)
async def humanize(request: ParaphraseRequest) -> ParaphraseResponse:
    """
    Paraphrase input text using T5 model with style support.
    
    Args:
        request: ParaphraseRequest containing text and style
        
    Returns:
        ParaphraseResponse with paraphrased text
        
    Raises:
        HTTPException: If model is not loaded or inference fails
    """
    if model is None or tokenizer is None:
        logger.error("Model not loaded")
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    input_text = request.text
    style = request.style
    logger.info(f"Received request: {len(input_text)} characters, {len(input_text.split())} words, style: {style}")
    
    try:
        # For short text, process directly
        if len(input_text.split()) <= 100:
            return await process_single_chunk(input_text, style)
        
        # For long text, use chunking
        return await process_long_text(input_text, style)
        
    except Exception as e:
        logger.error(f"Inference error: {str(e)}")
        # Return original text on error (graceful degradation)
        logger.warning("Returning original text due to inference error")
        return ParaphraseResponse(output=input_text)


async def process_single_chunk(text: str, style: ParaphraseStyle) -> ParaphraseResponse:
    """Process a single chunk of text"""
    # Prepare input with style-specific prompt
    input_text_prepared = build_prompt(text, style)
    
    # Tokenize
    input_ids = tokenizer.encode(
        input_text_prepared,
        return_tensors="pt",
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding=True
    )
    
    # Inference with no_grad for efficiency
    with torch.no_grad():
        # Generate with beam search for accuracy
        outputs = model.generate(
            input_ids,
            max_length=MAX_OUTPUT_LENGTH,
            num_beams=NUM_BEAMS,
            no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
            early_stopping=True,
            do_sample=False,  # Deterministic, no randomness
            temperature=None,  # Not used when do_sample=False
        )
    
    # Decode output
    paraphrased_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Clean output
    paraphrased_text = paraphrased_text.strip()
    
    # Fallback: if output is empty or too short, return original
    if not paraphrased_text or len(paraphrased_text.split()) < MIN_WORDS:
        logger.warning("Generated output too short, returning original text")
        paraphrased_text = text
    
    logger.info(f"Generated paraphrase: {len(paraphrased_text)} characters")
    
    return ParaphraseResponse(output=paraphrased_text)


async def process_long_text(text: str, style: ParaphraseStyle) -> ParaphraseResponse:
    """Process long text by chunking"""
    # Choose chunking strategy based on style
    if style == ParaphraseStyle.bullets:
        chunks = paragraph_chunks(text, tokenizer, CHUNK_TOKEN_LIMIT)
    else:
        chunks = chunk_text(text, tokenizer, CHUNK_TOKEN_LIMIT)
    
    if len(chunks) > MAX_DOC_CHUNKS:
        raise HTTPException(status_code=413, detail="Document too large")
    
    logger.info(f"Processing {len(chunks)} chunks")
    
    results = []
    for i, chunk in enumerate(chunks):
        logger.info(f"Processing chunk {i+1}/{len(chunks)}")
        
        # Process each chunk
        prompt = build_prompt(chunk, style)
        input_ids = tokenizer.encode(
            prompt,
            return_tensors="pt",
            max_length=MAX_INPUT_LENGTH,
            truncation=True
        )
        
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_length=MAX_OUTPUT_LENGTH,
                num_beams=NUM_BEAMS,
                no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
                early_stopping=True,
                do_sample=False
            )
        
        chunk_result = tokenizer.decode(output[0], skip_special_tokens=True)
        results.append(chunk_result.strip())
        
        # Small delay to prevent overwhelming the CPU
        await asyncio.sleep(0.1)
    
    # Join results appropriately based on style
    if style == ParaphraseStyle.bullets:
        final_output = "\n\n".join(results)
    else:
        final_output = " ".join(results)
    
    logger.info(f"Completed long text processing: {len(final_output)} characters")
    
    return ParaphraseResponse(output=final_output)


@app.post("/humanize-document")
async def humanize_document(
    file: UploadFile = File(...),
    style: ParaphraseStyle = ParaphraseStyle.neutral,
    output_format: str = "json"  # json | docx | pdf
):
    """
    Process uploaded document with style-specific paraphrasing.
    
    Args:
        file: Uploaded document (txt, docx, etc.)
        style: Paraphrasing style
        output_format: Output format (json, docx, pdf)
        
    Returns:
        Processed document in requested format
    """
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Read file content
    content = await file.read()
    filename = file.filename.lower() if file.filename else ""
    
    # Extract text based on file type
    try:
        if filename.endswith(".txt"):
            text = content.decode('utf-8')
        elif filename.endswith(".docx"):
            # For now, return error - DOCX parsing requires python-docx
            raise HTTPException(
                status_code=400, 
                detail="DOCX parsing not available in this version. Please convert to text first."
            )
        else:
            # Try to decode as text
            text = content.decode('utf-8')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not process file: {str(e)}")
    
    if not text or len(text.split()) < MIN_WORDS:
        raise HTTPException(status_code=400, detail="Could not extract valid text")
    
    # Process the text
    logger.info(f"Processing document: {len(text)} characters, style: {style}")
    
    # Use long text processing
    result = await process_long_text(text, style)
    paraphrased = result.output
    
    # Return in requested format
    if output_format == "json":
        return {"output": paraphrased, "style": style}
    
    elif output_format == "pdf":
        pdf_bytes = create_pdf(paraphrased)
        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=paraphrased.pdf"}
        )
    
    elif output_format == "docx":
        docx_bytes = create_docx(paraphrased)
        return StreamingResponse(
            io.BytesIO(docx_bytes),
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={"Content-Disposition": "attachment; filename=paraphrased.docx"}
        )
    
    else:
        raise HTTPException(status_code=400, detail="Invalid output format")


@app.post("/humanize-document-stream")
async def humanize_document_stream(
    file: UploadFile = File(...),
    style: ParaphraseStyle = ParaphraseStyle.neutral
):
    """
    Process document with streaming progress updates.
    
    Args:
        file: Uploaded document
        style: Paraphrasing style
        
    Returns:
        Streaming JSON responses with progress updates
    """
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Read and extract text
    content = await file.read()
    filename = file.filename.lower() if file.filename else ""
    
    try:
        if filename.endswith(".txt"):
            text = content.decode('utf-8')
        else:
            text = content.decode('utf-8')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not process file: {str(e)}")
    
    if not text or len(text.split()) < MIN_WORDS:
        raise HTTPException(status_code=400, detail="Could not extract valid text")
    
    # Choose chunking strategy
    if style == ParaphraseStyle.bullets:
        chunks = paragraph_chunks(text, tokenizer, CHUNK_TOKEN_LIMIT)
    else:
        chunks = chunk_text(text, tokenizer, CHUNK_TOKEN_LIMIT)
    
    if len(chunks) > MAX_DOC_CHUNKS:
        raise HTTPException(status_code=413, detail="Document too large")
    
    async def stream_processing():
        """Generator for streaming responses"""
        total = len(chunks)
        results = []
        
        for idx, chunk in enumerate(chunks, start=1):
            # Send progress update
            yield json.dumps({
                "type": "progress",
                "current": idx,
                "total": total,
                "message": f"Processing chunk {idx} of {total}"
            }) + "\n"
            
            # Process chunk
            prompt = build_prompt(chunk, style)
            input_ids = tokenizer.encode(
                prompt,
                return_tensors="pt",
                max_length=MAX_INPUT_LENGTH,
                truncation=True
            )
            
            with torch.no_grad():
                output = model.generate(
                    input_ids,
                    max_length=MAX_OUTPUT_LENGTH,
                    num_beams=NUM_BEAMS,
                    no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
                    early_stopping=True,
                    do_sample=False
                )
            
            chunk_result = tokenizer.decode(output[0], skip_special_tokens=True)
            results.append(chunk_result.strip())
            
            # Send chunk result
            yield json.dumps({
                "type": "chunk",
                "content": chunk_result.strip(),
                "index": idx
            }) + "\n"
            
            await asyncio.sleep(0.1)  # Allow event loop to process
        
        # Send final result
        if style == ParaphraseStyle.bullets:
            final_output = "\n\n".join(results)
        else:
            final_output = " ".join(results)
        
        yield json.dumps({
            "type": "done",
            "output": final_output,
            "style": style
        }) + "\n"
    
    return StreamingResponse(
        stream_processing(),
        media_type="application/json"
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
