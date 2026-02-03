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
import time
from typing import Optional, List
from enum import Enum
from contextlib import asynccontextmanager
import torch
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel, Field, field_validator
from transformers import T5ForConditionalGeneration, T5Tokenizer
from docx import Document
from reportlab.lib.pagesizes import LETTER, letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.enums import TA_LEFT, TA_JUSTIFY
from reportlab.pdfgen import canvas

# Configure enhanced logging with structured format
import uuid
from datetime import datetime

class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured logging"""
    
    def format(self, record):
        # Create structured log entry
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "component": record.name,
            "message": record.getMessage(),
        }
        
        # Add request ID if available
        if hasattr(record, 'request_id'):
            log_entry["request_id"] = record.request_id
        
        # Add extra context if available
        if hasattr(record, 'context'):
            log_entry["context"] = record.context
        
        # Add error details if it's an error
        if record.levelno >= logging.ERROR and record.exc_info:
            log_entry["error_type"] = record.exc_info[0].__name__ if record.exc_info[0] else None
            log_entry["stack_trace"] = self.formatException(record.exc_info) if record.exc_info else None
        
        return json.dumps(log_entry)

# Configure logging with structured format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Create structured logger for API requests
structured_handler = logging.StreamHandler()
structured_handler.setFormatter(StructuredFormatter())
api_logger = logging.getLogger("api_requests")
api_logger.addHandler(structured_handler)
api_logger.setLevel(logging.INFO)

logger = logging.getLogger(__name__)

# Error tracking utilities
class ErrorTracker:
    """Track and log errors with context"""
    
    @staticmethod
    def log_request(request_data: dict, request_id: str = None) -> str:
        """Log incoming request with unique ID"""
        if not request_id:
            request_id = str(uuid.uuid4())[:8]
        
        api_logger.info(
            "Incoming request",
            extra={
                "request_id": request_id,
                "context": {
                    "endpoint": request_data.get("endpoint"),
                    "method": request_data.get("method"),
                    "content_length": request_data.get("content_length", 0),
                    "user_agent": request_data.get("user_agent", "unknown")
                }
            }
        )
        return request_id
    
    @staticmethod
    def log_processing_stage(stage: str, data: dict, request_id: str = None):
        """Log processing stage with timing and context"""
        api_logger.info(
            f"Processing stage: {stage}",
            extra={
                "request_id": request_id,
                "context": {
                    "stage": stage,
                    **data
                }
            }
        )
    
    @staticmethod
    def capture_error(error: Exception, context: dict, request_id: str = None) -> str:
        """Capture error with full context"""
        error_id = str(uuid.uuid4())[:8]
        
        api_logger.error(
            f"Error occurred: {str(error)}",
            exc_info=True,
            extra={
                "request_id": request_id,
                "context": {
                    "error_id": error_id,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    **context
                }
            }
        )
        return error_id

error_tracker = ErrorTracker()

# Validate dependencies on startup
def validate_dependencies():
    """Validate that all required dependencies are available"""
    try:
        # Test ReportLab imports
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import SimpleDocTemplate, Paragraph
        logger.info("✅ ReportLab dependencies validated")
        
        # Test python-docx imports
        from docx import Document
        from docx.shared import Pt, Inches
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        logger.info("✅ python-docx dependencies validated")
        
        return True
    except ImportError as e:
        logger.error(f"❌ Missing dependency: {str(e)}")
        raise HTTPException(
            status_code=503, 
            detail=f"Required dependency missing: {str(e)}"
        )
    except Exception as e:
        logger.error(f"❌ Dependency validation error: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail=f"Dependency validation failed: {str(e)}"
        )

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
    
    # Validate dependencies first
    validate_dependencies()
    
    # Then load model
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
    """
    Create DOCX file from text with proper formatting.
    Handles paragraphs, bullet points, and headings.
    """
    try:
        from docx import Document
        from docx.shared import Pt, RGBColor, Inches
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        
        logger.info(f"📄 Starting DOCX generation for {len(text)} characters")
        start_time = time.time()
        
        doc = Document()
        
        # Set default styling
        style = doc.styles['Normal']
        font = style.font
        font.name = 'Calibri'
        font.size = Pt(11)
        
        # Set page margins
        sections = doc.sections
        for section in sections:
            section.top_margin = Inches(1)
            section.bottom_margin = Inches(1)
            section.left_margin = Inches(1)
            section.right_margin = Inches(1)
        
        # Split text into paragraphs
        paragraphs = text.split('\n\n')
        paragraph_count = 0
        bullet_count = 0
        
        for paragraph_text in paragraphs:
            if not paragraph_text.strip():
                continue
                
            # Split by lines for bullet points
            lines = paragraph_text.split('\n')
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Handle bullet points
                if line.startswith('•') or line.startswith('-') or line.startswith('*'):
                    # Remove bullet character
                    clean_line = line[1:].strip()
                    doc.add_paragraph(clean_line, style='List Bullet')
                    bullet_count += 1
                    
                # Handle headings (all caps, short)
                elif line.isupper() and len(line.split()) <= 5:
                    heading = doc.add_heading(line, level=2)
                    
                # Regular paragraph
                else:
                    para = doc.add_paragraph(line)
                    para.alignment = WD_ALIGN_PARAGRAPH.LEFT
                    paragraph_count += 1
            
            # Add spacing between sections (empty paragraph)
            doc.add_paragraph()
        
        # Remove last empty paragraph
        if len(doc.paragraphs) > 0 and not doc.paragraphs[-1].text.strip():
            doc._element.body.remove(doc.paragraphs[-1]._element)
        
        # Save to buffer
        buffer = io.BytesIO()
        doc.save(buffer)
        buffer.seek(0)
        docx_bytes = buffer.read()
        
        # Validate DOCX was created properly
        validation_result = validate_docx_content(docx_bytes)
        
        if not validation_result['valid']:
            error_msg = validation_result.get('error', 'DOCX validation failed')
            logger.error(f"❌ DOCX validation failed: {error_msg}")
            raise ValueError(error_msg)
        
        processing_time = time.time() - start_time
        logger.info(f"✅ DOCX created successfully: {len(docx_bytes)} bytes, "
                   f"{paragraph_count} paragraphs, {bullet_count} bullets, "
                   f"processed in {processing_time:.2f}s")
        return docx_bytes
        
    except Exception as e:
        logger.error(f"❌ DOCX generation error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"DOCX generation failed: {str(e)}"
        )


def create_pdf(text: str) -> bytes:
    """
    Create PDF file from text with proper formatting using ReportLab.
    Uses SimpleDocTemplate for robust PDF generation.
    """
    try:
        logger.info(f"📑 Starting PDF generation for {len(text)} characters")
        start_time = time.time()
        
        buffer = io.BytesIO()
        
        # Create PDF document with proper margins
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18,
        )
        
        # Container for flowable objects
        elements = []
        
        # Get default styles
        styles = getSampleStyleSheet()
        style_normal = ParagraphStyle(
            'CustomNormal',
            parent=styles['Normal'],
            fontSize=11,
            leading=16,
            alignment=TA_JUSTIFY,
            spaceAfter=12,
        )
        
        style_bullet = ParagraphStyle(
            'CustomBullet',
            parent=styles['Normal'],
            fontSize=11,
            leading=16,
            alignment=TA_LEFT,
            leftIndent=20,
            spaceAfter=8,
        )
        
        # Split text into paragraphs
        paragraphs = text.split('\n\n')
        paragraph_count = 0
        bullet_count = 0
        heading_count = 0
        
        for para_text in paragraphs:
            if not para_text.strip():
                continue
                
            # Check if this is a multi-line section with bullets
            lines = para_text.split('\n')
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Escape XML special characters for ReportLab
                line = (line
                    .replace('&', '&amp;')
                    .replace('<', '&lt;')
                    .replace('>', '&gt;')
                    .replace('"', '&quot;')
                    .replace("'", '&apos;')
                )
                
                # Detect and format bullet points
                if line.startswith('•') or line.startswith('-') or line.startswith('*'):
                    # Remove bullet character and create bullet paragraph
                    clean_line = line[1:].strip()
                    bullet_text = f"• {clean_line}"
                    para = Paragraph(bullet_text, style_bullet)
                    elements.append(para)
                    bullet_count += 1
                elif line.isupper() and len(line.split()) <= 5:
                    # Heading style
                    para = Paragraph(f"<b>{line}</b>", style_normal)
                    elements.append(para)
                    elements.append(Spacer(1, 0.15 * inch))
                    heading_count += 1
                else:
                    # Regular paragraph
                    para = Paragraph(line, style_normal)
                    elements.append(para)
                    paragraph_count += 1
            
            # Add spacing between sections
            elements.append(Spacer(1, 0.2 * inch))
        
        # Build PDF
        doc.build(elements)
        
        # Get PDF bytes
        buffer.seek(0)
        pdf_bytes = buffer.read()
        
        # Validate PDF was created properly
        validation_result = validate_pdf_content(pdf_bytes)
        
        if not validation_result['valid']:
            error_msg = validation_result.get('error', 'PDF validation failed')
            logger.error(f"❌ PDF validation failed: {error_msg}")
            raise ValueError(error_msg)
        
        processing_time = time.time() - start_time
        logger.info(f"✅ PDF created successfully: {len(pdf_bytes)} bytes, "
                   f"{paragraph_count} paragraphs, {bullet_count} bullets, "
                   f"{heading_count} headings, processed in {processing_time:.2f}s")
        return pdf_bytes
        
    except Exception as e:
        logger.error(f"❌ PDF generation error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"PDF generation failed: {str(e)}"
        )


def validate_pdf_content(pdf_bytes: bytes) -> dict:
    """
    Validate PDF file content and headers
    
    Args:
        pdf_bytes: The PDF file content as bytes
        
    Returns:
        dict: Validation result with status and details
    """
    try:
        # Check if content is empty
        if len(pdf_bytes) == 0:
            return {
                'valid': False,
                'error': 'PDF file is empty',
                'size': 0,
                'has_header': False
            }
        
        # Check PDF header
        has_pdf_header = pdf_bytes.startswith(b'%PDF')
        
        # Additional PDF structure validation
        has_eof = b'%%EOF' in pdf_bytes
        
        validation_result = {
            'valid': has_pdf_header and len(pdf_bytes) > 0,
            'size': len(pdf_bytes),
            'has_header': has_pdf_header,
            'has_eof': has_eof,
            'content_type': 'application/pdf'
        }
        
        if not has_pdf_header:
            validation_result['error'] = 'Invalid PDF header - file does not start with %PDF'
        elif not has_eof:
            validation_result['warning'] = 'PDF may be incomplete - missing %%EOF marker'
        
        logger.info(f"PDF validation: {validation_result}")
        return validation_result
        
    except Exception as e:
        logger.error(f"PDF validation error: {str(e)}")
        return {
            'valid': False,
            'error': f'PDF validation failed: {str(e)}',
            'size': len(pdf_bytes) if pdf_bytes else 0,
            'has_header': False
        }


def validate_docx_content(docx_bytes: bytes) -> dict:
    """
    Validate DOCX file content and headers
    
    Args:
        docx_bytes: The DOCX file content as bytes
        
    Returns:
        dict: Validation result with status and details
    """
    try:
        # Check if content is empty
        if len(docx_bytes) == 0:
            return {
                'valid': False,
                'error': 'DOCX file is empty',
                'size': 0,
                'has_header': False
            }
        
        # Check DOCX header (ZIP format)
        has_zip_header = docx_bytes.startswith(b'PK')
        
        # Additional DOCX structure validation
        # DOCX files should contain specific XML files
        has_content_types = b'[Content_Types].xml' in docx_bytes
        has_word_document = b'word/document.xml' in docx_bytes
        
        validation_result = {
            'valid': has_zip_header and len(docx_bytes) > 0,
            'size': len(docx_bytes),
            'has_header': has_zip_header,
            'has_content_types': has_content_types,
            'has_word_document': has_word_document,
            'content_type': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        }
        
        if not has_zip_header:
            validation_result['error'] = 'Invalid DOCX header - file does not start with PK (ZIP format)'
        elif not has_content_types:
            validation_result['warning'] = 'DOCX may be incomplete - missing Content_Types.xml'
        elif not has_word_document:
            validation_result['warning'] = 'DOCX may be incomplete - missing word/document.xml'
        
        logger.info(f"DOCX validation: {validation_result}")
        return validation_result
        
    except Exception as e:
        logger.error(f"DOCX validation error: {str(e)}")
        return {
            'valid': False,
            'error': f'DOCX validation failed: {str(e)}',
            'size': len(docx_bytes) if docx_bytes else 0,
            'has_header': False
        }


def validate_file_content(file_bytes: bytes, expected_format: str) -> dict:
    """
    Validate file content based on expected format
    
    Args:
        file_bytes: The file content as bytes
        expected_format: Expected format ('pdf' or 'docx')
        
    Returns:
        dict: Validation result with status and details
    """
    try:
        if expected_format.lower() == 'pdf':
            return validate_pdf_content(file_bytes)
        elif expected_format.lower() == 'docx':
            return validate_docx_content(file_bytes)
        else:
            return {
                'valid': False,
                'error': f'Unsupported format for validation: {expected_format}',
                'size': len(file_bytes) if file_bytes else 0
            }
    except Exception as e:
        logger.error(f"File validation error: {str(e)}")
        return {
            'valid': False,
            'error': f'File validation failed: {str(e)}',
            'size': len(file_bytes) if file_bytes else 0
        }


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
    """Enhanced health check endpoint with comprehensive system status"""
    import psutil
    import sys
    
    try:
        # Basic model status
        model_loaded = model is not None and tokenizer is not None
        
        # System information
        memory_info = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Test model inference if loaded
        inference_working = False
        inference_time = None
        if model_loaded:
            try:
                start_time = time.time()
                test_result = await process_single_chunk("Test sentence for health check.", ParaphraseStyle.neutral)
                inference_time = time.time() - start_time
                inference_working = len(test_result.output) > 0
            except Exception as e:
                logger.warning(f"Health check inference failed: {e}")
                inference_working = False
        
        # Determine overall status
        if model_loaded and inference_working:
            overall_status = "healthy"
        elif model_loaded:
            overall_status = "degraded"
        else:
            overall_status = "unhealthy"
        
        health_data = {
            "status": overall_status,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "model_status": {
                "loaded": model_loaded,
                "inference_working": inference_working,
                "inference_time_ms": round(inference_time * 1000, 2) if inference_time else None
            },
            "system_info": {
                "python_version": sys.version.split()[0],
                "memory_usage_mb": round(memory_info.used / 1024 / 1024, 2),
                "memory_available_mb": round(memory_info.available / 1024 / 1024, 2),
                "memory_percent": memory_info.percent,
                "cpu_percent": cpu_percent
            },
            "dependencies": {
                "torch_version": torch.__version__,
                "transformers_available": True,  # If we got here, transformers is available
                "reportlab_available": True,     # If we got here, reportlab is available
                "docx_available": True           # If we got here, python-docx is available
            }
        }
        
        # Log health check
        logger.info(f"Health check completed: {overall_status}")
        
        return health_data
        
    except Exception as e:
        error_id = error_tracker.capture_error(e, {"endpoint": "/health"})
        return {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "error": f"Health check failed: {str(e)}",
            "error_id": error_id
        }

@app.get("/model-status")
async def model_status():
    """Detailed model status endpoint"""
    try:
        model_loaded = model is not None and tokenizer is not None
        
        if not model_loaded:
            return {
                "loaded": False,
                "error": "Model or tokenizer not loaded",
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        
        # Test inference with timing
        test_texts = [
            "Short test.",
            "This is a medium length test sentence for model verification.",
            "This is a longer test sentence that will help us understand how the model performs with more substantial input text for comprehensive testing."
        ]
        
        inference_results = []
        for i, text in enumerate(test_texts):
            try:
                start_time = time.time()
                result = await process_single_chunk(text, ParaphraseStyle.neutral)
                inference_time = time.time() - start_time
                
                inference_results.append({
                    "test_case": i + 1,
                    "input_length": len(text),
                    "output_length": len(result.output),
                    "inference_time_ms": round(inference_time * 1000, 2),
                    "success": True
                })
            except Exception as e:
                inference_results.append({
                    "test_case": i + 1,
                    "input_length": len(text),
                    "success": False,
                    "error": str(e)
                })
        
        return {
            "loaded": True,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "model_info": {
                "model_name": "Vamsi/T5_Paraphrase_Paws",
                "device": "cpu"
            },
            "inference_tests": inference_results,
            "performance_summary": {
                "successful_tests": sum(1 for r in inference_results if r["success"]),
                "total_tests": len(inference_results),
                "avg_inference_time_ms": round(
                    sum(r.get("inference_time_ms", 0) for r in inference_results if r["success"]) / 
                    max(1, sum(1 for r in inference_results if r["success"])), 2
                )
            }
        }
        
    except Exception as e:
        error_id = error_tracker.capture_error(e, {"endpoint": "/model-status"})
        return {
            "loaded": False,
            "error": str(e),
            "error_id": error_id,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

@app.get("/environment")
async def environment_info():
    """System environment information endpoint"""
    try:
        import psutil
        import sys
        import platform
        
        # Get package versions
        package_versions = {}
        try:
            import torch
            package_versions["torch"] = torch.__version__
        except ImportError:
            package_versions["torch"] = "not_installed"
        
        try:
            import transformers
            package_versions["transformers"] = transformers.__version__
        except ImportError:
            package_versions["transformers"] = "not_installed"
        
        try:
            import reportlab
            package_versions["reportlab"] = reportlab.Version
        except ImportError:
            package_versions["reportlab"] = "not_installed"
        
        try:
            import docx
            package_versions["python-docx"] = "installed"  # python-docx doesn't have __version__
        except ImportError:
            package_versions["python-docx"] = "not_installed"
        
        # System information
        memory_info = psutil.virtual_memory()
        disk_info = psutil.disk_usage('/')
        
        return {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "system": {
                "platform": platform.platform(),
                "python_version": sys.version,
                "architecture": platform.architecture()[0],
                "processor": platform.processor() or "unknown"
            },
            "memory": {
                "total_gb": round(memory_info.total / 1024 / 1024 / 1024, 2),
                "available_gb": round(memory_info.available / 1024 / 1024 / 1024, 2),
                "used_gb": round(memory_info.used / 1024 / 1024 / 1024, 2),
                "percent_used": memory_info.percent
            },
            "disk": {
                "total_gb": round(disk_info.total / 1024 / 1024 / 1024, 2),
                "free_gb": round(disk_info.free / 1024 / 1024 / 1024, 2),
                "used_gb": round(disk_info.used / 1024 / 1024 / 1024, 2),
                "percent_used": round((disk_info.used / disk_info.total) * 100, 2)
            },
            "packages": package_versions,
            "environment_variables": {
                "PORT": os.getenv("PORT", "not_set"),
                "PYTHONPATH": os.getenv("PYTHONPATH", "not_set")
            }
        }
        
    except Exception as e:
        error_id = error_tracker.capture_error(e, {"endpoint": "/environment"})
        return {
            "error": str(e),
            "error_id": error_id,
            "timestamp": datetime.utcnow().isoformat() + "Z"
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
    style: ParaphraseStyle = Form(ParaphraseStyle.neutral),
    output_format: str = Form("json")  # json | docx | pdf
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
    # Generate request ID for tracking
    request_id = error_tracker.log_request({
        "endpoint": "/humanize-document",
        "method": "POST",
        "filename": file.filename,
        "content_type": file.content_type,
        "style": style,
        "output_format": output_format
    })
    
    request_start_time = time.time()
    
    try:
        # Check model availability
        if model is None or tokenizer is None:
            error_tracker.capture_error(
                Exception("Model not loaded"), 
                {"stage": "model_check", "request_id": request_id}
            )
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        error_tracker.log_processing_stage("file_upload", {
            "filename": file.filename,
            "content_type": file.content_type,
            "style": style,
            "output_format": output_format
        }, request_id)
        
        # Read file content
        content = await file.read()
        filename = file.filename.lower() if file.filename else ""
        
        error_tracker.log_processing_stage("file_read", {
            "file_size_bytes": len(content),
            "filename": filename
        }, request_id)
        
        # Extract text based on file type
        text_extraction_start = time.time()
        try:
            if filename.endswith(".txt"):
                text = content.decode('utf-8')
                error_tracker.log_processing_stage("text_extraction", {
                    "file_type": "txt",
                    "extracted_length": len(text),
                    "extraction_time_ms": round((time.time() - text_extraction_start) * 1000, 2)
                }, request_id)
            elif filename.endswith(".docx"):
                # Parse DOCX file using python-docx
                try:
                    from docx import Document
                    doc_file = io.BytesIO(content)
                    doc = Document(doc_file)
                    text = "\n".join([paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip()])
                    if not text.strip():
                        error_tracker.capture_error(
                            ValueError("DOCX file appears to be empty"),
                            {"stage": "docx_parsing", "filename": filename},
                            request_id
                        )
                        raise HTTPException(status_code=400, detail="DOCX file appears to be empty")
                    
                    error_tracker.log_processing_stage("text_extraction", {
                        "file_type": "docx",
                        "extracted_length": len(text),
                        "paragraph_count": len(doc.paragraphs),
                        "extraction_time_ms": round((time.time() - text_extraction_start) * 1000, 2)
                    }, request_id)
                except Exception as docx_error:
                    error_tracker.capture_error(docx_error, {
                        "stage": "docx_parsing",
                        "filename": filename,
                        "file_size": len(content)
                    }, request_id)
                    raise HTTPException(status_code=400, detail=f"Could not parse DOCX file: {str(docx_error)}")
            else:
                # Try to decode as text
                text = content.decode('utf-8')
                error_tracker.log_processing_stage("text_extraction", {
                    "file_type": "generic",
                    "extracted_length": len(text),
                    "extraction_time_ms": round((time.time() - text_extraction_start) * 1000, 2)
                }, request_id)
        except HTTPException:
            # Re-raise HTTP exceptions
            raise
        except Exception as e:
            error_tracker.capture_error(e, {
                "stage": "text_extraction",
                "filename": filename,
                "file_size": len(content)
            }, request_id)
            raise HTTPException(status_code=400, detail=f"Could not process file: {str(e)}")
        
        # Validate extracted text
        word_count = len(text.split())
        if not text or word_count < MIN_WORDS:
            error_tracker.capture_error(
                ValueError(f"Invalid text extracted: {word_count} words (minimum: {MIN_WORDS})"),
                {"stage": "text_validation", "word_count": word_count, "text_length": len(text)},
                request_id
            )
            raise HTTPException(status_code=400, detail="Could not extract valid text")
        
        error_tracker.log_processing_stage("text_validation", {
            "text_length": len(text),
            "word_count": word_count,
            "validation_passed": True
        }, request_id)
        
        # Process the text
        processing_start_time = time.time()
        error_tracker.log_processing_stage("text_processing_start", {
            "input_length": len(text),
            "word_count": word_count,
            "style": style
        }, request_id)
        
        result = await process_long_text(text, style)
        paraphrased = result.output
        processing_time = time.time() - processing_start_time
        
        error_tracker.log_processing_stage("text_processing_complete", {
            "output_length": len(paraphrased),
            "processing_time_ms": round(processing_time * 1000, 2)
        }, request_id)
        
        # Return in requested format
        if output_format == "json":
            total_time = time.time() - request_start_time
            error_tracker.log_processing_stage("response_json", {
                "total_time_ms": round(total_time * 1000, 2)
            }, request_id)
            return {"output": paraphrased, "style": style}
        
        elif output_format == "pdf":
            file_start_time = time.time()
            error_tracker.log_processing_stage("pdf_generation_start", {
                "input_length": len(paraphrased)
            }, request_id)
            
            pdf_bytes = create_pdf(paraphrased)
            file_time = time.time() - file_start_time
            total_time = time.time() - request_start_time
            
            error_tracker.log_processing_stage("pdf_generation_complete", {
                "pdf_size_bytes": len(pdf_bytes),
                "generation_time_ms": round(file_time * 1000, 2),
                "total_time_ms": round(total_time * 1000, 2)
            }, request_id)
            
            return Response(
                content=pdf_bytes,
                media_type="application/pdf",
                headers={
                    "Content-Disposition": "attachment; filename=humanized.pdf",
                    "Content-Length": str(len(pdf_bytes)),
                    "Cache-Control": "no-cache, no-store, must-revalidate"
                }
            )
        
        elif output_format == "docx":
            file_start_time = time.time()
            error_tracker.log_processing_stage("docx_generation_start", {
                "input_length": len(paraphrased)
            }, request_id)
            
            docx_bytes = create_docx(paraphrased)
            file_time = time.time() - file_start_time
            total_time = time.time() - request_start_time
            
            error_tracker.log_processing_stage("docx_generation_complete", {
                "docx_size_bytes": len(docx_bytes),
                "generation_time_ms": round(file_time * 1000, 2),
                "total_time_ms": round(total_time * 1000, 2)
            }, request_id)
            
            return Response(
                content=docx_bytes,
                media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                headers={
                    "Content-Disposition": "attachment; filename=humanized.docx",
                    "Content-Length": str(len(docx_bytes)),
                    "Cache-Control": "no-cache, no-store, must-revalidate"
                }
            )
        
        else:
            error_tracker.capture_error(
                ValueError(f"Invalid output format: {output_format}"),
                {"stage": "format_validation", "output_format": output_format},
                request_id
            )
            raise HTTPException(status_code=400, detail="Invalid output format")
    
    except HTTPException:
        # Re-raise HTTP exceptions (they're already logged)
        raise
    except Exception as e:
        # Capture any unexpected errors
        error_id = error_tracker.capture_error(e, {
            "stage": "unexpected_error",
            "request_id": request_id,
            "total_time_ms": round((time.time() - request_start_time) * 1000, 2)
        }, request_id)
        raise HTTPException(status_code=500, detail=f"Internal server error (ID: {error_id})")
    try:
        if filename.endswith(".txt"):
            text = content.decode('utf-8')
            logger.info(f"📝 Extracted text from TXT: {len(text)} characters")
        elif filename.endswith(".docx"):
            # Parse DOCX file using python-docx
            try:
                from docx import Document
                doc_file = io.BytesIO(content)
                doc = Document(doc_file)
                text = "\n".join([paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip()])
                if not text.strip():
                    logger.error("❌ DOCX file appears to be empty")
                    raise HTTPException(status_code=400, detail="DOCX file appears to be empty")
                logger.info(f"📄 Extracted text from DOCX: {len(text)} characters, "
                           f"{len(doc.paragraphs)} paragraphs")
            except Exception as docx_error:
                logger.error(f"❌ DOCX parsing error: {str(docx_error)}", exc_info=True)
                raise HTTPException(status_code=400, detail=f"Could not parse DOCX file: {str(docx_error)}")
        else:
            # Try to decode as text
            text = content.decode('utf-8')
            logger.info(f"📝 Extracted text from generic file: {len(text)} characters")
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"❌ File processing error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Could not process file: {str(e)}")
    
    if not text or len(text.split()) < MIN_WORDS:
        logger.error(f"❌ Invalid text extracted: {len(text.split())} words (minimum: {MIN_WORDS})")
        raise HTTPException(status_code=400, detail="Could not extract valid text")
    
    # Process the text
    word_count = len(text.split())
    logger.info(f"🤖 Processing document: {len(text)} characters, {word_count} words, style: {style}")
    
    # Use long text processing
    processing_start_time = time.time()
    result = await process_long_text(text, style)
    paraphrased = result.output
    processing_time = time.time() - processing_start_time
    
    logger.info(f"✅ Text processing completed in {processing_time:.2f}s: "
               f"{len(paraphrased)} characters output")
    
    # Return in requested format
    if output_format == "json":
        total_time = time.time() - request_start_time
        logger.info(f"📤 Returning JSON response (total time: {total_time:.2f}s)")
        return {"output": paraphrased, "style": style}
    
    elif output_format == "pdf":
        logger.info(f"📑 Generating PDF for {len(paraphrased)} characters")
        file_start_time = time.time()
        pdf_bytes = create_pdf(paraphrased)
        file_time = time.time() - file_start_time
        total_time = time.time() - request_start_time
        
        logger.info(f"📤 Returning PDF response: {len(pdf_bytes)} bytes "
                   f"(file generation: {file_time:.2f}s, total: {total_time:.2f}s)")
        
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": "attachment; filename=humanized.pdf",
                "Content-Length": str(len(pdf_bytes)),
                "Cache-Control": "no-cache, no-store, must-revalidate"
            }
        )
    
    elif output_format == "docx":
        logger.info(f"📄 Generating DOCX for {len(paraphrased)} characters")
        file_start_time = time.time()
        docx_bytes = create_docx(paraphrased)
        file_time = time.time() - file_start_time
        total_time = time.time() - request_start_time
        
        logger.info(f"📤 Returning DOCX response: {len(docx_bytes)} bytes "
                   f"(file generation: {file_time:.2f}s, total: {total_time:.2f}s)")
        
        return Response(
            content=docx_bytes,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={
                "Content-Disposition": "attachment; filename=humanized.docx",
                "Content-Length": str(len(docx_bytes)),
                "Cache-Control": "no-cache, no-store, must-revalidate"
            }
        )
    
    else:
        logger.error(f"❌ Invalid output format requested: {output_format}")
        raise HTTPException(status_code=400, detail="Invalid output format")


@app.post("/humanize-document-stream")
async def humanize_document_stream(
    file: UploadFile = File(...),
    style: ParaphraseStyle = Form(ParaphraseStyle.neutral)
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
