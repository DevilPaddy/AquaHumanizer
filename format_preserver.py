"""
Formatting Preservation Module
Ensures proper line breaks and structure are maintained
"""

import re
from typing import List


def extract_text_from_docx(doc) -> str:
    """
    Extract text from DOCX with PERFECT formatting preservation.
    Maintains:
    - Single line breaks
    - Double line breaks (paragraphs)
    - Empty lines
    - List structure
    """
    lines = []
    
    for para in doc.paragraphs:
        text = para.text.strip()
        
        # Empty paragraph = blank line
        if not text:
            lines.append('')
            continue
        
        # Add the paragraph text
        lines.append(text)
    
    # Join with single newlines
    return '\n'.join(lines)


def normalize_line_breaks(text: str) -> str:
    """
    Normalize line breaks to ensure consistency.
    - Preserves intentional blank lines
    - Removes excessive blank lines (more than 2)
    - Maintains paragraph structure
    """
    # Split into lines
    lines = text.split('\n')
    
    # Process lines
    normalized = []
    prev_empty = False
    empty_count = 0
    
    for line in lines:
        stripped = line.strip()
        
        if not stripped:
            # Empty line
            empty_count += 1
            if empty_count <= 2:  # Allow max 2 consecutive empty lines
                normalized.append('')
                prev_empty = True
        else:
            # Non-empty line
            normalized.append(stripped)
            prev_empty = False
            empty_count = 0
    
    # Join back
    return '\n'.join(normalized)


def preserve_structure(text: str) -> str:
    """
    Ensure structural elements are properly formatted.
    - Metadata lines (Title:, Authors:, etc.)
    - Headings
    - List items
    - Paragraphs
    """
    lines = text.split('\n')
    formatted = []
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        if not stripped:
            formatted.append('')
            continue
        
        # Check if it's a metadata line (Title:, Authors:, etc.)
        if re.match(r'^(Title|Authors?|Date|Copyright|Abstract|Figure|Table|Introduction|Conclusion|History|Background|Problem|Solution|Mechanism|Features):', stripped, re.IGNORECASE):
            # Ensure blank line before metadata (except first line)
            if i > 0 and formatted and formatted[-1] != '':
                formatted.append('')
            formatted.append(stripped)
            # Ensure blank line after metadata
            if i < len(lines) - 1 and lines[i + 1].strip():
                formatted.append('')
            continue
        
        # Check if it's a heading (short line without ending punctuation)
        words = stripped.split()
        if len(words) <= 8 and not stripped.endswith(('.', '!', '?', ',')):
            # Ensure blank line before heading (except first line)
            if i > 0 and formatted and formatted[-1] != '':
                formatted.append('')
            formatted.append(stripped)
            # Ensure blank line after heading
            if i < len(lines) - 1 and lines[i + 1].strip():
                formatted.append('')
            continue
        
        # Regular line
        formatted.append(stripped)
    
    return '\n'.join(formatted)


def format_output_text(text: str) -> str:
    """
    Main function to format output text properly.
    Call this AFTER humanization to ensure proper formatting.
    """
    # Step 1: Normalize line breaks
    text = normalize_line_breaks(text)
    
    # Step 2: Preserve structure
    text = preserve_structure(text)
    
    # Step 3: Clean up excessive blank lines
    text = re.sub(r'\n{4,}', '\n\n\n', text)
    
    # Step 4: Ensure no trailing/leading whitespace
    text = text.strip()
    
    return text
