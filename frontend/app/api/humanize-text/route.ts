import { NextRequest, NextResponse } from 'next/server'

// Simple in-memory cache (resets on cold starts)
const cache = new Map<string, { output: string; timestamp: number }>()
const CACHE_TTL = 60 * 60 * 1000 // 1 hour

interface HumanizeRequest {
  text: string
  style?: 'neutral' | 'formal' | 'ats' | 'bullets'
}

interface HFSpaceResponse {
  output: string
}

// Configuration
const HF_SPACE_URL = process.env.HF_SPACE_URL || 'https://devilseye2004-aq-humanizer.hf.space'
const REQUEST_TIMEOUT = 60000 // 60 seconds for long documents
const MAX_TEXT_LENGTH = 10000 // Increased for long documents
const MIN_WORDS = 5

function validateText(text: string): string {
  if (!text || !text.trim()) {
    throw new Error('Text cannot be empty')
  }
  
  const cleaned = text.trim().replace(/\s+/g, ' ')
  
  if (cleaned.length > MAX_TEXT_LENGTH) {
    throw new Error(`Text exceeds maximum length of ${MAX_TEXT_LENGTH} characters`)
  }
  
  const wordCount = cleaned.split(' ').length
  if (wordCount < MIN_WORDS) {
    throw new Error(`Text must contain at least ${MIN_WORDS} words`)
  }
  
  return cleaned
}

function getCacheKey(text: string, style: string): string {
  // Simple hash function for caching with style
  let hash = 0
  const combined = `${text}:${style}`
  for (let i = 0; i < combined.length; i++) {
    const char = combined.charCodeAt(i)
    hash = ((hash << 5) - hash) + char
    hash = hash & hash // Convert to 32-bit integer
  }
  return hash.toString()
}

async function callHFSpace(text: string, style: string = 'neutral'): Promise<string> {
  const controller = new AbortController()
  const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT)
  
  try {
    const response = await fetch(`${HF_SPACE_URL}/humanize`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ text, style }),
      signal: controller.signal,
    })
    
    clearTimeout(timeoutId)
    
    if (!response.ok) {
      throw new Error(`HF Space error: ${response.status}`)
    }
    
    const result: HFSpaceResponse = await response.json()
    return result.output || text
    
  } catch (error) {
    clearTimeout(timeoutId)
    console.error('Error calling HF Space:', error)
    throw error
  }
}

export async function POST(request: NextRequest) {
  try {
    const body: HumanizeRequest = await request.json()
    
    // Validate input
    const cleanedText = validateText(body.text)
    const style = body.style || 'neutral'
    
    // Check cache
    const cacheKey = getCacheKey(cleanedText, style)
    const cached = cache.get(cacheKey)
    
    if (cached && Date.now() - cached.timestamp < CACHE_TTL) {
      console.log('Returning cached result')
      return NextResponse.json({
        output: cached.output,
        cached: true,
        style: style
      })
    }
    
    // Call HF Space
    try {
      const humanizedText = await callHFSpace(cleanedText, style)
      
      // Cache the result
      cache.set(cacheKey, {
        output: humanizedText,
        timestamp: Date.now()
      })
      
      return NextResponse.json({
        output: humanizedText,
        cached: false,
        style: style
      })
      
    } catch (error) {
      console.error('HF Space error, returning original text:', error)
      return NextResponse.json({
        output: cleanedText,
        cached: false,
        style: style
      })
    }
    
  } catch (error) {
    console.error('API error:', error)
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Internal server error' },
      { status: 400 }
    )
  }
}

// Handle CORS for development
export async function OPTIONS() {
  return new NextResponse(null, {
    status: 200,
    headers: {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type',
    },
  })
}