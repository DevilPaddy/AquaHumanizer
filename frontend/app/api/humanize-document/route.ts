import { NextRequest, NextResponse } from 'next/server'

interface DocumentResponse {
  output: string
  style: string
}

// Configuration
const HF_SPACE_URL = process.env.HF_SPACE_URL || 'https://devilseye2004-aq-humanizer.hf.space'
const REQUEST_TIMEOUT = 120000 // 2 minutes for long documents

async function callHFSpaceDocument(
  file: File, 
  style: string = 'neutral',
  outputFormat: string = 'json'
): Promise<any> {
  const controller = new AbortController()
  const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT)
  
  try {
    const formData = new FormData()
    formData.append('file', file)
    formData.append('style', style)
    formData.append('output_format', outputFormat)
    
    const response = await fetch(`${HF_SPACE_URL}/humanize-document`, {
      method: 'POST',
      body: formData,
      signal: controller.signal,
    })
    
    clearTimeout(timeoutId)
    
    if (!response.ok) {
      throw new Error(`HF Space error: ${response.status}`)
    }
    
    // Handle different response types
    if (outputFormat === 'json') {
      return await response.json()
    } else {
      // Return blob for file downloads
      return {
        blob: await response.blob(),
        headers: Object.fromEntries(response.headers.entries())
      }
    }
    
  } catch (error) {
    clearTimeout(timeoutId)
    console.error('Error calling HF Space document API:', error)
    throw error
  }
}

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData()
    const file = formData.get('file') as File
    const style = (formData.get('style') as string) || 'neutral'
    const outputFormat = (formData.get('output_format') as string) || 'json'
    
    if (!file) {
      return NextResponse.json(
        { error: 'No file provided' },
        { status: 400 }
      )
    }
    
    // Validate file type
    const allowedTypes = ['.txt', '.docx']
    const fileExtension = file.name.toLowerCase().substring(file.name.lastIndexOf('.'))
    
    if (!allowedTypes.some(type => fileExtension.endsWith(type))) {
      return NextResponse.json(
        { error: 'Unsupported file type. Please upload .txt or .docx files.' },
        { status: 400 }
      )
    }
    
    console.log(`Processing document: ${file.name}, style: ${style}, format: ${outputFormat}`)
    
    try {
      const result = await callHFSpaceDocument(file, style, outputFormat)
      
      if (outputFormat === 'json') {
        return NextResponse.json(result)
      } else {
        // Return file download
        const headers = new Headers()
        if (result.headers['content-disposition']) {
          headers.set('Content-Disposition', result.headers['content-disposition'])
        }
        if (result.headers['content-type']) {
          headers.set('Content-Type', result.headers['content-type'])
        }
        
        return new NextResponse(result.blob, { headers })
      }
      
    } catch (error) {
      console.error('Document processing error:', error)
      return NextResponse.json(
        { error: 'Failed to process document. Please try again.' },
        { status: 500 }
      )
    }
    
  } catch (error) {
    console.error('API error:', error)
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Internal server error' },
      { status: 500 }
    )
  }
}

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