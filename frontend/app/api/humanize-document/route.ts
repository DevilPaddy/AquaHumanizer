import { NextRequest, NextResponse } from 'next/server'

// Configuration
const HF_SPACE_URL = process.env.HF_SPACE_URL 
const REQUEST_TIMEOUT = 300000 // 5 minutes for long documents and cold starts

async function callHFSpaceDocument(
  file: File, 
  style: string = 'neutral',
  outputFormat: string = 'json'
): Promise<any> {
  const controller = new AbortController()
  const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT)
  
  console.log(`🚀 Starting HF Space request: ${file.name} (${file.size} bytes), style: ${style}, format: ${outputFormat}`)
  const startTime = Date.now()
  
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
    const responseTime = Date.now() - startTime
    
    if (!response.ok) {
      console.error(`❌ HF Space error: ${response.status} (${responseTime}ms)`)
      throw new Error(`HF Space error: ${response.status}`)
    }
    
    console.log(`✅ HF Space response received: ${response.status} (${responseTime}ms)`)
    
    // Handle different response types
    if (outputFormat === 'json') {
      const jsonData = await response.json()
      console.log(`📄 JSON response: ${jsonData.output?.length || 0} characters`)
      return jsonData
    } else {
      // Return blob for file downloads with validation
      const blob = await response.blob()
      console.log(`📁 Blob received: ${blob.size} bytes, type: ${blob.type}`)
      
      // Debug: Log the actual response content for small blobs
      if (blob.size < 1000) {
        const text = await blob.text()
        console.log(`🔍 Small blob content (first 500 chars):`, text.substring(0, 500))
        
        // If it's JSON but we expected a file, there might be an error in the response
        if (outputFormat !== 'json' && blob.type.includes('application/json')) {
          try {
            const errorData = JSON.parse(text)
            console.error('❌ HF Space returned JSON error:', errorData)
            throw new Error(errorData.detail || errorData.error || 'HF Space returned JSON instead of file')
          } catch (parseError) {
            console.error('❌ HF Space returned unexpected JSON response')
            throw new Error('HF Space returned JSON instead of file format')
          }
        }
        
        // Recreate blob for further processing
        const newBlob = new Blob([text], { type: blob.type })
        return await processBlob(newBlob, response, outputFormat)
      }
      
      return await processBlob(blob, response, outputFormat)
    }
    
  } catch (error) {
    clearTimeout(timeoutId)
    const errorTime = Date.now() - startTime
    
    if (error instanceof Error && error.name === 'AbortError') {
      console.error(`❌ HF Space request timed out after ${errorTime}ms (${(errorTime/1000).toFixed(1)}s)`)
      throw new Error(`Request timed out after ${(errorTime/1000).toFixed(1)} seconds. The HF Space might be cold starting or overloaded. Please try again.`)
    }
    
    console.error(`❌ HF Space request failed (${errorTime}ms):`, error)
    throw error
  }
}

async function processBlob(blob: Blob, response: Response, outputFormat: string) {
  // Validate blob size (> 0 bytes)
  if (blob.size === 0) {
    console.error('❌ Blob validation failed: empty file')
    throw new Error('Received empty file from backend')
  }
  
  // Validate content-type for PDF and DOCX
  const contentType = response.headers.get('content-type') || ''
  if (outputFormat === 'pdf' && !contentType.includes('application/pdf')) {
    console.warn(`⚠️ Expected PDF content-type, got: ${contentType}`)
  }
  if (outputFormat === 'docx' && !contentType.includes('application/vnd.openxmlformats-officedocument.wordprocessingml.document')) {
    console.warn(`⚠️ Expected DOCX content-type, got: ${contentType}`)
  }
  
  // Additional blob validation by checking first few bytes
  const arrayBuffer = await blob.arrayBuffer()
  const uint8Array = new Uint8Array(arrayBuffer)
  
  if (outputFormat === 'pdf') {
    // Check PDF header (%PDF)
    const pdfHeader = String.fromCharCode(...Array.from(uint8Array.slice(0, 4)))
    if (!pdfHeader.startsWith('%PDF')) {
      console.error('❌ PDF validation failed: missing PDF header')
      throw new Error('Invalid PDF file - missing PDF header')
    }
    console.log('✅ PDF header validation passed')
  } else if (outputFormat === 'docx') {
    // Check DOCX header (ZIP format - PK)
    const zipHeader = String.fromCharCode(...Array.from(uint8Array.slice(0, 2)))
    if (zipHeader !== 'PK') {
      console.error('❌ DOCX validation failed: missing ZIP header')
      throw new Error('Invalid DOCX file - missing ZIP header')
    }
    console.log('✅ DOCX header validation passed')
  }
  
  console.log(`✅ Blob validation passed: ${blob.size} bytes, type: ${contentType}`)
  
  return {
    blob: new Blob([arrayBuffer], { type: contentType }),
    headers: Object.fromEntries(response.headers.entries())
  }
}

export async function POST(request: NextRequest) {
  const startTime = Date.now()
  
  try {
    const formData = await request.formData()
    const file = formData.get('file') as File
    const style = (formData.get('style') as string) || 'neutral'
    const outputFormat = (formData.get('output_format') as string) || 'json'
    
    console.log(`📥 Frontend API request: ${file?.name || 'unknown'} (${file?.size || 0} bytes), style: ${style}, format: ${outputFormat}`)
    
    if (!file) {
      console.error('❌ No file provided in request')
      return NextResponse.json(
        { error: 'No file provided' },
        { status: 400 }
      )
    }
    
    // Validate file type
    const allowedTypes = ['.txt', '.docx']
    const fileExtension = file.name.toLowerCase().substring(file.name.lastIndexOf('.'))
    
    if (!allowedTypes.some(type => fileExtension.endsWith(type))) {
      console.error(`❌ Invalid file type: ${fileExtension}`)
      return NextResponse.json(
        { error: 'Unsupported file type. Please upload .txt or .docx files.' },
        { status: 400 }
      )
    }
    
    console.log(`✅ File validation passed: ${fileExtension} file`)
    
    try {
      const result = await callHFSpaceDocument(file, style, outputFormat)
      const processingTime = Date.now() - startTime
      
      if (outputFormat === 'json') {
        console.log(`📤 Returning JSON response (${processingTime}ms)`)
        return NextResponse.json(result)
      } else {
        // Return file download with enhanced header preservation
        const headers = new Headers()
        
        // Set proper content type based on format with fallbacks
        if (outputFormat === 'pdf') {
          headers.set('Content-Type', result.headers['content-type'] || 'application/pdf')
          headers.set('Content-Disposition', result.headers['content-disposition'] || 'attachment; filename="paraphrased.pdf"')
        } else if (outputFormat === 'docx') {
          headers.set('Content-Type', result.headers['content-type'] || 'application/vnd.openxmlformats-officedocument.wordprocessingml.document')
          headers.set('Content-Disposition', result.headers['content-disposition'] || 'attachment; filename="paraphrased.docx"')
        }
        
        // Preserve additional backend headers
        if (result.headers['content-length']) {
          headers.set('Content-Length', result.headers['content-length'])
        }
        if (result.headers['cache-control']) {
          headers.set('Cache-Control', result.headers['cache-control'])
        } else {
          headers.set('Cache-Control', 'no-cache, no-store, must-revalidate')
        }
        
        console.log(`📤 Returning ${outputFormat.toUpperCase()} file: ${result.blob.size} bytes (${processingTime}ms)`)
        console.log(`📋 Headers:`, Object.fromEntries(headers.entries()))
        
        return new NextResponse(result.blob, { headers })
      }
      
    } catch (error) {
      const errorTime = Date.now() - startTime
      console.error(`❌ Document processing error (${errorTime}ms):`, error)
      return NextResponse.json(
        { error: 'Failed to process document. Please try again.' },
        { status: 500 }
      )
    }
    
  } catch (error) {
    const errorTime = Date.now() - startTime
    console.error(`❌ Frontend API error (${errorTime}ms):`, error)
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