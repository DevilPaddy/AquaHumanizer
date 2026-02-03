import { NextRequest, NextResponse } from 'next/server'

// Configuration
const HF_SPACE_URL = process.env.HF_SPACE_URL || 'https://devilseye2004-aq-humanizer.hf.space'

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData()
    const file = formData.get('file') as File
    const style = (formData.get('style') as string) || 'neutral'
    
    if (!file) {
      return NextResponse.json(
        { error: 'No file provided' },
        { status: 400 }
      )
    }
    
    console.log(`Starting streaming processing: ${file.name}, style: ${style}`)
    
    // Create streaming response
    const stream = new ReadableStream({
      async start(controller) {
        try {
          // Call HF Space streaming endpoint
          const streamFormData = new FormData()
          streamFormData.append('file', file)
          streamFormData.append('style', style)
          
          const response = await fetch(`${HF_SPACE_URL}/humanize-document-stream`, {
            method: 'POST',
            body: streamFormData,
          })
          
          if (!response.ok) {
            throw new Error(`HF Space error: ${response.status}`)
          }
          
          const reader = response.body?.getReader()
          if (!reader) {
            throw new Error('No response body')
          }
          
          const decoder = new TextDecoder()
          
          while (true) {
            const { done, value } = await reader.read()
            
            if (done) {
              break
            }
            
            // Forward the chunk to the client
            const chunk = decoder.decode(value, { stream: true })
            controller.enqueue(new TextEncoder().encode(chunk))
          }
          
          controller.close()
          
        } catch (error) {
          console.error('Streaming error:', error)
          const errorMessage = JSON.stringify({
            type: 'error',
            message: error instanceof Error ? error.message : 'Streaming failed'
          }) + '\n'
          
          controller.enqueue(new TextEncoder().encode(errorMessage))
          controller.close()
        }
      }
    })
    
    return new NextResponse(stream, {
      headers: {
        'Content-Type': 'application/json',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
      },
    })
    
  } catch (error) {
    console.error('Stream setup error:', error)
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