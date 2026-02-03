import { NextResponse } from 'next/server'

export async function GET() {
  const HF_SPACE_URL = process.env.HF_SPACE_URL || 'https://devilseye2004-aq-humanizer.hf.space'
  
  return NextResponse.json({
    status: 'healthy',
    service: 'AquaHumanizer Serverless API',
    hf_space_configured: !!HF_SPACE_URL,
    hf_space_url: HF_SPACE_URL,
    cache_enabled: true,
    timestamp: new Date().toISOString()
  })
}