/*
 * AquaHumanizer Pro - Frontend Application
 * Copyright (c) 2026 AquilaStudios
 * Licensed under the MIT License
 */

'use client'

import { useState, useRef, useEffect } from 'react'
import { 
  SparklesIcon, 
  ClipboardDocumentIcon, 
  ArrowDownTrayIcon,
  CheckIcon,
  ExclamationTriangleIcon,
  DocumentTextIcon,
  Cog6ToothIcon,
  CloudArrowUpIcon,
  DocumentIcon
} from '@heroicons/react/24/outline'

type ParaphraseStyle = 'neutral' | 'formal' | 'ats' | 'bullets'
type OutputFormat = 'json' | 'docx' | 'pdf'

interface ToastMessage {
  id: number
  message: string
  type: 'success' | 'error' | 'info'
}

interface StreamChunk {
  type: 'progress' | 'chunk' | 'done' | 'error'
  current?: number
  total?: number
  message?: string
  content?: string
  output?: string
  style?: string
}

export default function Home() {
  const [inputText, setInputText] = useState('')
  const [outputText, setOutputText] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [style, setStyle] = useState<ParaphraseStyle>('neutral')
  const [progress, setProgress] = useState(0)
  const [progressMessage, setProgressMessage] = useState('')
  const [processingTime, setProcessingTime] = useState<number | null>(null)
  const [copied, setCopied] = useState(false)
  const [showComparison, setShowComparison] = useState(false)
  const [toasts, setToasts] = useState<ToastMessage[]>([])
  
  // Document processing
  const [activeTab, setActiveTab] = useState<'text' | 'document'>('text')
  const [documentFile, setDocumentFile] = useState<File | null>(null)
  const [outputFormat, setOutputFormat] = useState<OutputFormat>('json')
  const [streamingProgress, setStreamingProgress] = useState<string>('')
  const [streamingResults, setStreamingResults] = useState<string[]>([])
  const [isDragOver, setIsDragOver] = useState(false)

  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  // Character and word counts
  const charCount = inputText.length
  const wordCount = inputText.trim() ? inputText.trim().split(/\s+/).length : 0
  const maxChars = 10000

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`
    }
  }, [inputText])

  // Prevent default drag behavior on the entire document
  useEffect(() => {
    const preventDefault = (e: DragEvent) => {
      e.preventDefault()
    }

    const handleDocumentDrop = (e: DragEvent) => {
      e.preventDefault()
      // Only handle drops if we're on the document tab and not over the drop zone
      if (activeTab === 'document' && !documentFile) {
        const files = e.dataTransfer?.files
        if (files && files.length > 0) {
          processUploadedFile(files[0])
        }
      }
    }

    // Prevent default drag behavior
    document.addEventListener('dragover', preventDefault)
    document.addEventListener('dragenter', preventDefault)
    document.addEventListener('drop', handleDocumentDrop)

    return () => {
      document.removeEventListener('dragover', preventDefault)
      document.removeEventListener('dragenter', preventDefault)
      document.removeEventListener('drop', handleDocumentDrop)
    }
  }, [activeTab, documentFile])

  // Toast management
  const addToast = (message: string, type: 'success' | 'error' | 'info') => {
    const id = Date.now()
    setToasts(prev => [...prev, { id, message, type }])
    setTimeout(() => {
      setToasts(prev => prev.filter(toast => toast.id !== id))
    }, 4000)
  }

  const handleHumanize = async () => {
    if (!inputText.trim()) {
      addToast('Please enter some text to humanize', 'error')
      return
    }

    if (wordCount < 5) {
      addToast('Text must contain at least 5 words', 'error')
      return
    }

    setLoading(true)
    setError(null)
    setOutputText('')
    setProgress(0)
    setProgressMessage('Analyzing text...')
    const startTime = Date.now()

    // Simulate progress for better UX
    const progressInterval = setInterval(() => {
      setProgress(prev => {
        if (prev < 90) return prev + Math.random() * 10
        return prev
      })
    }, 200)

    try {
      setProgressMessage('Processing with AI model...')
      
      const response = await fetch('/api/humanize-text', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: inputText, style }),
      })

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ error: 'Unknown error' }))
        throw new Error(errorData.error || `HTTP ${response.status}`)
      }

      const data = await response.json()
      setOutputText(data.output || '')
      setProgress(100)
      setProgressMessage('Complete!')
      
      const endTime = Date.now()
      setProcessingTime((endTime - startTime) / 1000)
      
      addToast(data.cached ? 'Retrieved from cache' : 'Text humanized successfully!', 'success')
      
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to humanize text')
      addToast('Failed to humanize text', 'error')
      setProgress(0)
      setProgressMessage('')
    } finally {
      clearInterval(progressInterval)
      setLoading(false)
      
      // Reset progress after delay
      setTimeout(() => {
        setProgress(0)
        setProgressMessage('')
      }, 2000)
    }
  }

  const handleDocumentUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file) {
      processUploadedFile(file)
    }
  }

  const processUploadedFile = (file: File) => {
    // Validate file type
    const allowedTypes = ['.txt', '.docx']
    const fileExtension = file.name.toLowerCase().substring(file.name.lastIndexOf('.'))
    
    if (!allowedTypes.some(type => fileExtension.endsWith(type))) {
      addToast('Please upload .txt or .docx files only', 'error')
      return
    }
    
    if (file.size > 10 * 1024 * 1024) { // 10MB limit
      addToast('File size must be less than 10MB', 'error')
      return
    }
    
    setDocumentFile(file)
    addToast(`File "${file.name}" uploaded successfully`, 'success')
  }

  const handleDragOver = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault()
    event.stopPropagation()
    setIsDragOver(true)
  }

  const handleDragLeave = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault()
    event.stopPropagation()
    setIsDragOver(false)
  }

  const handleDrop = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault()
    event.stopPropagation()
    setIsDragOver(false)

    const files = event.dataTransfer.files
    if (files && files.length > 0) {
      const file = files[0]
      processUploadedFile(file)
    }
  }

  const handleDocumentProcess = async () => {
    if (!documentFile) {
      addToast('Please upload a document first', 'error')
      return
    }

    // Freeze the current outputFormat to prevent race conditions
    const currentOutputFormat = outputFormat

    setLoading(true)
    setError(null)
    setOutputText('')
    setProgress(0)
    setProgressMessage('Processing document...')
    setStreamingProgress('')
    setStreamingResults([])
    const startTime = Date.now()

    console.log(`🚀 Starting document processing: ${documentFile.name} (${documentFile.size} bytes), style: ${style}, format: ${currentOutputFormat}`)

    try {
      const formData = new FormData()
      formData.append('file', documentFile)
      formData.append('style', style)
      formData.append('output_format', currentOutputFormat)

      if (currentOutputFormat === 'json') {
        // Regular processing for JSON output
        console.log('📄 Processing for JSON output')
        const response = await fetch('/api/humanize-document', {
          method: 'POST',
          body: formData,
        })

        if (!response.ok) {
          const errorText = await response.text()
          console.error(`❌ API error: ${response.status} - ${errorText}`)
          throw new Error(errorText || `HTTP ${response.status}`)
        }

        // Check Content-Type before parsing JSON
        const contentType = response.headers.get('content-type') || ''
        if (!contentType.includes('application/json')) {
          const text = await response.text()
          console.error('❌ Expected JSON, received:', contentType, text.slice(0, 200))
          throw new Error('Server returned non-JSON response')
        }

        const data = await response.json()
        setOutputText(data.output || '')
        setProgress(100)
        setProgressMessage('Document processed!')
        
        const endTime = Date.now()
        const processingTime = (endTime - startTime) / 1000
        setProcessingTime(processingTime)
        
        console.log(`✅ JSON processing completed: ${data.output?.length || 0} characters (${processingTime.toFixed(2)}s)`)
        addToast('Document processed successfully!', 'success')
      } else {
        // File download for DOCX/PDF
        console.log(`📁 Processing for ${currentOutputFormat.toUpperCase()} download`)
        const response = await fetch('/api/humanize-document', {
          method: 'POST',
          body: formData,
        })

        if (!response.ok) {
          const errorText = await response.text()
          console.error(`❌ Download API error: ${response.status} - ${errorText}`)
          throw new Error(errorText || `HTTP ${response.status}`)
        }

        // Download the file with enhanced validation
        const blob = await response.blob()
        console.log(`📦 Blob received: ${blob.size} bytes, type: ${blob.type}`)
        
        // Validate blob before download
        if (blob.size === 0) {
          console.error('❌ Blob validation failed: empty file')
          throw new Error('Received empty file from server')
        }
        
        // Additional validation for file format
        const arrayBuffer = await blob.arrayBuffer()
        const uint8Array = new Uint8Array(arrayBuffer)
        
        if (currentOutputFormat === 'pdf') {
          // Check PDF header
          const pdfHeader = String.fromCharCode(...Array.from(uint8Array.slice(0, 4)))
          if (!pdfHeader.startsWith('%PDF')) {
            console.error('❌ PDF validation failed: invalid header')
            throw new Error('Invalid PDF file received')
          }
          console.log('✅ PDF validation passed')
        } else if (currentOutputFormat === 'docx') {
          // Check DOCX header (ZIP format)
          const zipHeader = String.fromCharCode(...Array.from(uint8Array.slice(0, 2)))
          if (zipHeader !== 'PK') {
            console.error('❌ DOCX validation failed: invalid header')
            throw new Error('Invalid DOCX file received')
          }
          console.log('✅ DOCX validation passed')
        }
        
        console.log(`✅ File validation passed: ${blob.size} bytes`)
        
        // Create download with improved filename generation
        const url = URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.href = url
        
        // Generate proper filename with extension
        const baseFilename = documentFile.name.split('.')[0] || 'humanized'
        const timestamp = new Date().toISOString().slice(0, 10) // YYYY-MM-DD
        const filename = `${baseFilename}-${style}-${timestamp}.${currentOutputFormat}`
        a.download = filename
        
        console.log(`💾 Initiating download: ${filename}`)
        
        document.body.appendChild(a)
        a.click()
        
        // Clean up URL after download
        setTimeout(() => {
          URL.revokeObjectURL(url)
          document.body.removeChild(a)
          console.log('🧹 Download cleanup completed')
        }, 100)
        
        setProgress(100)
        setProgressMessage('File downloaded!')
        
        const endTime = Date.now()
        const processingTime = (endTime - startTime) / 1000
        setProcessingTime(processingTime)
        
        console.log(`✅ Download completed: ${filename} (${processingTime.toFixed(2)}s)`)
        addToast(`${currentOutputFormat.toUpperCase()} file downloaded successfully!`, 'success')
      }
      
    } catch (err) {
      const errorTime = Date.now() - startTime
      const errorMessage = err instanceof Error ? err.message : 'Failed to process document'
      console.error(`❌ Document processing failed (${(errorTime / 1000).toFixed(2)}s):`, err)
      
      setError(errorMessage)
      addToast('Failed to process document', 'error')
      setProgress(0)
      setProgressMessage('')
    } finally {
      setLoading(false)
      
      // Reset progress after delay
      setTimeout(() => {
        setProgress(0)
        setProgressMessage('')
      }, 2000)
    }
  }

  const handleStreamingProcess = async () => {
    if (!documentFile) {
      addToast('Please upload a document first', 'error')
      return
    }

    setLoading(true)
    setError(null)
    setOutputText('')
    setStreamingProgress('')
    setStreamingResults([])
    const startTime = Date.now()

    try {
      const formData = new FormData()
      formData.append('file', documentFile)
      formData.append('style', style)

      const response = await fetch('/api/humanize-stream', {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`)
      }

      const reader = response.body?.getReader()
      if (!reader) {
        throw new Error('No response body')
      }

      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''

        for (const line of lines) {
          if (line.trim()) {
            try {
              const chunk: StreamChunk = JSON.parse(line)
              
              if (chunk.type === 'progress') {
                setStreamingProgress(chunk.message || '')
                if (chunk.current && chunk.total) {
                  setProgress((chunk.current / chunk.total) * 100)
                }
              } else if (chunk.type === 'chunk') {
                setStreamingResults(prev => [...prev, chunk.content || ''])
              } else if (chunk.type === 'done') {
                setOutputText(chunk.output || streamingResults.join(''))
                setProgress(100)
                setStreamingProgress('Complete!')
                
                const endTime = Date.now()
                setProcessingTime((endTime - startTime) / 1000)
                addToast('Document processed with streaming!', 'success')
              } else if (chunk.type === 'error') {
                throw new Error(chunk.message || 'Streaming error')
              }
            } catch (parseError) {
              console.warn('Failed to parse chunk:', line)
            }
          }
        }
      }
      
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to process document')
      addToast('Failed to process document', 'error')
      setProgress(0)
      setStreamingProgress('')
    } finally {
      setLoading(false)
      
      // Reset progress after delay
      setTimeout(() => {
        setProgress(0)
        setStreamingProgress('')
      }, 2000)
    }
  }

  const handleCopy = async () => {
    if (!outputText) return
    
    try {
      await navigator.clipboard.writeText(outputText)
      setCopied(true)
      addToast('Copied to clipboard!', 'success')
      setTimeout(() => setCopied(false), 2000)
    } catch (err) {
      addToast('Failed to copy text', 'error')
    }
  }

  const handleDownloadTxt = () => {
    if (!outputText) return
    
    const blob = new Blob([outputText], { type: 'text/plain' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `humanized-${style}.txt`
    document.body.appendChild(a)
    a.click()
    URL.revokeObjectURL(url)
    document.body.removeChild(a)
    
    addToast('File downloaded!', 'success')
  }

  const handleClear = () => {
    setInputText('')
    setOutputText('')
    setError(null)
    setProgress(0)
    setProgressMessage('')
    setProcessingTime(null)
    setDocumentFile(null)
    setStreamingProgress('')
    setStreamingResults([])
    setIsDragOver(false)
    if (textareaRef.current) {
      textareaRef.current.focus()
    }
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      e.preventDefault()
      if (activeTab === 'text') {
        handleHumanize()
      } else {
        handleDocumentProcess()
      }
    }
  }

  const exampleTexts = [
    "I worked on backend systems and made them faster and more reliable for users.",
    "The company implemented new policies to improve employee satisfaction and productivity.",
    "Our team developed a mobile application that helps users track their daily activities."
  ]

  const handleExampleClick = (text: string) => {
    setInputText(text)
    setActiveTab('text')
    if (textareaRef.current) {
      textareaRef.current.focus()
    }
  }

  return (
    <div className="min-h-screen bg-dark-950 text-dark-50">
      {/* Header */}
      <header className="border-b border-dark-700 bg-dark-950/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-4xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-8 h-8 bg-gradient-to-br from-primary-600 to-success-600 rounded-lg flex items-center justify-center">
                <SparklesIcon className="w-5 h-5 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-semibold text-dark-50">AquaHumanizer Pro</h1>
                <p className="text-sm text-dark-400">Professional AI Text Enhancement</p>
              </div>
            </div>
            <div className="flex items-center space-x-2">
              <span className="text-xs text-dark-400 bg-dark-900 px-2 py-1 rounded-full">
                v2.0
              </span>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-4xl mx-auto px-6 py-8">
        {/* Hero Section */}
        <div className="text-center mb-12">
          <h2 className="text-3xl font-bold text-dark-50 mb-4">
            Transform AI Text into Professional Content
          </h2>
          <p className="text-lg text-dark-400 max-w-2xl mx-auto">
            Enhance your AI-generated content with professional styles, ATS optimization, 
            and bullet-point formatting for resumes and business documents.
          </p>
        </div>

        {/* Tab Navigation */}
        <div className="flex space-x-1 mb-8 bg-dark-900 p-1 rounded-lg">
          <button
            onClick={() => setActiveTab('text')}
            className={`flex-1 flex items-center justify-center space-x-2 px-4 py-3 rounded-md transition-all duration-200 ${
              activeTab === 'text'
                ? 'bg-primary-600 text-white shadow-lg'
                : 'text-dark-400 hover:text-dark-50 hover:bg-dark-800'
            }`}
          >
            <DocumentTextIcon className="w-5 h-5" />
            <span>Text Input</span>
          </button>
          <button
            onClick={() => setActiveTab('document')}
            className={`flex-1 flex items-center justify-center space-x-2 px-4 py-3 rounded-md transition-all duration-200 ${
              activeTab === 'document'
                ? 'bg-primary-600 text-white shadow-lg'
                : 'text-dark-400 hover:text-dark-50 hover:bg-dark-800'
            }`}
          >
            <CloudArrowUpIcon className="w-5 h-5" />
            <span>Document Upload</span>
          </button>
        </div>

        {/* Style Selection */}
        <div className="card p-6 mb-8">
          <div className="flex items-center space-x-2 mb-4">
            <Cog6ToothIcon className="w-5 h-5 text-primary-600" />
            <h3 className="text-lg font-semibold text-dark-50">Enhancement Style</h3>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
            {[
              { value: 'neutral', label: 'Neutral', desc: 'General enhancement' },
              { value: 'formal', label: 'Formal', desc: 'Business tone' },
              { value: 'ats', label: 'ATS Resume', desc: 'Job-optimized' },
              { value: 'bullets', label: 'Bullet Points', desc: 'Structured format' }
            ].map((option) => (
              <button
                key={option.value}
                onClick={() => setStyle(option.value as ParaphraseStyle)}
                className={`p-4 rounded-lg border transition-all duration-200 text-left ${
                  style === option.value
                    ? 'border-primary-600 bg-primary-600/10 shadow-lg shadow-primary-600/20'
                    : 'border-dark-700 bg-dark-800 hover:border-dark-600 hover:bg-dark-700'
                }`}
              >
                <div className="font-medium text-dark-50 mb-1">{option.label}</div>
                <div className="text-sm text-dark-400">{option.desc}</div>
              </button>
            ))}
          </div>
        </div>

        {/* Input Section */}
        {activeTab === 'text' ? (
          <div className="card p-6 mb-8">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center space-x-2">
                <DocumentTextIcon className="w-5 h-5 text-primary-600" />
                <h3 className="text-lg font-semibold text-dark-50">Input Text</h3>
              </div>
              <div className="flex items-center space-x-4">
                <span className={`text-sm ${charCount > maxChars * 0.9 ? 'text-yellow-500' : 'text-dark-400'}`}>
                  {charCount.toLocaleString()}/{maxChars.toLocaleString()}
                </span>
                <span className="text-sm text-dark-400">
                  {wordCount} words
                </span>
                {inputText && (
                  <button
                    onClick={handleClear}
                    className="text-sm text-dark-400 hover:text-dark-50 transition-colors"
                  >
                    Clear
                  </button>
                )}
              </div>
            </div>

            <div className="relative">
              <textarea
                ref={textareaRef}
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Paste your AI-generated text here... (Ctrl+Enter to process)"
                className="input-field w-full min-h-[200px] resize-none"
                maxLength={maxChars}
                disabled={loading}
              />
              
              {/* Progress Bar */}
              {(loading || progress > 0) && (
                <div className="mt-4">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm text-dark-400">{progressMessage}</span>
                    <span className="text-sm text-primary-600">{Math.round(progress)}%</span>
                  </div>
                  <div className="w-full bg-dark-700 rounded-full h-2">
                    <div 
                      className="bg-gradient-to-r from-primary-600 to-success-600 h-2 rounded-full transition-all duration-300 ease-out"
                      style={{ width: `${progress}%` }}
                    />
                  </div>
                </div>
              )}
            </div>

            {/* Action Buttons */}
            <div className="flex items-center justify-between mt-6">
              <div className="flex items-center space-x-3">
                <button
                  onClick={handleHumanize}
                  disabled={loading || !inputText.trim() || wordCount < 5}
                  className="btn-primary disabled:from-dark-700 disabled:to-dark-700 disabled:text-dark-400 disabled:scale-100 disabled:cursor-not-allowed"
                >
                  {loading ? (
                    <>
                      <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                      <span>Processing...</span>
                    </>
                  ) : (
                    <>
                      <SparklesIcon className="w-4 h-4" />
                      <span>Humanize Text</span>
                    </>
                  )}
                </button>
                
                <span className="text-xs text-dark-500">
                  Ctrl+Enter
                </span>
              </div>

              {processingTime && (
                <span className="text-sm text-dark-400">
                  Processed in {processingTime.toFixed(1)}s
                </span>
              )}
            </div>

            {/* Example Texts */}
            {!inputText && (
              <div className="mt-6 pt-6 border-t border-dark-700">
                <h4 className="text-sm font-medium text-dark-50 mb-3">Try these examples:</h4>
                <div className="space-y-2">
                  {exampleTexts.map((text, index) => (
                    <button
                      key={index}
                      onClick={() => handleExampleClick(text)}
                      className="block w-full text-left p-3 bg-dark-800 hover:bg-dark-700 border border-dark-700 hover:border-dark-600 rounded-lg text-sm text-dark-400 hover:text-dark-50 transition-all duration-200"
                    >
                      {text}
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>
        ) : (
          <div className="card p-6 mb-8">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center space-x-2">
                <CloudArrowUpIcon className="w-5 h-5 text-primary-600" />
                <h3 className="text-lg font-semibold text-dark-50">Document Upload</h3>
              </div>
              {documentFile && (
                <button
                  onClick={handleClear}
                  className="text-sm text-dark-400 hover:text-dark-50 transition-colors"
                >
                  Clear
                </button>
              )}
            </div>

            {/* File Upload Area */}
            <div className="mb-6">
              <input
                ref={fileInputRef}
                type="file"
                accept=".txt,.docx"
                onChange={handleDocumentUpload}
                className="hidden"
              />
              
              {!documentFile ? (
                <div
                  onClick={() => fileInputRef.current?.click()}
                  onDragOver={handleDragOver}
                  onDragLeave={handleDragLeave}
                  onDrop={handleDrop}
                  className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-all duration-200 ${
                    isDragOver
                      ? 'border-primary-500 bg-primary-500/10 scale-105'
                      : 'border-dark-600 hover:border-primary-600 hover:bg-primary-600/5'
                  }`}
                >
                  <CloudArrowUpIcon className={`w-12 h-12 mx-auto mb-4 transition-colors duration-200 ${
                    isDragOver ? 'text-primary-500' : 'text-dark-400'
                  }`} />
                  <p className="text-dark-50 font-medium mb-2">
                    {isDragOver ? 'Drop your document here' : 'Upload your document'}
                  </p>
                  <p className="text-sm text-dark-400 mb-4">
                    {isDragOver 
                      ? 'Release to upload your file' 
                      : 'Drag and drop or click to select .txt or .docx files'
                    }
                  </p>
                  <p className="text-xs text-dark-500">
                    Maximum file size: 10MB
                  </p>
                </div>
              ) : (
                <div className="bg-dark-800 border border-dark-600 rounded-lg p-4 flex items-center space-x-3">
                  <DocumentIcon className="w-8 h-8 text-primary-600 flex-shrink-0" />
                  <div className="flex-1 min-w-0">
                    <p className="text-dark-50 font-medium truncate">{documentFile.name}</p>
                    <p className="text-sm text-dark-400">
                      {(documentFile.size / 1024 / 1024).toFixed(2)} MB
                    </p>
                  </div>
                  <button
                    onClick={() => fileInputRef.current?.click()}
                    className="text-sm text-primary-600 hover:text-primary-500 transition-colors"
                  >
                    Change
                  </button>
                </div>
              )}
            </div>

            {/* Output Format Selection */}
            <div className="mb-6">
              <h4 className="text-sm font-medium text-dark-50 mb-3">Output Format</h4>
              <div className="grid grid-cols-3 gap-3">
                {[
                  { value: 'json', label: 'View Online', desc: 'Display in browser' },
                  { value: 'docx', label: 'Download DOCX', desc: 'Word document' },
                  { value: 'pdf', label: 'Download PDF', desc: 'PDF document' }
                ].map((option) => (
                  <button
                    key={option.value}
                    onClick={() => setOutputFormat(option.value as OutputFormat)}
                    className={`p-3 rounded-lg border transition-all duration-200 text-left ${
                      outputFormat === option.value
                        ? 'border-primary-600 bg-primary-600/10'
                        : 'border-dark-700 bg-dark-800 hover:border-dark-600'
                    }`}
                  >
                    <div className="font-medium text-dark-50 text-sm">{option.label}</div>
                    <div className="text-xs text-dark-400">{option.desc}</div>
                  </button>
                ))}
              </div>
            </div>

            {/* Progress Bar */}
            {(loading || progress > 0 || streamingProgress) && (
              <div className="mb-6">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm text-dark-400">
                    {streamingProgress || progressMessage}
                  </span>
                  <span className="text-sm text-primary-600">{Math.round(progress)}%</span>
                </div>
                <div className="w-full bg-dark-700 rounded-full h-2">
                  <div 
                    className="bg-gradient-to-r from-primary-600 to-success-600 h-2 rounded-full transition-all duration-300 ease-out"
                    style={{ width: `${progress}%` }}
                  />
                </div>
              </div>
            )}

            {/* Action Buttons */}
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-3">
                <button
                  onClick={handleDocumentProcess}
                  disabled={loading || !documentFile}
                  className="btn-primary disabled:from-dark-700 disabled:to-dark-700 disabled:text-dark-400 disabled:scale-100 disabled:cursor-not-allowed"
                >
                  {loading ? (
                    <>
                      <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                      <span>Processing...</span>
                    </>
                  ) : (
                    <>
                      <SparklesIcon className="w-4 h-4" />
                      <span>Process Document</span>
                    </>
                  )}
                </button>
                
                <button
                  onClick={handleStreamingProcess}
                  disabled={loading || !documentFile}
                  className="btn-secondary disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  <span>Stream Process</span>
                </button>
                
                <span className="text-xs text-dark-500">
                  Ctrl+Enter
                </span>
              </div>

              {processingTime && (
                <span className="text-sm text-dark-400">
                  Processed in {processingTime.toFixed(1)}s
                </span>
              )}
            </div>
          </div>
        )}

        {/* Output Section */}
        {outputText && (
          <div className="card p-6 mb-8 animate-fadeIn">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center space-x-2">
                <CheckIcon className="w-5 h-5 text-success-600" />
                <h3 className="text-lg font-semibold text-dark-50">Enhanced Text</h3>
                <span className="text-xs bg-success-600/20 text-success-400 px-2 py-1 rounded-full">
                  {style.toUpperCase()}
                </span>
              </div>
              
              <div className="flex items-center space-x-2">
                <button
                  onClick={() => setShowComparison(!showComparison)}
                  className="text-sm text-dark-400 hover:text-dark-50 transition-colors"
                >
                  {showComparison ? 'Hide' : 'Show'} Comparison
                </button>
                <button
                  onClick={handleCopy}
                  className="btn-secondary flex items-center space-x-1 text-sm"
                >
                  {copied ? (
                    <>
                      <CheckIcon className="w-4 h-4 text-success-600" />
                      <span className="text-success-600">Copied!</span>
                    </>
                  ) : (
                    <>
                      <ClipboardDocumentIcon className="w-4 h-4" />
                      <span>Copy</span>
                    </>
                  )}
                </button>
                <button
                  onClick={handleDownloadTxt}
                  className="btn-secondary flex items-center space-x-1 text-sm"
                >
                  <ArrowDownTrayIcon className="w-4 h-4" />
                  <span>Download</span>
                </button>
              </div>
            </div>

            {showComparison ? (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div>
                  <h4 className="text-sm font-medium text-dark-400 mb-2">Original</h4>
                  <div className="bg-dark-950 border border-dark-700 rounded-lg p-4 text-dark-50 whitespace-pre-wrap">
                    {activeTab === 'text' ? inputText : documentFile?.name || 'Document'}
                  </div>
                </div>
                <div>
                  <h4 className="text-sm font-medium text-dark-400 mb-2">Enhanced</h4>
                  <div className="bg-dark-950 border border-dark-700 rounded-lg p-4 text-dark-50 whitespace-pre-wrap">
                    {outputText}
                  </div>
                </div>
              </div>
            ) : (
              <div className="bg-dark-950 border border-dark-700 rounded-lg p-4 text-dark-50 whitespace-pre-wrap">
                {outputText}
              </div>
            )}
          </div>
        )}

        {/* Error Display */}
        {error && (
          <div className="bg-red-500/10 border border-red-500/20 rounded-xl p-4 mb-8 flex items-start space-x-3">
            <ExclamationTriangleIcon className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
            <div>
              <h4 className="text-red-400 font-medium mb-1">Error</h4>
              <p className="text-red-300 text-sm">{error}</p>
            </div>
          </div>
        )}
      </main>

      {/* Toast Notifications */}
      <div className="fixed top-4 right-4 z-50 space-y-2">
        {toasts.map((toast) => (
          <div
            key={toast.id}
            className={`px-4 py-3 rounded-lg shadow-lg backdrop-blur-sm border animate-slideIn ${
              toast.type === 'success'
                ? 'toast-success'
                : toast.type === 'error'
                ? 'toast-error'
                : 'toast-info'
            }`}
          >
            <div className="flex items-center space-x-2">
              {toast.type === 'success' && <CheckIcon className="w-4 h-4" />}
              {toast.type === 'error' && <ExclamationTriangleIcon className="w-4 h-4" />}
              <span className="text-sm font-medium">{toast.message}</span>
            </div>
          </div>
        ))}
      </div>

      {/* Footer */}
      <footer className="border-t border-dark-700 bg-dark-950 mt-16">
        <div className="max-w-4xl mx-auto px-6 py-8">
          <div className="flex items-center justify-between">
            <div className="text-sm text-dark-400">
              © 2026 AquilaStudios. Professional AI text enhancement.
            </div>
            <div className="flex items-center space-x-6 text-sm text-dark-400">
              <span className="bg-success-600/20 text-success-400 px-2 py-1 rounded-full text-xs text-center">
                Document Processing Available
              </span>
            </div>
          </div>
        </div>
      </footer>
    </div>
  )
}