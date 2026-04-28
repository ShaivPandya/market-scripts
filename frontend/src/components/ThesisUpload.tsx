import { useEffect, useRef, useState, type ChangeEvent } from "react"
import { useQueryClient } from "@tanstack/react-query"
import { FileUp, Loader2 } from "lucide-react"

import { uploadThesisDocument, type ThesisStatus } from "@/lib/api"

type ThesisUploadProps = {
  ticker: string
  status?: ThesisStatus
}

export function ThesisUpload({ ticker, status = "missing" }: ThesisUploadProps) {
  const queryClient = useQueryClient()
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [isUploading, setIsUploading] = useState(false)
  const [notice, setNotice] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!notice && !error) return
    const timer = window.setTimeout(() => {
      setNotice(null)
      setError(null)
    }, 4000)
    return () => window.clearTimeout(timer)
  }, [notice, error])

  const handlePick = () => {
    if (isUploading) return
    fileInputRef.current?.click()
  }

  const handleFileChange = async (event: ChangeEvent<HTMLInputElement>) => {
    const selectedFile = event.target.files?.[0]
    event.target.value = ""
    if (!selectedFile) return

    const fileName = (selectedFile.name || "").toLowerCase()
    const isPdf = selectedFile.type === "application/pdf" || fileName.endsWith(".pdf")
    const isMarkdown =
      selectedFile.type === "text/markdown" ||
      selectedFile.type === "text/x-markdown" ||
      fileName.endsWith(".md")
    if (!isPdf && !isMarkdown) {
      setNotice(null)
      setError("PDF or Markdown only")
      return
    }

    setIsUploading(true)
    setNotice(null)
    setError(null)
    try {
      await uploadThesisDocument(ticker, selectedFile)
      setNotice("Saved")
      await queryClient.invalidateQueries({ queryKey: ["thesis", "status"] })
    } catch (e) {
      const message = e instanceof Error ? e.message : "Upload failed"
      setError(message)
    } finally {
      setIsUploading(false)
    }
  }

  const isPopulated = status === "populated"
  const title = isUploading
    ? `Saving ${ticker} thesis...`
    : isPopulated
      ? `Replace ${ticker} thesis from PDF or Markdown`
      : `Upload PDF or Markdown thesis for ${ticker}`

  return (
    <div className="flex items-center gap-1">
      <button
        type="button"
        onClick={handlePick}
        disabled={isUploading}
        className="inline-flex h-6 w-6 items-center justify-center rounded-md border border-gray-200 bg-white transition-colors hover:bg-gray-50 disabled:cursor-not-allowed"
        title={title}
        aria-label={title}
      >
        {isUploading ? (
          <Loader2 className="h-3.5 w-3.5 animate-spin text-gray-500" />
        ) : (
          <FileUp className={`h-3.5 w-3.5 ${isPopulated ? "text-green-600" : "text-gray-400"}`} />
        )}
      </button>
      {notice && <span className="text-[11px] font-medium text-green-600">{notice}</span>}
      {error && <span className="text-[11px] font-medium text-red-600">{error}</span>}
      <input
        ref={fileInputRef}
        type="file"
        accept=".pdf,.md,application/pdf,text/markdown,text/x-markdown"
        className="hidden"
        onChange={handleFileChange}
      />
    </div>
  )
}
