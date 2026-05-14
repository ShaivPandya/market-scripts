import { useEffect, useRef, useState, type ChangeEvent } from "react"
import { useQueryClient } from "@tanstack/react-query"
import { Loader2, UserRoundCheck } from "lucide-react"

import { formatApprovalDisplayLabel } from "@/components/shared/StagedProposalNotice"
import { useDocumentGenerationUpload } from "@/hooks/useDocumentGenerationUpload"
import { invalidateApprovalSummaries } from "@/lib/approvalQueries"
import type { StagedMutationResponse } from "@/lib/api"

type ManagementQualityUploadProps = {
  ticker: string
  hasContent: boolean
}

export function ManagementQualityUpload({ ticker, hasContent }: ManagementQualityUploadProps) {
  const queryClient = useQueryClient()
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [notice, setNotice] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const { isUploading, startUpload } = useDocumentGenerationUpload<StagedMutationResponse>({
    kind: "management_quality",
    ticker,
    onSuccess: async result => {
      setNotice(`${formatApprovalDisplayLabel(result.approval_id)} staged`)
      await queryClient.invalidateQueries({ queryKey: ["dossier", ticker] })
      await invalidateApprovalSummaries(queryClient)
    },
    onError: message => setError(message),
  })

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

    setNotice(null)
    setError(null)
    await startUpload(selectedFile)
  }

  const title = isUploading
    ? `Generating ${ticker} management quality...`
    : hasContent
      ? `Replace ${ticker} management quality from PDF or Markdown`
      : `Upload PDF or Markdown management quality for ${ticker}`
  const label = isUploading ? "Saving..." : hasContent ? "Replace Management" : "Upload Management"

  return (
    <div className="flex items-center gap-1">
      <button
        type="button"
        onClick={handlePick}
        disabled={isUploading}
        className="inline-flex items-center gap-1.5 rounded-lg border border-app px-2.5 py-1 text-xs font-medium text-muted hover:text-app transition-colors disabled:cursor-not-allowed disabled:opacity-60"
        title={title}
        aria-label={title}
      >
        {isUploading ? (
          <Loader2 className="h-3.5 w-3.5 animate-spin" />
        ) : (
          <UserRoundCheck className={`h-3.5 w-3.5 ${hasContent ? "text-green-600 dark:text-green-400" : ""}`} />
        )}
        <span>{label}</span>
      </button>
      {notice && <span className="text-[11px] font-medium text-amber-600">{notice}</span>}
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
