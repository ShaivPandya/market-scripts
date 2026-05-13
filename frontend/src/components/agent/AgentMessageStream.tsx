import { AlertCircle, MessageCircle } from "lucide-react"
import { AgentMessage } from "./AgentMessage"
import { EMPTY_STATE_PROMPTS } from "./agentChatPrompts"
import type { AgentMessage as AgentMessageType } from "@/hooks/useAgentChat"
import type { RefObject } from "react"

interface AgentMessageStreamProps {
  messages: AgentMessageType[]
  isStreaming: boolean
  error: string | null
  onPrompt: (prompt: string) => void
  messagesEndRef: RefObject<HTMLDivElement | null>
}

export function AgentMessageStream({
  messages,
  isStreaming,
  error,
  onPrompt,
  messagesEndRef,
}: AgentMessageStreamProps) {
  const isEmpty = messages.length === 0 && !isStreaming

  return (
    <div className="flex-1 overflow-y-auto bg-app px-4 py-4">
      {isEmpty ? (
        <div className="flex min-h-full flex-col items-center justify-center text-center">
          <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-2xl border border-app bg-card text-link shadow-sm">
            <MessageCircle size={22} aria-hidden="true" />
          </div>
          <p className="text-base font-semibold text-app">Stan is ready</p>
          <p className="mt-1 max-w-[28rem] text-sm leading-6 text-muted">
            Ask for a portfolio read, market regime check, or risk review.
          </p>
          <div className="mt-5 grid w-full max-w-[34rem] grid-cols-1 gap-2 sm:grid-cols-2">
            {EMPTY_STATE_PROMPTS.map(prompt => (
              <button
                key={prompt}
                type="button"
                onClick={() => onPrompt(prompt)}
                className="theme-button-secondary min-h-12 justify-start rounded-xl px-3 py-2 text-left text-xs font-medium leading-5 text-muted hover:text-app"
              >
                {prompt}
              </button>
            ))}
          </div>
        </div>
      ) : (
        <div className="mx-auto flex w-full max-w-[54rem] flex-col">
          {messages.map(message => (
            <AgentMessage key={message.id} message={message} />
          ))}
          {error && (
            <div
              className="mb-3 flex items-start gap-2 rounded-xl border border-[hsl(var(--destructive)/0.22)] bg-[hsl(var(--destructive-muted))] px-3 py-2 text-sm text-negative"
              role="alert"
            >
              <AlertCircle size={15} className="mt-0.5 shrink-0" aria-hidden="true" />
              <div>
                <p className="font-semibold">Request interrupted</p>
                <p className="mt-0.5 text-xs leading-5">{error}</p>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
      )}
    </div>
  )
}
