export const CHAT_TEXTAREA_MAX_HEIGHT = 120

export function resizeChatTextarea(el: HTMLTextAreaElement) {
  el.style.height = "auto"
  const nextHeight = Math.min(el.scrollHeight, CHAT_TEXTAREA_MAX_HEIGHT)
  el.style.height = `${nextHeight}px`
  el.style.overflowX = "hidden"
  el.style.overflowY = el.scrollHeight > CHAT_TEXTAREA_MAX_HEIGHT ? "auto" : "hidden"
}
