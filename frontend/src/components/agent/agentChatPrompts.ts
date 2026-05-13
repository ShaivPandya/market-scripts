import type { QuickPromptGroup } from "./AgentChatTypes"

export const QUICK_PROMPT_GROUPS: QuickPromptGroup[] = [
  {
    title: "Market Regime",
    prompts: [
      "What's the current market risk environment?",
      "How is global liquidity affecting risk assets?",
    ],
  },
  {
    title: "Portfolio Review",
    prompts: [
      "Summarize my portfolio's performance",
      "Given current positioning data, macro liquidity, and my portfolio's sector tilts, what are my top 3 risks?",
    ],
  },
  {
    title: "Positioning",
    prompts: [
      "What does positioning data say about crowded trades?",
    ],
  },
]

export const EMPTY_STATE_PROMPTS = QUICK_PROMPT_GROUPS
  .flatMap(group => group.prompts)
  .slice(0, 4)
