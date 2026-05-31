export const STAN_OPEN_EVENT = "talisman:stan-open"

export interface StanOpenDetail {
  command: string
  durable?: boolean
}

export function openStanWithCommand(command: string, options?: { durable?: boolean }) {
  window.dispatchEvent(
    new CustomEvent<StanOpenDetail>(STAN_OPEN_EVENT, {
      detail: { command, durable: options?.durable ?? true },
    }),
  )
}
