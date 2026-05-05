import { useId, useLayoutEffect, useRef, useState, type CSSProperties, type ReactNode } from "react"
import { createPortal } from "react-dom"

import { cn } from "@/lib/utils"

interface BoundedTooltipProps {
  children: ReactNode
  className?: string
  tooltip?: string
  ariaLabel?: string
}

interface TooltipPosition {
  left: number
  top: number
  arrowX: number
  side: "top" | "bottom"
}

type TooltipStyle = CSSProperties & {
  "--theme-tooltip-arrow-x"?: string
}

export function BoundedTooltip({ children, className, tooltip, ariaLabel }: BoundedTooltipProps) {
  const triggerRef = useRef<HTMLSpanElement>(null)
  const tooltipRef = useRef<HTMLDivElement>(null)
  const tooltipId = useId()
  const [open, setOpen] = useState(false)
  const [position, setPosition] = useState<TooltipPosition | null>(null)

  useLayoutEffect(() => {
    if (!open || !tooltip) return

    const updatePosition = () => {
      const triggerRect = triggerRef.current?.getBoundingClientRect()
      const tooltipRect = tooltipRef.current?.getBoundingClientRect()
      if (!triggerRect || !tooltipRect) return

      const margin = 8
      const gap = 8
      const anchorX = triggerRect.left + triggerRect.width / 2
      const maxLeft = window.innerWidth - tooltipRect.width - margin
      const left = Math.max(margin, Math.min(anchorX - tooltipRect.width / 2, maxLeft))
      const topBelow = triggerRect.bottom + gap
      const topAbove = triggerRect.top - tooltipRect.height - gap
      const hasRoomBelow = topBelow + tooltipRect.height + margin <= window.innerHeight
      const side = hasRoomBelow || topAbove < margin ? "bottom" : "top"
      const top = side === "bottom" ? topBelow : topAbove
      const arrowX = Math.max(12, Math.min(anchorX - left, tooltipRect.width - 12))

      setPosition({ left, top: Math.max(margin, top), arrowX, side })
    }

    updatePosition()
    window.addEventListener("resize", updatePosition)
    window.addEventListener("scroll", updatePosition, true)

    return () => {
      window.removeEventListener("resize", updatePosition)
      window.removeEventListener("scroll", updatePosition, true)
    }
  }, [open, tooltip])

  const tooltipStyle: TooltipStyle = {
    left: position?.left ?? 0,
    top: position?.top ?? 0,
    visibility: position ? "visible" : "hidden",
    "--theme-tooltip-arrow-x": position ? `${position.arrowX}px` : "50%",
  }

  return (
    <>
      <span
        ref={triggerRef}
        className={cn(className, tooltip && "theme-tooltip")}
        aria-label={ariaLabel}
        aria-describedby={open && tooltip ? tooltipId : undefined}
        onPointerEnter={() => setOpen(true)}
        onPointerLeave={() => setOpen(false)}
        onFocus={() => setOpen(true)}
        onBlur={() => setOpen(false)}
        onKeyDown={event => {
          if (event.key === "Escape") setOpen(false)
        }}
      >
        {children}
      </span>
      {open && tooltip
        ? createPortal(
            <div
              ref={tooltipRef}
              id={tooltipId}
              role="tooltip"
              className="theme-tooltip-content"
              data-side={position?.side ?? "bottom"}
              style={tooltipStyle}
            >
              {tooltip}
            </div>,
            document.body,
          )
        : null}
    </>
  )
}
