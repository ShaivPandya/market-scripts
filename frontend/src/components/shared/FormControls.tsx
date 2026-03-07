import { type ReactNode } from "react"
import { cn } from "@/lib/utils"

/* ─── Segmented Control ─────────────────────────────────────────────────────── */

interface SegmentedControlProps<T extends string> {
  options: { value: T; label: string }[]
  value: T
  onChange: (value: T) => void
  size?: "sm" | "md"
}

export function SegmentedControl<T extends string>({
  options,
  value,
  onChange,
  size = "md",
}: SegmentedControlProps<T>) {
  return (
    <div className="theme-segmented inline-flex items-center rounded-full p-0.5">
      {options.map(o => (
        <button
          key={o.value}
          type="button"
          onClick={() => onChange(o.value)}
          className={cn(
            "theme-segmented-option rounded-full transition-all duration-150",
            size === "sm" ? "px-3 py-1 text-xs" : "px-3.5 py-1.5 text-sm",
          )}
          data-active={value === o.value}
        >
          {o.label}
        </button>
      ))}
    </div>
  )
}

/* ─── Slider ────────────────────────────────────────────────────────────────── */

interface SliderInputProps {
  label: string
  value: number
  onChange: (value: number) => void
  min: number
  max: number
  step: number
  formatValue?: (v: number) => string
  minLabel?: string
  maxLabel?: string
}

export function SliderInput({
  label,
  value,
  onChange,
  min,
  max,
  step,
  formatValue,
  minLabel,
  maxLabel,
}: SliderInputProps) {
  const display = formatValue ? formatValue(value) : String(value)
  return (
    <div>
      <label className="mb-2 flex items-baseline justify-between text-sm text-muted">
        <span>{label}</span>
        <span className="text-sm font-semibold text-app">{display}</span>
      </label>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={e => onChange(Number(e.target.value))}
        className="hig-slider w-full"
      />
      {(minLabel || maxLabel) && (
        <div className="mt-1 flex justify-between text-[11px] text-subtle">
          <span>{minLabel}</span>
          <span>{maxLabel}</span>
        </div>
      )}
    </div>
  )
}

/* ─── Select ────────────────────────────────────────────────────────────────── */

interface SelectInputProps {
  label?: string
  value: string
  onChange: (value: string) => void
  options: { value: string; label: string }[]
  className?: string
}

export function SelectInput({ label, value, onChange, options, className }: SelectInputProps) {
  return (
    <div className={className}>
      {label && (
        <label className="mb-1.5 block text-sm text-muted">{label}</label>
      )}
      <select
        value={value}
        onChange={e => onChange(e.target.value)}
        className="theme-input w-full appearance-none pr-8"
        style={{ backgroundImage: `url("data:image/svg+xml,%3Csvg width='10' height='6' viewBox='0 0 10 6' fill='none' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M1 1L5 5L9 1' stroke='%239CA3AF' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round'/%3E%3C/svg%3E")`, backgroundRepeat: "no-repeat", backgroundPosition: "right 12px center" }}
      >
        {options.map(o => (
          <option key={o.value} value={o.value}>{o.label}</option>
        ))}
      </select>
    </div>
  )
}

/* ─── Text Input ────────────────────────────────────────────────────────────── */

interface TextInputProps {
  label?: string
  value: string
  onChange: (value: string) => void
  placeholder?: string
  className?: string
  type?: string
}

export function TextInput({ label, value, onChange, placeholder, className, type = "text" }: TextInputProps) {
  return (
    <div className={className}>
      {label && (
        <label className="mb-1.5 block text-sm text-muted">{label}</label>
      )}
      <input
        type={type}
        value={value}
        onChange={e => onChange(e.target.value)}
        placeholder={placeholder}
        className="theme-input w-full"
      />
    </div>
  )
}

/* ─── Toggle Switch ─────────────────────────────────────────────────────────── */

interface ToggleProps {
  label: string
  checked: boolean
  onChange: (checked: boolean) => void
  description?: string
}

export function Toggle({ label, checked, onChange, description }: ToggleProps) {
  return (
    <label className="flex items-center gap-3 cursor-pointer select-none">
      <button
        type="button"
        role="switch"
        aria-checked={checked}
        onClick={() => onChange(!checked)}
        className="relative inline-flex h-[22px] w-[40px] shrink-0 rounded-full transition-colors duration-200"
        style={{
          backgroundColor: checked
            ? "hsl(var(--accent))"
            : "hsl(var(--muted-3))",
        }}
      >
        <span
          className={cn(
            "pointer-events-none inline-block h-[18px] w-[18px] rounded-full shadow-sm transition-transform duration-200",
            checked ? "translate-x-[20px]" : "translate-x-[2px]",
            "mt-[2px]",
          )}
          style={{ backgroundColor: "hsl(var(--card))" }}
        />
      </button>
      <div>
        <span className="text-sm text-app">{label}</span>
        {description && <span className="mt-0.5 block text-xs text-subtle">{description}</span>}
      </div>
    </label>
  )
}

/* ─── Action Button ─────────────────────────────────────────────────────────── */

interface ActionButtonProps {
  onClick?: () => void
  type?: "button" | "submit"
  disabled?: boolean
  loading?: boolean
  loadingText?: string
  children: ReactNode
  className?: string
}

export function ActionButton({
  onClick,
  type = "button",
  disabled,
  loading,
  loadingText,
  children,
  className,
}: ActionButtonProps) {
  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled || loading}
      className={cn(
        "theme-button-primary w-full rounded-lg py-2.5 text-sm font-semibold transition-all duration-150",
        "active:scale-[0.98]",
        "disabled:opacity-50 disabled:pointer-events-none",
        className,
      )}
    >
      {loading ? loadingText ?? children : children}
    </button>
  )
}

/* ─── Control Panel ─────────────────────────────────────────────────────────── */

interface ControlPanelProps {
  children: ReactNode
  maxWidth?: string
}

export function ControlPanel({ children, maxWidth = "max-w-md" }: ControlPanelProps) {
  return (
    <div className={cn("theme-surface mb-6 space-y-5 rounded-xl p-5", maxWidth)}>
      {children}
    </div>
  )
}
