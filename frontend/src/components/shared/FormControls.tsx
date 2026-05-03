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
    <div className="theme-segmented">
      {options.map(o => (
        <button
          key={o.value}
          type="button"
          onClick={() => onChange(o.value)}
          className={cn(
            "theme-segmented-option rounded-full transition-all duration-150",
            size === "sm" ? "px-3 py-1 text-xs" : "px-4 py-1.5 text-sm",
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
  helperText?: string
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
  helperText,
}: SliderInputProps) {
  const display = formatValue ? formatValue(value) : String(value)
  return (
    <div className="space-y-2">
      <label className="flex items-baseline justify-between text-sm text-muted">
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
      {helperText && <p className="theme-field-caption">{helperText}</p>}
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
  helperText?: string
  errorText?: string
  id?: string
  disabled?: boolean
}

export function SelectInput({
  label,
  value,
  onChange,
  options,
  className,
  helperText,
  errorText,
  id,
  disabled,
}: SelectInputProps) {
  const describedBy = errorText ? `${id ?? label}-error` : helperText ? `${id ?? label}-help` : undefined
  return (
    <div className={cn("space-y-1.5", className)}>
      {label && (
        <label htmlFor={id} className="theme-field-label">{label}</label>
      )}
      <select
        id={id}
        value={value}
        onChange={e => onChange(e.target.value)}
        disabled={disabled}
        aria-invalid={Boolean(errorText)}
        aria-describedby={describedBy}
        className="theme-input w-full appearance-none pr-10"
        style={{
          backgroundImage: `url("data:image/svg+xml,%3Csvg width='10' height='6' viewBox='0 0 10 6' fill='none' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M1 1L5 5L9 1' stroke='%238a8f99' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round'/%3E%3C/svg%3E")`,
          backgroundRepeat: "no-repeat",
          backgroundPosition: "right 14px center",
        }}
      >
        {options.map(o => (
          <option key={o.value} value={o.value}>{o.label}</option>
        ))}
      </select>
      {errorText ? (
        <p id={`${id ?? label}-error`} className="theme-field-caption theme-field-error">{errorText}</p>
      ) : helperText ? (
        <p id={`${id ?? label}-help`} className="theme-field-caption">{helperText}</p>
      ) : null}
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
  helperText?: string
  errorText?: string
  id?: string
  disabled?: boolean
  uppercase?: boolean
}

export function TextInput({
  label,
  value,
  onChange,
  placeholder,
  className,
  type = "text",
  helperText,
  errorText,
  id,
  disabled,
  uppercase,
}: TextInputProps) {
  const describedBy = errorText ? `${id ?? label}-error` : helperText ? `${id ?? label}-help` : undefined
  return (
    <div className={cn("space-y-1.5", className)}>
      {label && (
        <label htmlFor={id} className="theme-field-label">{label}</label>
      )}
      <input
        id={id}
        type={type}
        value={value}
        onChange={e => onChange(uppercase ? e.target.value.toUpperCase() : e.target.value)}
        placeholder={placeholder}
        disabled={disabled}
        autoCapitalize={uppercase ? "characters" : undefined}
        aria-invalid={Boolean(errorText)}
        aria-describedby={describedBy}
        className="theme-input w-full"
      />
      {errorText ? (
        <p id={`${id ?? label}-error`} className="theme-field-caption theme-field-error">{errorText}</p>
      ) : helperText ? (
        <p id={`${id ?? label}-help`} className="theme-field-caption">{helperText}</p>
      ) : null}
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
    <label className="flex min-h-11 items-center gap-3 cursor-pointer select-none">
      <button
        type="button"
        role="switch"
        aria-checked={checked}
        onClick={() => onChange(!checked)}
        className={cn(
          "relative inline-flex h-7 w-12 shrink-0 rounded-full border transition-colors duration-200",
          checked
            ? "border-transparent bg-[hsl(var(--accent))]"
            : "border-app bg-card-muted",
        )}
      >
        <span
          className={cn(
            "pointer-events-none mt-[3px] inline-block h-5 w-5 rounded-full bg-elevated shadow-sm transition-transform duration-200",
            checked ? "translate-x-6" : "translate-x-[3px]",
          )}
        />
      </button>
      <div>
        <span className="text-sm font-medium text-app">{label}</span>
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
        "theme-button-base theme-button-primary w-full px-4",
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
