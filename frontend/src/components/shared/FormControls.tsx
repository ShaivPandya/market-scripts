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
    <div className="inline-flex items-center rounded-full bg-gray-100 p-0.5">
      {options.map(o => (
        <button
          key={o.value}
          type="button"
          onClick={() => onChange(o.value)}
          className={cn(
            "rounded-full transition-all duration-150",
            size === "sm" ? "px-3 py-1 text-xs" : "px-3.5 py-1.5 text-sm",
            value === o.value
              ? "bg-white text-gray-900 font-medium shadow-sm"
              : "text-gray-500 hover:text-gray-700",
          )}
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
      <label className="flex items-baseline justify-between text-sm text-gray-600 mb-2">
        <span>{label}</span>
        <span className="text-sm font-semibold text-gray-900">{display}</span>
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
        <div className="flex justify-between text-[11px] text-gray-400 mt-1">
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
        <label className="block text-sm text-gray-600 mb-1.5">{label}</label>
      )}
      <select
        value={value}
        onChange={e => onChange(e.target.value)}
        className="w-full appearance-none rounded-lg border border-gray-200 bg-white px-3 py-2 text-sm text-gray-900 shadow-sm outline-none transition-colors focus:border-blue-400 focus:ring-2 focus:ring-blue-100"
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
        <label className="block text-sm text-gray-600 mb-1.5">{label}</label>
      )}
      <input
        type={type}
        value={value}
        onChange={e => onChange(e.target.value)}
        placeholder={placeholder}
        className="w-full rounded-lg border border-gray-200 bg-white px-3 py-2 text-sm text-gray-900 shadow-sm outline-none transition-colors placeholder:text-gray-400 focus:border-blue-400 focus:ring-2 focus:ring-blue-100"
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
        className={cn(
          "relative inline-flex h-[22px] w-[40px] shrink-0 rounded-full transition-colors duration-200",
          checked ? "bg-blue-500" : "bg-gray-300",
        )}
      >
        <span
          className={cn(
            "pointer-events-none inline-block h-[18px] w-[18px] rounded-full bg-white shadow-sm transition-transform duration-200",
            checked ? "translate-x-[20px]" : "translate-x-[2px]",
            "mt-[2px]",
          )}
        />
      </button>
      <div>
        <span className="text-sm text-gray-700">{label}</span>
        {description && <span className="block text-xs text-gray-400 mt-0.5">{description}</span>}
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
  children: React.ReactNode
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
        "w-full py-2.5 rounded-lg text-sm font-semibold transition-all duration-150",
        "bg-blue-500 text-white shadow-sm",
        "hover:bg-blue-600 active:scale-[0.98]",
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
  children: React.ReactNode
  maxWidth?: string
}

export function ControlPanel({ children, maxWidth = "max-w-md" }: ControlPanelProps) {
  return (
    <div className={cn("rounded-xl border border-gray-200/80 bg-white p-5 mb-6 space-y-5", maxWidth)}>
      {children}
    </div>
  )
}
