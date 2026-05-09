export interface ParsedFinancialMetric {
  value: string | null
  context: string
}

export interface DebtTranche {
  tranche: string
  rate: string
  maturity: string
}

export interface ParsedDebt {
  summary: string
  tranches: DebtTranche[]
}

export interface ParsedFinancials {
  revenue_growth: ParsedFinancialMetric | null
  eps_growth: ParsedFinancialMetric | null
  interest_coverage: ParsedFinancialMetric | null
  operating_margin: ParsedFinancialMetric | null
  net_income_margin: ParsedFinancialMetric | null
  debt: ParsedDebt | null
  reinvestment: string | null
}

export interface SensitivityRow {
  factor: string
  sensitivity: string
  capacity: string
}

export interface PorterForce {
  force: string
  rating: string
  description: string
}

export interface OutlookPoint {
  label?: string
  text: string
}

export interface ParsedOutlookSection {
  rating: string | null
  points: (string | OutlookPoint)[]
}

export interface SupplyChainCounterparty {
  name: string
  relationship: string | null
  exposure: string | null
  notes: string | null
}

export interface ParsedSupplyChain {
  suppliers: SupplyChainCounterparty[]
  customers: SupplyChainCounterparty[]
}

export interface ParsedOverview {
  financials: ParsedFinancials | null
  sensitivity: SensitivityRow[] | null
  porters_five_forces: PorterForce[] | null
  supply_outlook: ParsedOutlookSection | null
  demand_outlook: ParsedOutlookSection | null
  supply_chain: ParsedSupplyChain | null
}
