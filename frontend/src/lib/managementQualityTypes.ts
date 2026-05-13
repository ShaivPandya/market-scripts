export interface ManagementQualitySummaryQuestion {
  rating: string | null
  text: string | null
}

export interface ManagementQualitySummary {
  overall_rating?: string
  bottom_line?: string
  owner_mindset?: ManagementQualitySummaryQuestion
  business_value_understanding?: ManagementQualitySummaryQuestion
  follow_through?: ManagementQualitySummaryQuestion
}

export interface ManagementQualityScorecardRow {
  id?: string
  source_id?: number | null
  question: string
  rating: string
  evidence: string
}

export interface ManagementQualityBullet {
  id?: string
  source_id?: number | null
  title: string | null
  text: string
  response_rating?: string
  response_text?: string | null
}

export interface ManagementQualityAssessment {
  id: string
  source_id?: number | null
  issuer_id: string
  ticker?: string | null
  status: "active" | "superseded" | "archived" | string
  overall_rating?: string | null
  bottom_line?: string | null
  owner_mindset_rating?: string | null
  owner_mindset_text?: string | null
  business_value_understanding_rating?: string | null
  business_value_understanding_text?: string | null
  follow_through_rating?: string | null
  follow_through_text?: string | null
  content_hash?: string | null
  document_id?: string | null
  scorecard?: ManagementQualityScorecardRow[]
  accomplishments?: ManagementQualityBullet[]
  setbacks?: ManagementQualityBullet[]
  parsed?: ParsedManagementQuality | null
}

export interface ParsedManagementQuality {
  summary: ManagementQualitySummary | null
  scorecard: ManagementQualityScorecardRow[] | null
  accomplishments: ManagementQualityBullet[] | null
  setbacks: ManagementQualityBullet[] | null
}
