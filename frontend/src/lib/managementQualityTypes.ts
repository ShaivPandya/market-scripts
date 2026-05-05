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
  question: string
  rating: string
  evidence: string
}

export interface ManagementQualityBullet {
  title: string | null
  text: string
  response_rating?: string
  response_text?: string | null
}

export interface ParsedManagementQuality {
  summary: ManagementQualitySummary | null
  scorecard: ManagementQualityScorecardRow[] | null
  accomplishments: ManagementQualityBullet[] | null
  setbacks: ManagementQualityBullet[] | null
}
