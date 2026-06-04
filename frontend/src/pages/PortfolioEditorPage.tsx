import { Link } from "react-router-dom"
import { useRegisterScreenContext } from "@/contexts/ScreenContext"
import { PortfolioEditorPanel } from "@/components/PortfolioEditor"
import { PageHeader } from "@/components/shared/PageHeader"
import { Notice } from "@/components/shared/Notice"

export function PortfolioEditorPage() {
  useRegisterScreenContext({
    pageName: "Edit Portfolio",
    metrics: {},
    summary: "Edit portfolio positions, hedges, and book size. Changes are staged for approval.",
    correspondingTools: ["get_portfolio_positions", "get_hedge_positions", "propose_portfolio_positions_update"],
  })

  return (
    <div>
      <PageHeader
        title="Edit Portfolio"
        subtitle="Stage portfolio and hedge changes for approval. IBKR Flex imports classify short SPY, IWM, and QQQ stock rows as hedges."
        actions={(
          <Link
            to="/"
            className="theme-button-base theme-button-secondary px-4"
          >
            Back to Dashboard
          </Link>
        )}
      />

      <Notice tone="info" className="mb-6">
        Nothing is applied until proposals are reviewed and approved in Workspace.
      </Notice>

      <PortfolioEditorPanel />
    </div>
  )
}
