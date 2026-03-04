import { useState } from "react"
import { LoadingSpinner, ErrorMessage } from "@/components/shared/LoadingSpinner"
import { fetchWeeklyReport } from "@/lib/api"

interface WeeklyReportResponse {
    report: string
}

export function WeeklyReport() {
    const [isLoading, setIsLoading] = useState(false)
    const [error, setError] = useState<string | null>(null)
    const [data, setData] = useState<WeeklyReportResponse | null>(null)

    const handleGenerate = async () => {
        setIsLoading(true)
        setError(null)
        try {
            const json = await fetchWeeklyReport()
            setData(json)
        } catch (err: unknown) {
            setError(err instanceof Error ? err.message : String(err))
        } finally {
            setIsLoading(false)
        }
    }


    return (
        <div className="max-w-4xl mx-auto pb-12">
            <div className="flex items-center justify-between mb-8">
                <div>
                    <h1 className="text-3xl font-bold text-gray-900 tracking-tight">Weekly Report</h1>
                    <p className="mt-2 text-sm text-gray-500">
                        A summary of all notable market moves from the past week.
                    </p>
                </div>
                <button
                    onClick={handleGenerate}
                    disabled={isLoading}
                    className={`inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed`}
                >
                    {isLoading ? "Generating..." : "Generate Report"}
                </button>
            </div>

            {isLoading && (
                <div className="bg-white p-8 rounded-xl border shadow-sm flex flex-col items-center justify-center min-h-[400px]">
                    <LoadingSpinner />
                    <p className="mt-4 text-sm text-gray-500 animate-pulse">
                        Compiling index, commodity, FX, and technical data. This may take a minute...
                    </p>
                </div>
            )}

            {!isLoading && error && (
                <div className="bg-white p-6 rounded-xl border shadow-sm">
                    <ErrorMessage message={String(error)} />
                </div>
            )}

            {data && !isLoading && (
                <div className="bg-white rounded-xl border shadow-sm overflow-hidden">
                    <div className="px-6 py-8 md:px-8">
                        <div className="prose prose-blue prose-sm sm:prose-base max-w-none prose-headings:font-semibold prose-a:text-blue-600 prose-table:border-collapse prose-th:px-4 prose-th:py-2 prose-th:bg-gray-50 prose-th:border prose-th:border-gray-200 prose-td:px-4 prose-td:py-2 prose-td:border prose-td:border-gray-200 whitespace-pre-wrap font-mono text-xs">
                            {data.report}
                        </div>
                    </div>
                </div>
            )}

            {!data && !isLoading && !error && (
                <div className="bg-white p-12 rounded-xl border shadow-sm text-center">
                    <svg className="mx-auto h-12 w-12 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
                        <path vectorEffect="non-scaling-stroke" strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                    </svg>
                    <h3 className="mt-2 text-sm font-medium text-gray-900">No report generated</h3>
                    <p className="mt-1 text-sm text-gray-500">
                        Click the button above to fetch data and generate a new summary.
                    </p>
                </div>
            )}
        </div>
    )
}
