# Industry Monitor — Transcript Upload Runbook

The Industry Monitor reads earnings-call PDFs from one of two sources, depending on environment:

| Environment        | Source                                                                   |
|--------------------|--------------------------------------------------------------------------|
| Local development  | `macro/industry/files/{sector_dir}/{TICKER}.pdf` (this repo's working tree) |
| Production / Cloud Run | `gs://${GCS_STATE_BUCKET}/${INDUSTRY_TRANSCRIPTS_PREFIX}/{sector_dir}/{TICKER}.pdf` |

`INDUSTRY_TRANSCRIPTS_PREFIX` defaults to `industry-transcripts/` if unset. Sector directories are the lowercased sector name with spaces replaced by underscores (e.g. `Capital Goods` → `capital_goods`). Filenames use the ticker, with one exception: `ODFL` is stored as `ODL.pdf` (see `_TICKER_FILENAME_MAP` in `industry_monitor.py`).

## Adding a new quarterly transcript in production

1. Save the transcript locally as a PDF named after the ticker (e.g. `DHI.pdf`, or `ODL.pdf` for ODFL).
2. Upload to the production bucket:

   ```sh
   gsutil cp /path/to/DHI.pdf "gs://${GCS_STATE_BUCKET}/industry-transcripts/housing/DHI.pdf"
   ```

3. Trigger a refresh so the worker re-extracts and re-summarizes:

   ```sh
   curl -X GET "https://<api-host>/industry-monitor?refresh=true"
   ```

   The refresh dispatches via RQ to the Cloud Run worker. Tail worker logs for `Industry data fetch and summarization complete`.

4. Verify the UI now shows `M / N companies` with the updated ticker reflected.

## One-time backfill (initial cutover)

To populate the bucket from a workstation that has the local PDFs:

```sh
# Dry run first to see what would happen.
STATE_STORAGE_BACKEND=gcs GCS_STATE_BUCKET=talisman-state-prod \
    python -m api.industry_pdf_backfill upload --dry-run

# Live upload; idempotent — re-runs skip files whose md5 already matches.
STATE_STORAGE_BACKEND=gcs GCS_STATE_BUCKET=talisman-state-prod \
    python -m api.industry_pdf_backfill upload
```

The script applies the `ODFL → ODL.pdf` rename automatically.

## Repository policy

Transcript PDFs and the local `industry_transcripts.sqlite3` cache are intentionally not kept in git. Production reads PDFs from Cloud Storage and uses the configured state database. Local development can still use PDFs placed under `macro/industry/files/{sector_dir}/{TICKER}.pdf`; those files are ignored so they do not enter commits or Docker/GCloud build contexts.

## IAM gotcha

Both the API service (`talisman-api`) and the RQ worker run as Cloud Run services with their own service accounts. The **worker SA** is what actually executes the refresh job, so it must have `roles/storage.objectViewer` on `${GCS_STATE_BUCKET}`. If transcripts upload successfully but the UI still shows `0 / N`, check the worker SA's bucket permissions first.

## Bucket hygiene

Enable a lifecycle rule on the `industry-transcripts/` prefix to retain noncurrent versions for ~365 days. That lets you recover from accidental overwrites of a quarterly PDF (which would otherwise be silently masked by its replacement on next refresh).
