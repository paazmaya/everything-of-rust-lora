# Master pipeline runner (PowerShell)
$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host " Starting Rust LoRA Data Pipeline" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

if (-not $env:GITHUB_TOKEN) {
    Write-Host "`nERROR: GITHUB_TOKEN environment variable is required!" -ForegroundColor Red
    Write-Host "Set it with: " -ForegroundColor Red -NoNewline
    Write-Host "`$env:GITHUB_TOKEN='your_github_token'" -ForegroundColor Yellow
    exit 1
}

Write-Host "`nGITHUB_TOKEN found. Proceeding with pipeline.`n" -ForegroundColor Green

Write-Host "`n[1/5] Collecting official Rust documentation..." -ForegroundColor Green
uv run python scripts/01_collect_rust_book.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "`n[2/5] Collecting docs.rs crate documentation..." -ForegroundColor Green
uv run python scripts/02_collect_docs_rs.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "`n[3/5] Collecting GitHub READMEs and examples..." -ForegroundColor Green
uv run python scripts/03_collect_github.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "`n[4/5] Collecting ESP-RS and Embedded docs..." -ForegroundColor Green
uv run python scripts/04_collect_esp_rs.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "`n[5/5] Collecting Blogs and Best Practices..." -ForegroundColor Green
uv run python scripts/05_collect_blogs.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "`n[6/6] Transforming and chunking data..." -ForegroundColor Green
uv run python scripts/06_transform_data.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "`n[7/7] Indexing into Vector Database..." -ForegroundColor Green
uv run python scripts/07_vector_store.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "`n[8/8] Creating Training Dataset..." -ForegroundColor Green
uv run python scripts/08_create_dataset.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host " Data Pipeline Complete!" -ForegroundColor Cyan
Write-Host " Next: Upload 'data/datasets/train.jsonl' to Unsloth Studio and run train_granite.py" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan
