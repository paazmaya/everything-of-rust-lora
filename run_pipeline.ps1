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

Write-Host "`n[1/10] Collecting official Rust documentation..." -ForegroundColor Green
uv run python scripts/01_collect_rust_book.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "`n[2/10] Collecting rustc releases, changelogs, and version diffs..." -ForegroundColor Green
uv run python scripts/01b_collect_rust_releases.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "`n[3/10] Collecting docs.rs crate documentation..." -ForegroundColor Green
uv run python scripts/02_collect_docs_rs.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "`n[4/10] Collecting Cybersecurity, RustSec, Kali Linux, and Living-Off-The-Land (LOTL)..." -ForegroundColor Green
uv run python scripts/02b_collect_cybersec.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "`n[5/10] Collecting GitHub READMEs and examples..." -ForegroundColor Green
uv run python scripts/03_collect_github.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "`n[6/10] Collecting ESP-RS and Embedded docs..." -ForegroundColor Green
uv run python scripts/04_collect_esp_rs.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "`n[7/10] Collecting Blogs and Best Practices..." -ForegroundColor Green
uv run python scripts/05_collect_blogs.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "`n[8/10] Transforming and chunking data..." -ForegroundColor Green
uv run python scripts/06_transform_data.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "`n[9/10] Indexing into Vector Database..." -ForegroundColor Green
uv run python scripts/07_vector_store.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "`n[10/10] Creating Training Dataset..." -ForegroundColor Green
uv run python scripts/08_create_dataset.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host " Data Pipeline Complete!" -ForegroundColor Cyan
Write-Host " Next: Train LoRA with:" -ForegroundColor Cyan
Write-Host "   - train_k2_horizon.py (IFM K2-Horizon 3.7B - recommended for coding agents)" -ForegroundColor Cyan
Write-Host "   - train_granite.py (IBM Granite 4.2 8B)" -ForegroundColor Cyan
Write-Host "   - train_qwen.py (Qwen 3.5 4B)" -ForegroundColor Cyan
Write-Host "   - train_qwen3_8.py (Qwen 3.8 27B)" -ForegroundColor Cyan
Write-Host "   - train_nemotron3_nano.py (Nemotron 3 Nano 4B)" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan
