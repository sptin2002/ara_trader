Write-Host "-------------------------------------------------------" -ForegroundColor Cyan
Write-Host "🚀 STARTING ARATRADEQLIB DAILY AUTOMATION" -ForegroundColor Cyan
Write-Host "-------------------------------------------------------" -ForegroundColor Cyan

# 1. Sync Source of Truth
Write-Host "[Phase 1] Fetching IBKR & Sentiment data..."
python fetch_data.py
if ($LASTEXITCODE -ne 0) { Write-Error "Phase 1 Failed"; exit }

# 2. Update Acceleration Layer
Write-Host "[Phase 2] Syncing SQLite to Qlib Binary..."
python data_to_qlib.py
if ($LASTEXITCODE -ne 0) { Write-Error "Phase 2 Failed"; exit }

# 3. Generate Signals
Write-Host "[Phase 3] Running TFT Inference via Expression Engine..."
python predict_refined_signals.py
if ($LASTEXITCODE -ne 0) { Write-Error "Phase 3 Failed"; exit }

# 4. Unified Ranking
Write-Host "[Phase 4] Applying Cross-Sectional Ranking..."
python rank_and_filter.py
if ($LASTEXITCODE -ne 0) { Write-Error "Phase 4 Failed"; exit }

# 5. Risk Engine
Write-Host "[Phase 5] Calculating Position Sizing & Order Sheet..."
python final_plan.py
if ($LASTEXITCODE -ne 0) { Write-Error "Phase 5 Failed"; exit }

Write-Host "-------------------------------------------------------" -ForegroundColor Green
Write-Host "✅ AUTOMATION COMPLETE: Check final_trade_orders.csv" -ForegroundColor Green
Write-Host "-------------------------------------------------------" -ForegroundColor Green