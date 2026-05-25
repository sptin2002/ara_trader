#!/bin/bash

# AraTradeQlib Orchestrator
# Exit on any error
set -e

echo "-------------------------------------------------------"
echo "🚀 STARTING ARATRADEQLIB DAILY AUTOMATION"
echo "-------------------------------------------------------"

# 1. Sync Source of Truth
echo "[Phase 1] Fetching IBKR & Sentiment data..."
python3 fetch_data.py

# 2. Update Acceleration Layer
echo "[Phase 2] Syncing SQLite to Qlib Binary..."
python3 data_to_qlib.py

# 3. Generate Signals
echo "[Phase 3] Running TFT Inference via Expression Engine..."
python3 predict_refined_signals.py

# 4. Unified Ranking
echo "[Phase 4] Applying Cross-Sectional Ranking..."
python3 rank_and_filter.py

# 5. Risk Engine
echo "[Phase 5] Calculating Position Sizing & Order Sheet..."
python3 final_plan.py

echo "-------------------------------------------------------"
echo "✅ AUTOMATION COMPLETE: Check final_trade_orders.csv"
echo "-------------------------------------------------------"