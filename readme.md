# AraTradeSmart (ARA-QLIB) 🚀

[![License: GPL-3.0](https://img.shields.io/badge/License-GPL%203.0-blue.svg)](https://opensource.org/licenses/GPL-3.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework: TensorFlow/PyTorch](https://img.shields.io/badge/Framework-TensorFlow%20%2F%20PyTorch-orange.svg)](https://tensorflow.org/)
[![Data: Qlib Accelerated](https://img.shields.io/badge/Data-Microsoft%20Qlib-0078d4.svg)](https://github.com/microsoft/qlib)

**AraTradeSmart** is an open-source, production-ready, multi-modal algorithmic trading framework engineered for US Equities. The project's core philosophy is **ARA: Automated Risk-Adjusted Alpha**. It bridges the gap between state-of-the-art deep learning models and programmatic execution, ensuring that alpha generation is always structurally bound by tight capital-protection guardrails.

We are opening this framework to build a community of quantitative researchers, data engineers, and machine learning engineers to advance the boundaries of multi-modal market modeling.

---

## 🏗️ The System Architecture

AraTradeSmart is structured as a modular pipeline that converts raw market fragments into high-precision, risk-adjusted institutional orders.


```

[ Polygon News / IBKR Prices ] ──> Ingested to SQLite (Source of Truth)
│
▼
[ Qlib Acceleration Layer ] ─────> Fast Binary Conversion (.bin)
│
▼
[ TFT Transformer Brain ] ───────> Multi-Modal Inference (Tech Factors + FinBERT Sentiment)
│
▼
[ Refinement Gate ] ─────────────> Restrictive Overfit Guardrails (Acc >= 55%, Delta <= 12%)
│
▼
[ Unified Ranking Engine ] ──────> 70% Confidence / 30% Structural Floor Cross-Sectional Sort
│
▼
[ ARA.AI Risk Engine ] ──────────> Dynamic Sizing via ATR & Precise Order Generation for IBKR

```

### Core Components
1. **Data & Acceleration Layer (`trading_vault.db` & `qlib_data/bin`)**: Continuously ingests daily price arrays via Interactive Brokers (IBKR) and historical news feeds via the Polygon API. A pipeline bridge (`data_to_qlib.py`) translates raw SQLite records into optimized, flattened NumPy binary files (`.bin`), accelerating downstream computations.
2. **The Multi-Modal Predictive Brain**: Utilizes Microsoft Qlib's optimized *Expression Engine* to derive alpha factors ($RSI$, $ROC$, $ATR$, $VWAP$, etc.) simultaneously. These are fed alongside low-dimensional NLP sentiment vectors (mined via a local **FinBERT** engine) into a **Temporal Fusion Transformer (TFT)**.
3. **The Refinement Gate**: A strict automated quality control gate. If a model's validation directional accuracy drops under $55\%$, or if its generalization gap (overfitting delta) exceeds $12\%$, the training artifact is automatically rejected.
4. **Unified Ranking & Filtering**: Sorts a watchlist of highly liquid assets using a cross-sectional composite formula balancing **ML Prediction Confidence (70%)** against **Structural Support Floor Strength (30%)**.
5. **The ARA.AI Risk Engine**: Dynamically calculates position sizes by mapping static capital risk limits ($1\%$ to $2\%$) against an $ATR$ volatility multiplier, embedding front-run protective $Stop\ Loss$ and $Take\ Profit$ parameters natively into execution sheets.

---

## 📂 Repository Structure

The framework is highly modular. While components can run out of a flat root directory, contributors are encouraged to align developments with the target layout below:

```text
ara_trader/
├── config.json              # System configuration thresholds, settings, and watchlists
├── qlib_factor_config.yaml  # Qlib mathematical expression definitions
├── run_predict.sh           # Master Linux/macOS automation orchestrator
├── run_predict.ps1          # Master Windows PowerShell automation orchestrator
├── train_refined_model.py   # Scheduled multi-modal retraining utility
├── finbert-local/           # LOCAL ONLY: Downloaded FinBERT weights (GIT IGNORED)
│   ├── config.json          # FinBERT architecture configurations
│   ├── pytorch_model.bin    # FinBERT core model tensor weights (~418MB)
│   └── vocab.txt            # FinBERT tokenizer dictionary
├── config/                  # Future structural target for pipeline configurations
├── src/
│   ├── data/                # Ingestion engines (fetch_data.py, data_to_qlib.py)
│   ├── features/            # FinBERT sentiment mining & feature transformations
│   ├── models/              # Model architectures, training scripts, and inference logic
│   ├── selection/           # Refinement gates and ranking (rank_and_filter.py)
│   └── execution/           # Risk calculation and execution planning (final_plan.py)
└── tests/                   # Unit tests for features and order-generation logic

```

---

## ⚡ Quick Start

### 1. Prerequisites

* Python 3.10 or 3.11
* PyTorch & TensorFlow (CUDA supported for accelerated environments)
* **Interactive Brokers Connection:** You must have an active instance of **Trader Workstation (TWS)** or **IB Gateway** running locally with the API enabled via port `7497` (or configured via `config.json`).

### 2. Installation

Clone the repository and install the required dependencies:

```bash
git clone [https://github.com/sptin2002/ara_trader.git](https://github.com/sptin2002/ara_trader.git)
cd ara_trader
pip install -r requirements.txt

```

### 3. Setting up FinBERT Weights Locally

Because GitHub restricts files over 100MB, the model weights (`pytorch_model.bin`) are git-ignored. You must fetch them manually before executing the pipeline:

1. Create the target directory inside the root of your workspace:
```bash
mkdir -p finbert-local

```


2. Download the required model files from Hugging Face (`ProsusAI/finbert`):
* **Weights:** [pytorch_model.bin](https://www.google.com/search?q=https://huggingface.co/ProsusAI/finbert/resolve/main/pytorch_model.bin) (~418MB)
* **Config:** [config.json](https://www.google.com/search?q=https://huggingface.co/ProsusAI/finbert/resolve/main/config.json)
* **Vocabulary:** [vocab.txt](https://www.google.com/search?q=https://huggingface.co/ProsusAI/finbert/resolve/main/vocab.txt)


3. Place all three downloaded files directly inside the `finbert-local/` folder. Your local structure must mirror this exactly:
```text
ara_trader/
└── finbert-local/
    ├── config.json
    ├── pytorch_model.bin
    └── vocab.txt

```


### 4. Pipeline Automation Execution

The complete multi-stage execution pipeline is fully automated. To run data collection, data normalization, neural inference, unified ranking, and position sizing in sequence, execute the master orchestration shell script:

```bash
# For Linux / macOS systems
chmod +x run_predict.sh
./run_predict.sh

# For Windows systems
.\run_predict.ps1

```

Once execution completes successfully, check your generated file: `final_trade_orders.csv`.


---

## 🔄 Model Maintenance & Retraining

Financial markets are non-stationary, causing machine learning signals to experience statistical drift over time. To combat this decay, the Temporal Fusion Transformer network must be systematically retrained on a regular cadence (**weekly or monthly**) using out-of-sample data expansions.

To pull the latest historical feature arrays from your database and execute a supervised recalibration loop, run:

```bash
python3 train_refined_model.py

```

*Note: All newly generated model parameters will automatically be screened by the **Refinement Gate** guardrails ($\ge 55\%$ directional validation accuracy and $\le 12\%$ generalization gap) before replacing your active inference engine artifact.*


---

### 🚀 Model Maintenance Pipeline

You must execute the pipeline scripts in the exact chronological sequence defined below. Skipping steps or running them out of order will introduce lookahead bias or induce shape mismatches during inference.

```
[fetch_data.py]  -->  [data_to_qlib.py]  -->  [train_refined_model.py]
(Ingestion Layer)     (Binary Conversion)     (Model Engine Training)

```

#### 1. Ingestion Layer (`fetch_data.py`)

* **Role:** Establishes connection to the IBKR API gateway to synchronize historical price bars while pulling raw news headlines via the Polygon API.


* **Mechanics:** Resolves weekend kinetic vacuums by compounding weekend news volume into Monday trading boundaries and applies an implicit intraday cutoff to isolate text data from lookahead contamination.
* **Command:**
```bash
    python fetch_data.py


```


#### 2. Binary Compilation (`data_to_qlib.py`)
*   **Role:** Performs a complete infrastructure reset of the local acceleration layer and extracts multi-modal sequences out of the SQLite ledger.
*   **Mechanics:** Compiles and flattens price features along with raw text features (`open`, `high`, `low`, `close`, `volume`, `avg_sentiment`, `news_volume`) into optimized 32-bit floating-point binary blocks (`.bin`) mapped to an explicit Unix timestamp index.
*   **Command:**
```bash
    python data_to_qlib.py

```


#### 3. Model Training & Refinement Gate (`train_refined_model.py`)

* **Role:** Initializes Qlib's data provider to ingest the binary matrix and dynamically evaluates structural expressions mapped out in `qlib_factor_config.yaml`.


* **Mechanics:** Constructs a Temporal Fusion Transformer (TFT) style architecture featuring custom multi-head attention blocks and cross-modal layers. The model conditions technical vectors against volume-weighted sentiment metrics.
* **The Regularization Guardrail:** The script reads thresholds natively from `config.json`. If a symbol satisfies the validation constraints (`min_train_accuracy >= 0.55` and `validation_delta <= 0.12`), its compiled neural weights are exported to disk; otherwise, the artifact is rejected.


* **Command:**
```bash
python train_refined_model.py

```



### 📊 Downstream Production Handoff

Once the refinement training sequence completes, execution moves seamlessly to production inference and capital allocation:

1.  **`predict_refined_signals.py`**: Loads the verified `.keras` model checkpoints, pulls real-time factor matrices while isolating pure feature inputs from risk markers, and calculates cross-modal prediction probabilities.
2.  **`rank_and_filter.py`**: Compiles model inference outputs into a joint cross-sectional ranking schema, balancing alpha predictions against physical support-floor characteristics to establish an exponential capital decay multiplier.
3.  **`final_plan.py`**: Governs final trade orders by evaluating support/resistance levels to scale exact order sizing, outputting a risk-managed execution sheet ready for the IBKR Bridge.



---



## 🤝 Roadmap & Areas to Contribute

We are actively seeking contributors to expand the capacity of AraTradeSmart. If you specialize in any of the following fields, jump in!

* [ ] **Alternative Transformers:** Experiments adapting Informer, Autoformer, or PatchTST architectures to the multi-modal factor pipeline.
* [ ] **Low-Latency Ingestion:** Transitioning data pipelines to process real-time WebSockets stream arrays into memory-mapped files.
* [ ] **LLM Agent Routing:** Upgrading the local FinBERT structure to a specialized LLM routing agent that parses complex macroeconomic event transcripts.
* [ ] **Advanced Execution Algos:** Enhancing the IBKR bridge with local TWAP/VWAP execution slice logics to hide large volume footprints.


---

## 🛠️ Contribution Guidelines

1. **Fork the Repository:** Create a feature branch off of `main` (`git checkout -b feature/amazing-alpha-factor`).
2. **Keep it Modular:** Ensure code changes fit cleanly within the decoupled operational layers (Data, Brain, Gate, Risk, Execution).
3. **Write Unit Tests:** Verify execution logic, factor math, and risk boundaries in a corresponding test file.
4. **Submit a Pull Request:** Describe your improvements, provide validation metrics, and open the PR for review.


---

## 📜 License

Distributed under the **GNU General Public License v3.0 (GPL-3.0)**. See `LICENSE` for more information.


---

## 💬 Join the Discussion

Have questions, suggestions, or want to discuss alpha models?

* Open an **[Issue](https://github.com/sptin2002/ara_trader/issues)** for bugs or feature proposals.
* Start a thread in **[Discussions](https://github.com/sptin2002/ara_trader/discussions)** to share quant insights.

*Disclaimer: This software is for educational and research purposes only. Algorithmic trading carries substantial financial risk. Use at your own discretion.*