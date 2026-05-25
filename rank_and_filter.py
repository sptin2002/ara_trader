import pandas as pd
import numpy as np
import json
import os

def load_config():
    with open('config.json', 'r') as f:
        return json.load(f)

def run_unified_ranking():
    config = load_config()
    input_file = "ml_candidates.csv"
    output_file = "final_execution_list.csv"
    
    if not os.path.exists(input_file):
        print("[!] No candidates found. Ensure inference ran successfully.")
        return

    df = pd.read_csv(input_file)
    if df.empty:
        print("[!] Candidates list is empty. No signals pass current thresholds.")
        return

    print(f"[*] Applying Unified Ranking to {len(df)} candidates...")

    # --- 1. COMPOSITE SCORING ENGINE ---
    df['norm_sup_str'] = df['Sup_Strength'].clip(0, 10) / 10.0
    
    df['Composite_Score'] = (
        (df['Confidence'] * 0.7) + 
        (df['norm_sup_str'] * 0.3)
    )

    # --- 2. CROSS-SECTIONAL RANKING ---
    df = df.sort_values(by='Composite_Score', ascending=False).reset_index(drop=True)
    df['Rank'] = df.index + 1

    # --- 3. RISK ENGINE PREPARATION ---
    top_k = config['settings'].get('top_k_picks', 5)
    final_list = df.head(top_k).copy()

    # --- 4. DYNAMIC POSITION SCORING ---
    final_list['Position_Multiplier'] = np.exp(-0.2 * (final_list['Rank'] - 1))
    
    # Precise rounding without deleting or dropping other core structural columns
    final_list = final_list.round({
        'Confidence': 3,
        'Composite_Score': 3,
        'Position_Multiplier': 2
    })

    # --- 5. LOGGING & OUTPUT ---
    print("\n--- ARA.AI FINAL EXECUTION LIST ---")
    for _, row in final_list.iterrows():
        print(f"RANK {int(row['Rank'])}: {row['Ticker']} | Score: {row['Composite_Score']:.3f} | Mult: {row['Position_Multiplier']:.2f}x")

    # Save the FULL dataframe containing all named structural metrics
    final_list.to_csv(output_file, index=False)
    print(f"\n[*] Exported {len(final_list)} picks to {output_file}. Ready for final_plan.py.")

if __name__ == "__main__":
    run_unified_ranking()