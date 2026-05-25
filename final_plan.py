import pandas as pd
import json
import os
from datetime import datetime

def load_config():
    with open('config.json', 'r') as f:
        return json.load(f)

class AraRiskEngine:
    def __init__(self):
        self.config = load_config()
        self.equity = self.config['settings'].get('account_equity', 30000)
        self.target_gain = self.config['model_settings'].get('target_gain_pct', 0.015)
        self.max_risk_per_trade = self.config['settings'].get('risk_per_trade', 0.02)

    def calculate_position_size(self, price, confidence, multiplier, atr):
        if price <= 0 or atr <= 0:
            return 0
            
        risk_amount = self.equity * self.max_risk_per_trade * multiplier
        confidence_adj = confidence / 0.70
        final_risk = risk_amount * confidence_adj
        
        stop_dist = atr * 2
        if stop_dist <= 0: 
            return 0
        
        return int(final_risk / stop_dist)

    def execute_plan(self):
        input_file = "final_execution_list.csv"
        output_file = "final_trade_orders.csv"
        
        if not os.path.exists(input_file):
            print(f"[!] Target file '{input_file}' missing. Run rank_and_filter.py first.")
            return

        df = pd.read_csv(input_file)
        if df.empty:
            print("[!] Target execution dataframe is empty. Terminating trade generation.")
            return

        trade_plan = []
        print(f"[*] ARA.AI Risk Engine: Processing {len(df)} Ranked Picks...")

        generation_date = datetime.now().strftime("%Y-%m-%d")

        for _, row in df.iterrows():
            try:
                ticker = str(row['Ticker'])
                
                # Column data extractions
                price = float(row['Price'])
                confidence = float(row['Confidence'])
                multiplier = float(row['Position_Multiplier'])
                atr = float(row['ATR'])
                
                nearest_sup = float(row['Nearest_Sup'])
                sup_strength = float(row['Sup_Strength'])
                nearest_res = float(row['Nearest_Res'])
                res_strength = float(row['Res_Strength'])

                if price <= 0:
                    print(f"[-] {ticker} skipped due to invalid price value: {price}")
                    continue

                # --- GUARDRAIL 1: DYNAMIC PROXIMITY FILTER ---
                # Avoid if the distance to a major resistance wall is smaller than your needed target_gain_pct
                if res_strength > 3.0 and price < nearest_res:
                    distance_to_res = (nearest_res - price) / price
                    if distance_to_res < self.target_gain:
                        print(f"[!] {ticker} AVOIDED: Distance to resistance ({distance_to_res:.2%}) is less than required target profit ({self.target_gain:.2%}). Ceiling wall at ${nearest_res:.2f}.")
                        continue

                # 1. Structural Stop Loss Placement
                if sup_strength > 3.0:
                    stop_loss = nearest_sup * 0.995  
                else:
                    stop_loss = price - (atr * 2)

                # --- GUARDRAIL 2: INVERSION FILTER ---
                if stop_loss >= price:
                    print(f"[!] {ticker} DISQUALIFIED: Calculated Stop Loss (${stop_loss:.2f}) is inverted relative to Entry (${price:.2f}).")
                    continue

                # 2. Standard Take Profit Placement
                take_profit = price * (1.0 + self.target_gain)

                # 3. Size Calculation
                shares = self.calculate_position_size(price, confidence, multiplier, atr)
                
                if shares > 0:
                    trade_plan.append({
                        "Date": generation_date,
                        "Rank": int(row['Rank']),
                        "Ticker": ticker,
                        "Action": "BUY",
                        "Quantity": shares,
                        "Entry_Price": round(price, 2),
                        "Stop_Loss": round(stop_loss, 2),
                        "Take_Profit": round(take_profit, 2),
                        "Conviction": f"{confidence:.1%}"
                    })
            except KeyError as ke:
                print(f"[CRITICAL] Column layout missing from input file: {ke}")
                return
            except Exception as e:
                print(f"[Error] Skipping candidate tracking for row: {e}")

        # Output Generation
        if trade_plan:
            new_plan_df = pd.DataFrame(trade_plan)
            
            if os.path.exists(output_file):
                try:
                    print(f"[*] Found existing trade log '{output_file}'. Loading historical orders...")
                    old_plan_df = pd.read_csv(output_file)
                    
                    # Ensure alignment of text data types before comparison/concatenation
                    old_plan_df['Date'] = old_plan_df['Date'].astype(str)
                    old_plan_df['Ticker'] = old_plan_df['Ticker'].astype(str)
                    
                    # Filter out matches from the same day to prevent duplicated processing rows
                    duplicate_mask = (old_plan_df['Date'] == generation_date) & (old_plan_df['Ticker'].isin(new_plan_df['Ticker']))
                    if duplicate_mask.any():
                        print(f"[*] Removing matching existing orders for {generation_date} to prevent duplicate execution rows.")
                        old_plan_df = old_plan_df[~duplicate_mask]
                    
                    # Concatenate the historical log frames with the fresh tactical layout
                    final_df = pd.concat([old_plan_df, new_plan_df], ignore_index=True)
                except Exception as ex:
                    print(f"[!] Error reading historical order vault: {ex}. Starting clean history.")
                    final_df = new_plan_df
            else:
                print(f"[*] Creating a fresh historical database trace file at '{output_file}'.")
                final_df = new_plan_df

            # Display and Save outputs
            print("\n" + "="*85)
            print("                       ARA-QLIB HISTORICAL ORDER TRACKING")
            print("="*85)
            print(final_df[['Date', 'Rank', 'Ticker', 'Quantity', 'Entry_Price', 'Stop_Loss', 'Take_Profit', 'Conviction']].to_string(index=False))
            print("="*85)
            
            final_df.to_csv(output_file, index=False)
            print(f"[*] Total history ({len(final_df)} entries) safely committed back into '{output_file}'.")
        else:
            print("[!] 0 trades generated. Historical execution file unchanged.")

if __name__ == "__main__":
    engine = AraRiskEngine()
    engine.execute_plan()