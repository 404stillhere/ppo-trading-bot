"""
HybridCryptoTrader — PPO execution + Bybit live trading.
Dry-run по умолчанию. Для реальных сделок: DRY_RUN=False.
"""
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv

# Bybit
from bybit_trader import BybitTrader

# PPO + trading env
from sb3_contrib import MaskablePPO
from gymnasium import spaces
import gymnasium as gym

load_dotenv()

BYBIT_KEY = os.getenv("BYBIT_API_KEY", "")
BYBIT_SECRET = os.getenv("BYBIT_API_SECRET", "")
DRY_RUN = os.getenv("DRY_RUN", "true").lower() in ("true", "1", "yes")

MODEL_PATH = "maskable_ppo_crypto_v2.zip"
TICKERS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
COMMISSION = 0.0005  # Bybit spot maker ~0.1%, taker 0.1%
STOP_LOSS_PCT = 0.03   # -3% trailing stop (FIX #5)
TAKE_PROFIT_PCT = 0.06  # +6% take-profit (FIX #5)


# ─────────────────────────────────────────────────────────────────────────────
# Indicators
# ─────────────────────────────────────────────────────────────────────────────

def compute_indicators(closes: np.ndarray, window: int = 60):
    """Return rsi, bb_pos, atr_pct from close prices."""
    c = closes
    n = len(c)

    # RSI-14
    deltas = np.diff(c, prepend=c[0])
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = np.mean(gains[:14]) if n >= 14 else np.mean(gains)
    avg_loss = np.mean(losses[:14]) if n >= 14 else np.mean(losses)
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))

    # Bollinger Bands
    sma = np.convolve(c, np.ones(window) / window, mode="valid")
    std = np.array([np.std(c[i : i + window]) for i in range(len(sma))])
    bb_pos = np.zeros(n)
    sma_full = np.zeros(n)
    std_full = np.zeros(n)
    sma_full[window - 1 :] = sma
    std_full[window - 1 :] = std
    bb_pos = (c - sma_full) / (2 * std_full + 1e-10)
    bb_pos = np.clip(bb_pos, 0, 1)

    # ATR (simplified: high-low close)
    if n < 14:
        atr_pct = 0.0
    else:
        tr = np.zeros(n - 1)
        for i in range(1, n):
            h = max(c[i], c[i - 1])
            l = min(c[i], c[i - 1])
            tr[i - 1] = h - l
        atr = np.mean(tr[-14:])
        atr_pct = (atr / c[-1]) * 100

    return float(rsi), float(bb_pos[-1]), float(atr_pct)


def get_market_data(ticker: str, limit: int = 500):
    """Download hourly candles from Bybit."""
    try:
        from pybit.unified_trading import HTTP
        client = HTTP(api_key=BYBIT_KEY, api_secret=BYBIT_SECRET)
        resp = client.get_kline(
            category="linear",
            symbol=ticker,
            interval=60,
            limit=limit,
        )
        if resp.get("retCode") != 0:
            print(f"  [!] Bybit error {ticker}: {resp.get('retMsg')}")
            return None
        raw = resp["result"]["list"]
        if not raw:
            return None
        rows = []
        for r in reversed(raw):
            ts_ms = int(r[0])
            if ts_ms > 1e12:
                ts_ms //= 1000
            open_, high, low, close, vol = map(float, r[1:6])
            rows.append({
                "open_time": datetime.fromtimestamp(ts_ms, tz=timezone.utc),
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": float(vol),
            })
        import pandas as pd
        df = pd.DataFrame(rows)
        df = df.sort_values("open_time").reset_index(drop=True)
        return df
    except Exception as e:
        print(f"  [!] Download failed {ticker}: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Minimal TradeEnv (gymnasium)
# ─────────────────────────────────────────────────────────────────────────────

class CryptoEnv(gym.Env):
    """Single-ticker step environment for inference. obs_dim = 5 + window."""

    metadata = {"render_modes": []}

    def __init__(self, df: np.ndarray, window: int = 60):
        super().__init__()
        self.df = df
        self.n = len(df)
        self.close = df[:, 3].astype(np.float32)  # close column
        self.window = window
        self.current_step = window - 1
        self.max_step = self.n - 1
        self.position = 0
        self.entry_price = 0.0

        # 5 indicators + WINDOW returns = 65 (for window=60)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(5 + window,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(2)  # 0=HOLD, 1=BUY/close

    def _get_obs(self) -> np.ndarray:
        i = min(self.current_step, self.n - 1)
        rsi, bb_pos, atr_pct = compute_indicators(self.close[: i + 1])
        vol = float(np.std(self.close[max(0, i - 20) : i + 1]) / (np.mean(self.close[max(0, i - 20) : i + 1]) + 1e-10) * 100)
        pos = float(self.position)
        # Returns window (last window closes)
        rets = []
        for j in range(max(0, i - self.window + 1), i + 1):
            if j > 0:
                rets.append((self.close[j] - self.close[j - 1]) / (self.close[j - 1] + 1e-10) * 100)
        ret_win = np.array(rets[-self.window :] if len(rets) >= self.window else np.pad(rets, (self.window - len(rets), 0)))
        return np.concatenate([[rsi, bb_pos, atr_pct, vol, pos], ret_win]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window - 1
        self.position = 0
        self.entry_price = 0.0
        return self._get_obs(), {}

    def step(self, action: int):
        i = min(self.current_step, self.n - 1)
        c = self.close[i]
        reward = 0.0

        # FIX #1: check action BEFORE touching position
        # FIX #5: SL/TP checked after every action on open position
        if action == 0:
            # HOLD — only close via SL/TP if a position is open
            if self.position != 0:
                pnl_pct = (c - self.entry_price) / (self.entry_price + 1e-10)
                # Stop-loss: -3%, take-profit: +6%
                if pnl_pct <= -STOP_LOSS_PCT:
                    reward = pnl_pct - self.commission
                    self.position = 0
                    self.entry_price = 0.0
                    self._log_reason = "SL"
                elif pnl_pct >= TAKE_PROFIT_PCT:
                    reward = pnl_pct - self.commission
                    self.position = 0
                    self.entry_price = 0.0
                    self._log_reason = "TP"
                # else: keep position, no reward this step
        elif action == 1:
            if self.position == 0:
                # Open long
                self.position = 1
                self.entry_price = c
            else:
                # Close — check SL/TP first for live mode safety
                pnl_pct = (c - self.entry_price) / (self.entry_price + 1e-10)
                if pnl_pct <= -STOP_LOSS_PCT:
                    reward = pnl_pct - self.commission
                    self._log_reason = "SL"
                elif pnl_pct >= TAKE_PROFIT_PCT:
                    reward = pnl_pct - self.commission
                    self._log_reason = "TP"
                else:
                    reward = pnl_pct - self.commission
                self.position = 0
                self.entry_price = 0.0

        self.current_step += 1
        done = self.current_step >= self.max_step
        truncated = False
        obs = self._get_obs()
        reason = getattr(self, "_log_reason", "")
        if self.position == 0 and reason:
            delattr(self, "_log_reason")
        info = {"price": float(c), "position": self.position,
                "rsi": None, "bb_pos": None, "reason": reason}
        return obs, reward, done, truncated, info

    def action_masks(self) -> np.ndarray:
        return np.array([True, True], dtype=bool)


# ─────────────────────────────────────────────────────────────────────────────
# HybridCryptoTrader
# ─────────────────────────────────────────────────────────────────────────────

class HybridCryptoTrader:
    """
    Connects PPO (execution) + Bybit (live trading).
    Optional OpenMythos strategic layer (once trained).
    """

    def __init__(self, dry_run: bool = True):
        self.dry_run = dry_run
        self.tickers = TICKERS
        self.window = 60

        # Load PPO
        if Path(MODEL_PATH).exists():
            self.ppo = MaskablePPO.load(MODEL_PATH)
            print(f"[Trader] PPO loaded: {MODEL_PATH}")
        else:
            self.ppo = None
            print("[Trader] WARNING: No PPO model found — will print signals only")

        # Bybit client
        if not dry_run and BYBIT_KEY and BYBIT_SECRET:
            self.bybit = BybitTrader(BYBIT_KEY, BYBIT_SECRET, testnet=False)
            print(f"[Trader] Bybit LIVE — balance: {self.bybit.get_balance():.2f} USDT")
        else:
            self.bybit = None
            print("[Trader] Bybit DRY-RUN mode")

        self.log_file = Path("trades_log.csv")

    def _log_trade(self, ticker: str, action: str, price: float, balance: float, equity_pct: float):
        ts = datetime.now().isoformat()
        row = f"{ts},{ticker},{action},{price:.4f},{balance:.2f},{equity_pct:.4f}\n"
        with open(self.log_file, "a") as f:
            f.write(row)

    def _run_ticker(self, ticker: str, df, equity_start: float) -> dict:
        """Run PPO inference on a single ticker and execute."""
        prices = df[["open", "high", "low", "close", "volume"]].values

        env = CryptoEnv(prices, window=self.window)
        obs, _ = env.reset()

        # Use last WINDOW bars (recent market context)
        env.current_step = env.max_step
        obs = env._get_obs()  # shape (25,)

        mask = env.action_masks()
        action_idx, _ = self.ppo.predict(obs, action_masks=mask, deterministic=True)

        price = float(df["close"].iloc[-1])
        balance = self.bybit.get_balance() if self.bybit else 0.0
        position = self.bybit.get_position(ticker) if self.bybit else 0.0

        action_map = {0: "HOLD", 1: "BUY/CLOSE"}
        action_str = action_map.get(int(action_idx), "HOLD")

        rsi, bb_pos, atr_pct = compute_indicators(env.close)

        result = {
            "ticker": ticker,
            "price": price,
            "action": action_str,
            "action_idx": int(action_idx),
            "rsi": rsi,
            "bb_pos": bb_pos,
            "atr_pct": atr_pct,
            "position": position,
            "balance": balance,
            "equity_pct": 0.0,
        }

        # Execute
        if self.dry_run:
            print(f"  {ticker:8s} | price={price:>10.2f} | action={action_str:10s} | "
                  f"RSI={rsi:5.1f} | BB={bb_pos:.2f} | ATR={atr_pct:.1f}% | "
                  f"pos={position} | dry-run")
            return result

        # Live execution
        try:
            if action_idx == 1:
                # BUY — close short first if any, then buy
                if position < 0:
                    qty = abs(position)
                    self.bybit.buy_market(ticker, qty)
                    time.sleep(0.5)
                # ATR-based position sizing (FIX #6)
                # 1% risk per trade; stop = ATR, so qty = risk / ATR_dollar
                atr_pct = atr_pct  # from compute_indicators above
                risk_amount = balance * 0.01
                atr_dollar = price * (atr_pct / 100)
                qty = risk_amount / atr_dollar if atr_dollar > 0 else balance * 0.25 / price
                qty = max(qty, balance * 0.02 / price)  # cap at 2% min
                order_id = self.bybit.buy_market(ticker, qty)
                if order_id:
                    print(f"  {ticker:8s} BUY  {qty:.6f} @ {price:.2f} — order {order_id}")
            elif action_idx == 0 and position > 0:
                # HOLD — check SL/TP (FIX #5)
                entry = self.bybit.get_entry_price(ticker) if hasattr(self.bybit, "get_entry_price") else price * 0.97
                pnl_pct = (price - entry) / (entry + 1e-10)
                if pnl_pct <= -STOP_LOSS_PCT or pnl_pct >= TAKE_PROFIT_PCT:
                    qty = position
                    self.bybit.sell_market(ticker, qty)
                    reason = "SL" if pnl_pct < 0 else "TP"
                    print(f"  {ticker:8s} {reason}  {qty:.6f} @ {price:.2f}")
        except Exception as e:
            print(f"  [!] Execution error {ticker}: {e}")

        return result

    def run_cycle(self) -> list[dict]:
        """Fetch data for all tickers, run PPO, execute."""
        print(f"\n{'='*60}")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Trading cycle — "
              f"{'DRY-RUN' if self.dry_run else 'LIVE'}")
        print(f"{'='*60}")

        results = []
        for ticker in self.tickers:
            try:
                df = get_market_data(ticker, limit=500)
                if df is None or len(df) < self.window + 5:
                    print(f"  [!] {ticker}: insufficient data")
                    continue

                result = self._run_ticker(ticker, df, 1.0)
                results.append(result)
                self._log_trade(
                    result["ticker"],
                    result["action"],
                    result["price"],
                    result["balance"],
                    result["equity_pct"],
                )
            except Exception as e:
                print(f"  [!] {ticker} error: {traceback.format_exc()}")
                continue

        print(f"\nCycle complete: {len(results)} tickers processed")
        return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"DRY_RUN={DRY_RUN}")
    print(f"Bybit key set: {bool(BYBIT_KEY)}")

    trader = HybridCryptoTrader(dry_run=DRY_RUN)

    if len(sys.argv) > 1 and sys.argv[1] == "--once":
        # Single cycle
        results = trader.run_cycle()
        for r in results:
            print(f"  {r['ticker']}: {r['action']} @ {r['price']:.2f}")
    else:
        # Continuous loop
        print("Starting trading loop (Ctrl+C to stop)...")
        while True:
            try:
                trader.run_cycle()
                print("Sleeping 60 minutes until next cycle...")
                time.sleep(3600)
            except KeyboardInterrupt:
                print("\nStopped.")
                break
            except Exception as e:
                print(f"[!] Cycle error: {e}")
                time.sleep(300)
