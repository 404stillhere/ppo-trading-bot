"""
PPO Crypto Trainer v2 — Bybit BTC/ETH/SOL/BNB
Pattern: trading_portfolio.py (yfinance) with Bybit data.
"""
import os, sys, time
from datetime import datetime, timezone
sys.path.insert(0, "M:/temp_downloads")
os.chdir("C:/OpenMythos")

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import torch
from dotenv import load_dotenv
from pybit.unified_trading import HTTP

# Force UTF-8 on Windows
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

load_dotenv("C:/OpenMythos/.env")
BYBIT_KEY    = os.getenv("BYBIT_API_KEY", "")
BYBIT_SECRET = os.getenv("BYBIT_API_SECRET", "")

# Config
TICKERS   = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
WINDOW    = 60
TRAIN_RATIO = 0.80
COMMISSION  = 0.0005

print("=" * 60)
print("  PPO Crypto Trainer v2 — BTC/ETH/SOL/BNB")
print("=" * 60)
print(f"  CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

# ─── DATA ────────────────────────────────────────────────────────────────────

def download_bybit(ticker):
    """Download up to 1000 hourly candles from Bybit."""
    try:
        client = HTTP(api_key=BYBIT_KEY, api_secret=BYBIT_SECRET)
        resp = client.get_kline(category="linear", symbol=ticker,
                                 interval=60, limit=1000)
        if resp.get("retCode") != 0:
            print(f"    [!] {ticker}: {resp['retMsg']}")
            return None
        rows = resp["result"]["list"]
        all_rows = []
        cursor = resp["result"].get("nextPageCursor")
        for r in rows:
            ts_ms = int(r[0])
            if ts_ms > 1e12:
                ts_ms //= 1000
            o, h, l, c, v = map(float, r[1:6])
            all_rows.append(dict(
                ts=datetime.fromtimestamp(ts_ms, tz=timezone.utc),
                open=o, high=h, low=l, close=c, volume=v,
            ))
        # Pagination
        while cursor and len(all_rows) < 2000:
            resp = client.get_kline(category="linear", symbol=ticker,
                                    interval=60, limit=200, cursor=cursor)
            if resp.get("retCode") != 0:
                break
            rows = resp["result"]["list"]
            if not rows:
                break
            for r in rows:
                ts_ms = int(r[0])
                if ts_ms > 1e12:
                    ts_ms //= 1000
                o, h, l, c, v = map(float, r[1:6])
                all_rows.append(dict(
                    ts=datetime.fromtimestamp(ts_ms, tz=timezone.utc),
                    open=o, high=h, low=l, close=c, volume=v,
                ))
            cursor = resp["result"].get("nextPageCursor")
            if not cursor:
                break
        df = pd.DataFrame(all_rows[::-1]).reset_index(drop=True)
        print(f"  {ticker}: {len(df)} bars | {df['ts'].iloc[0].date()} - {df['ts'].iloc[-1].date()}")
        return df
    except Exception as e:
        print(f"  [!] {ticker}: {e}")
        return None

# ─── INDICATORS ──────────────────────────────────────────────────────────────

def compute_indicators(close, high, low):
    n = len(close)
    # RSI-14 (vectorized)
    deltas = np.diff(close, prepend=close[0])
    gains  = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = np.convolve(gains,  np.ones(14)/14, mode="same")
    avg_loss = np.convolve(losses, np.ones(14)/14, mode="same")
    rs  = avg_gain / (avg_loss + 1e-10)
    rsi = (100 - (100 / (1 + rs))).astype(np.float32)

    # BB (20-period)
    sma20 = np.convolve(close, np.ones(20)/20, mode="same")
    std20 = np.array([close[max(0,i-20):i+1].std() for i in range(n)], dtype=np.float32)
    bb_up = (sma20 + 2 * std20).astype(np.float32)
    bb_lo = (sma20 - 2 * std20).astype(np.float32)

    # ATR-14
    tr = np.zeros(n-1, dtype=np.float32)
    for i in range(1, n):
        tr[i-1] = max(high[i], close[i-1]) - min(low[i], close[i-1])
    atr = np.convolve(tr, np.ones(14)/14, mode="same").astype(np.float32)

    # Returns
    pct_ret = np.concatenate([[0], np.diff(close) / (close[:-1] + 1e-10)]).astype(np.float32)

    # Volatility (annualized 20-period)
    vol = np.full(n, 0.0, dtype=np.float32)
    for i in range(20, n):
        v = np.std(pct_ret[i-20:i]) * np.sqrt(365 * 24)  # hourly
        vol[i] = v if not np.isnan(v) else 0.0

    return rsi, bb_up, bb_lo, atr, pct_ret, vol

# ─── ENV ──────────────────────────────────────────────────────────────────────

class CryptoTradeEnv(gym.Env):
    """
    obs = [rsi/100, bb_pos, atr_pct, vol, position] + returns_window(WINDOW)
    action = 0 (HOLD), 1 (BUY/open or close)
    """
    def __init__(self, close, high, low, rsi, bb_up, bb_lo, atr, pct_ret, vol,
                 commission=COMMISSION, window=WINDOW):
        super().__init__()
        self.close   = close.astype(np.float32)
        self.high    = high.astype(np.float32)
        self.low     = low.astype(np.float32)
        self.rsi     = np.nan_to_num(rsi, nan=50.0).astype(np.float32)
        self.bb_up   = bb_up.astype(np.float32)
        self.bb_lo   = bb_lo.astype(np.float32)
        self.atr     = atr.astype(np.float32)
        self._rets   = pct_ret.astype(np.float32)
        self._vol    = vol.astype(np.float32)
        self.commission = commission
        self.window   = window
        self.n        = len(close)
        self.max_step = self.n - window - 1
        self.current_step = window
        self.position = 0
        self.entry_price = 0.0

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(window + 5,),
            dtype=np.float32)

    def _get_obs(self):
        i = self.current_step
        # Returns window
        rets_win = self._rets[i - self.window:i].copy()
        if len(rets_win) < self.window:
            rets_win = np.pad(rets_win, (self.window - len(rets_win), 0))
        # Indicators
        rsi_v = float(self.rsi[i] / 100)
        bb_p  = float((self.close[i] - self.bb_lo[i]) /
                       (self.bb_up[i] - self.bb_lo[i] + 1e-10))
        atr_p = float(self.atr[i] / (self.close[i] + 1e-10))
        vol_v = float(self._vol[i])
        pos   = 1.0 if self.position else 0.0
        feat  = np.array([rsi_v, bb_p, atr_p, vol_v, pos], dtype=np.float32)
        obs   = np.concatenate([feat, rets_win])
        assert obs.shape == (self.window + 5,), f"obs shape {obs.shape} != {(self.window+5,)}"
        return obs

    def action_masks(self):
        return np.array([True, True], dtype=bool)

    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed)
        self.position    = 0
        self.entry_price = 0.0
        self.current_step = self.window
        return self._get_obs(), {}

    def step(self, action):
        i  = self.current_step
        c  = self.close[i]
        reward = 0.0

        if self.position != 0:
            reward = (c - self.entry_price) / (self.entry_price + 1e-10) - self.commission
            self.position    = 0
            self.entry_price = 0.0

        if action == 1:
            self.position    = 1
            self.entry_price = c

        self.current_step += 1
        done = self.current_step >= self.max_step
        return self._get_obs(), float(reward), done, False, {}


class MultiInputEnv(gym.Env):
    """Randomly picks ticker each step — SB3-compatible standalone env."""
    def __init__(self, envs):
        super().__init__()
        self.envs = envs
        self.n    = len(envs)
        self._active = 0
        self.observation_space = envs[0].observation_space
        self.action_space      = envs[0].action_space
        self._rng = np.random.default_rng(42)

    def _set_active(self):
        self._active = int(self._rng.integers(0, self.n))

    def reset(self, seed=None, **kwargs):
        self._set_active()
        obs, info = self.envs[self._active].reset(seed=seed)
        self._sync()
        return obs.astype(np.float32), info

    def _sync(self):
        e = self.envs[self._active]
        self.close        = e.close
        self.current_step = e.current_step
        self.position     = e.position
        self.entry_price  = e.entry_price

    def step(self, action):
        self._set_active()
        result = self.envs[self._active].step(int(action))
        obs, reward, done, trunc, info = result
        self._sync()
        return obs.astype(np.float32), reward, done, trunc, info

    def action_masks(self):
        return self.envs[self._active].action_masks()


# ─── VALIDATION ──────────────────────────────────────────────────────────────

def run_agent(model, env):
    obs, _ = env.reset()
    done   = False
    cum    = 0.0
    curr_pos, curr_entry = 0, 0.0
    trades = []

    while not done:
        mask = env.action_masks()
        act, _ = model.predict(obs, action_masks=mask, deterministic=True)
        prev_pos = curr_pos

        obs, _, done, _, _ = env.step(int(act))

        if env.position != prev_pos:
            if prev_pos == 1:
                pnl = (env.close[env.current_step-1] - curr_entry) / curr_entry * 100
                cum += pnl
                trades.append({"pnl": pnl/100})
            if env.position == 1:
                curr_pos, curr_entry = 1, env.close[env.current_step-1]
            else:
                curr_pos, curr_entry = 0, 0.0

    pnls = [t["pnl"] for t in trades]
    win  = sum(1 for p in pnls if p > 0) / max(len(pnls), 1) * 100 if pnls else 0
    return {"return": round(cum, 2), "n_trades": len(trades), "win_rate": round(win, 1)}


# ─── MAIN ─────────────────────────────────────────────────────────────────────

print("\n[1] Downloading Bybit data...")
all_data = {}
for ticker in TICKERS:
    df = download_bybit(ticker)
    if df is not None and len(df) >= WINDOW + 100:
        all_data[ticker] = df
    time.sleep(0.3)

print(f"\nDownloaded: {len(all_data)}/{len(TICKERS)} tickers")
if len(all_data) < len(TICKERS):
    print("[!] Some tickers failed — continuing anyway")

if not all_data:
    print("[!] No data. Exiting.")
    sys.exit(1)

print("\n[2] Building environments...")
train_envs = []
val_envs   = {}

for ticker, df in all_data.items():
    close = df["close"].values
    high  = df["high"].values
    low   = df["low"].values
    rsi, bb_up, bb_lo, atr, rets, vol = compute_indicators(close, high, low)

    split = int(len(close) * TRAIN_RATIO)
    env_tr = CryptoTradeEnv(
        close[:split], high[:split], low[:split],
        rsi[:split], bb_up[:split], bb_lo[:split],
        atr[:split], rets[:split], vol[:split],
        commission=COMMISSION, window=WINDOW)
    env_val = CryptoTradeEnv(
        close[split:], high[split:], low[split:],
        rsi[split:], bb_up[split:], bb_lo[split:],
        atr[split:], rets[split:], vol[split:],
        commission=COMMISSION, window=WINDOW)

    train_envs.append(env_tr)
    val_envs[ticker]  = env_val
    print(f"  {ticker}: train={split} bars, val={len(close)-split} bars")

multi_env = MultiInputEnv(train_envs)
print(f"  obs_dim={multi_env.observation_space.shape[0]}, n_envs={len(train_envs)}")

# Verify obs shape
test_obs, _ = multi_env.reset()
print(f"  test obs shape: {test_obs.shape}")

# Train
print("\n[3] Training MaskablePPO (200k steps)...")
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

ppo = MaskablePPO(
    MaskableActorCriticPolicy, multi_env,
    policy_kwargs=dict(net_arch=dict(pi=[128, 128, 64], vf=[128, 128, 64])),
    learning_rate=8e-5,
    n_steps=1024,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    ent_coef=0.02,
    verbose=1,
    seed=42,
)

ppo.learn(total_timesteps=200_000, progress_bar=True)

MODEL_OUT = "C:/OpenMythos/maskable_ppo_crypto_v2.zip"
ppo.save(MODEL_OUT)
print(f"\nModel saved: {MODEL_OUT}")

# Validate
print("\n[4] Per-ticker validation...")
total_ret = 0
for ticker, env in val_envs.items():
    m = run_agent(ppo, env)
    total_ret += m["return"]
    print(f"  {ticker:8s} | Ret={m['return']:>8.2f}% | Trades={m['n_trades']:>3d} | WinRate={m['win_rate']:>5.1f}%")

print(f"\n  Portfolio Avg Return: {total_ret/len(val_envs):.2f}%")
print("\n[OK] Training complete!")
