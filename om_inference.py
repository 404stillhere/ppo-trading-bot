"""
OpenMythos inference wrapper for trading decisions.

Wraps M:/OpenMythos/ to make BUY/SELL/HOLD decisions from structured market data.
"""

import sys
sys.path.insert(0, 'M:/OpenMythos')

import re
import traceback
from typing import Optional

import torch
from open_mythos import OpenMythos, mythos_1b


# ---------------------------------------------------------------------------
# Character-level tokenizer — IDs in [0, 255], guaranteed < vocab_size 32000
# ---------------------------------------------------------------------------

class CharTokenizer:
    """
    Simple character-level tokenizer. Vocabulary = 256 ASCII codes.

    Maps every character (including newlines, spaces) to an integer ID in
    [0, 255]. Since vocab_size in mythos_1b() is 32000, all IDs are valid.
    """
    PAD_ID = 0

    def __init__(self):
        # ASCII 0..255 — each character is its own token
        self._vocab_size = 256

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def encode(self, text: str) -> list[int]:
        """
        Encode a string to a list of byte values (0–255).

        Non-ASCII characters are encoded as UTF-8 byte sequences,
        so any character maps to valid IDs in [0, 255].
        """
        return list(text.encode('utf-8'))

    def decode(self, token_ids: list[int]) -> str:
        """Decode byte-value token IDs back to a string."""
        byte_vals = [max(0, min(255, int(b))) for b in token_ids]
        return bytes(byte_vals).decode('utf-8', errors='replace')


# Global tokenizer instance (lazy — created once per process)
_char_tokenizer = None


def _get_tokenizer() -> CharTokenizer:
    global _char_tokenizer
    if _char_tokenizer is None:
        _char_tokenizer = CharTokenizer()
    return _char_tokenizer


# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------

def _rsi_label(rsi: float) -> str:
    if rsi < 30:
        return "oversold"
    if rsi < 40:
        return "slightly oversold"
    if rsi < 50:
        return "bearish"
    if rsi < 60:
        return "bullish"
    if rsi < 70:
        return "slightly overbought"
    return "overbought"


def _bb_label(pos: float) -> str:
    if pos < 0.25:
        return "lower band"
    if pos < 0.5:
        return "lower half"
    if pos < 0.75:
        return "upper half"
    return "upper band"


def _vol_label(v: float) -> str:
    if v < 30:
        return "low"
    if v < 60:
        return "moderate"
    if v < 90:
        return "high"
    return "very high"


# ---------------------------------------------------------------------------
# MarketDataPrompt
# ---------------------------------------------------------------------------

class MarketDataPrompt:
    """Converts structured market data into an OpenMythos trading prompt."""

    def __init__(self):
        self.tokenizer = _get_tokenizer()

    def build(
        self,
        ticker: str,
        date: str,
        close: float,
        rsi: float,
        bb_pos: float,
        atr_pct: float,
        vol: float,
        position: str,
        returns: list[float],
    ) -> str:
        """
        Build a structured trading prompt.

        Parameters
        ----------
        ticker   : e.g. "BTCUSDT"
        date     : e.g. "2024-03-15"
        close    : current close price
        rsi      : RSI(14) value
        bb_pos   : Bollinger Band position [0, 1]
        atr_pct  : ATR as percentage of price
        vol      : annualized volatility in percent
        position : current holding, e.g. "flat", "long", "short"
        returns  : list of recent period returns in percent, e.g. [+1.2, -0.8]

        Returns
        -------
        Formatted prompt string ready for tokenization.
        """
        rsi_str = f"{rsi:.1f}"
        rsi_lbl = _rsi_label(rsi)
        bb_lbl  = _bb_label(bb_pos)
        vol_lbl = _vol_label(vol)

        ret_str = ", ".join(f"+{r:.1f}%" if r >= 0 else f"{r:.1f}%"
                            for r in returns)

        lines = [
            f"<Ticker: {ticker}> <Date: {date}>",
            f"Price: ${close:,.2f} | RSI(14): {rsi_str} [{rsi_lbl}]",
            f"Bollinger: {bb_pos:.2f} [{bb_lbl}] | ATR: {atr_pct:.1f}%",
            f"Volatility: {vol:.1f}% annualized | Position: {position}",
            f"Last {len(returns)} returns: {ret_str}",
            "Should I BUY, SELL, or HOLD?",
            "Decision: DECISION:",
        ]
        return "\n".join(lines)

    def tokenize(self, prompt: str) -> torch.Tensor:
        """Encode prompt to token IDs. Returns shape (1, T)."""
        ids = self.tokenizer.encode(prompt)
        return torch.tensor([[self.tokenizer.PAD_ID] + ids], dtype=torch.long)

    def decode(self, token_ids: torch.Tensor) -> str:
        """Decode token IDs back to a string."""
        ids = token_ids
        if ids.ndim == 2:
            ids = ids[0]
        return self.tokenizer.decode(ids.tolist())


# ---------------------------------------------------------------------------
# MythosTrader
# ---------------------------------------------------------------------------

class MythosTrader:
    """
    Loads an OpenMythos model and wraps it for single-shot trading decisions.

    Parameters
    ----------
    model_path : str, optional
        Path to a saved checkpoint (torch.save). If None, the model is
        initialized randomly from ``mythos_1b()`` (expected until fine-tuning).
    device     : str, optional
        "cuda" or "cpu". Defaults to CUDA if available.
    """

    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        cfg = mythos_1b()
        self.model = OpenMythos(cfg)

        if model_path is not None:
            state = torch.load(model_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state, strict=False)
            print(f"[MythosTrader] Loaded checkpoint from {model_path}")
        else:
            print(f"[MythosTrader] No checkpoint provided - using random init (expected)")

        self.model = self.model.to(self.device)
        self.model.eval()
        self.prompt_builder = MarketDataPrompt()

        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"[MythosTrader] Model running on {self.device} | {n_params:,} params")

    def decide(
        self,
        ticker: str,
        date: str,
        close: float,
        rsi: float,
        bb_pos: float,
        atr_pct: float,
        vol: float,
        position: str,
        returns: list[float],
        max_new_tokens: int = 30,
        n_loops: int = 4,
        temperature: float = 0.2,
        top_k: int = 10,
    ) -> tuple[str, str]:
        """
        Run a single inference pass and return a trading decision.

        Parameters
        ----------
        ticker         : asset ticker, e.g. "BTCUSDT"
        date           : date string, e.g. "2024-03-15"
        close          : current close price
        rsi            : RSI(14) value
        bb_pos         : Bollinger Band position [0, 1]
        atr_pct        : ATR as % of price
        vol            : annualized volatility (%)
        position       : "flat", "long", or "short"
        returns        : recent period returns in % (e.g. [+1.2, -0.8])
        max_new_tokens : max tokens to generate (default 30)
        n_loops        : OpenMythos loop depth (default 4)
        temperature    : sampling temperature (default 0.2)
        top_k          : top-K sampling threshold (default 10)

        Returns
        -------
        (decision, reasoning)
            decision : "BUY", "SELL", or "HOLD"
            reasoning: human-readable text extracted after "DECISION:"
        """
        prompt = self.prompt_builder.build(
            ticker, date, close, rsi, bb_pos, atr_pct, vol, position, returns
        )
        input_ids = self.prompt_builder.tokenize(prompt).to(self.device)

        print(f"\n[MythosTrader] Prompt:\n{prompt}\n")

        try:
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    n_loops=n_loops,
                    temperature=temperature,
                    top_k=top_k,
                )

            generated_ids = output_ids[0][input_ids.shape[1]:]
            raw_text = self.prompt_builder.decode(generated_ids)

            try:
                print(f"[MythosTrader] Raw generation:\n{raw_text}\n")
            except UnicodeEncodeError:
                safe = raw_text.encode('ascii', 'replace').decode('ascii')
                print(f"[MythosTrader] Raw generation:\n{safe}\n")

            decision, reasoning = self._parse_decision(raw_text)
            return decision, reasoning

        except Exception:
            print(f"[MythosTrader] Inference error:\n{traceback.format_exc()}")
            return "HOLD", "Inference failed — defaulting to HOLD."

    def _parse_decision(self, text: str) -> tuple[str, str]:
        """
        Extract DECISION: XXX and the reasoning that follows it.

        Looks for the first occurrence of ``DECISION: BUY``, ``DECISION: SELL``,
        or ``DECISION: HOLD`` (case-insensitive) and returns the matched label
        along with everything after it as reasoning.

        If nothing matches, returns ("HOLD", text) so the full generation is
        preserved for inspection.
        """
        text = text.strip()

        # Normalise: collapse repeated "DECISION:" tokens that are part of the prompt
        text_clean = re.sub(r'Decision:\s*', 'Decision: ', text, flags=re.IGNORECASE)

        # Case-insensitive match
        m = re.search(r'DECISION:\s*(BUY|SELL|HOLD)', text_clean, re.IGNORECASE)
        if m:
            decision = m.group(1).upper()
            reasoning = text_clean[m.start():].strip()
            return decision, reasoning

        # Fallback: look for any bare BUY / SELL / HOLD in the generated portion
        m2 = re.search(r'\b(BUY|SELL|HOLD)\b', text_clean, re.IGNORECASE)
        if m2:
            decision = m2.group(1).upper()
            reasoning = text_clean.strip()
            return decision, reasoning

        return "HOLD", text


# ---------------------------------------------------------------------------
# CLI / smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("OpenMythos Trading Inference - smoke test")
    print("=" * 60)

    # Construct a realistic-but-synthetic BTC scenario
    scenario = dict(
        ticker    = "BTCUSDT",
        date      = "2024-03-15",
        close     = 67234.50,
        rsi       = 65.2,
        bb_pos    = 0.72,
        atr_pct   = 2.3,
        vol       = 78.5,
        position  = "flat",
        returns   = [1.2, -0.8, 2.1, 0.3, -1.5],
    )

    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"Scenario: {scenario['ticker']} on {scenario['date']}")
    print(f"  Close:  ${scenario['close']:,.2f}")
    print(f"  RSI:    {scenario['rsi']}")
    print(f"  BB pos: {scenario['bb_pos']}")
    print(f"  ATR:    {scenario['atr_pct']}%")
    print(f"  Vol:    {scenario['vol']}%")
    print(f"  Pos:    {scenario['position']}")
    print(f"  Returns:{scenario['returns']}")
    print("-" * 60)

    trader = MythosTrader(model_path=None)

    decision, reasoning = trader.decide(**scenario)

    print("=" * 60)
    print(f"Final decision : {decision}")
    safe_reasoning = reasoning.encode('ascii', 'replace').decode('ascii')
    print(f"Reasoning (ASCII):\n{safe_reasoning}")
    print("=" * 60)