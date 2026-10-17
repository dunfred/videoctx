from __future__ import annotations

from dataclasses import dataclass, field

# USD per million tokens (input, output). Update as pricing changes.
PRICES: dict[str, tuple[float, float]] = {
    "claude-opus-4-8": (5.00, 25.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "gemini-native": (0.0, 0.0),  # filled in by the eval baseline, provider-specific
}


@dataclass
class CostMeter:
    """Accumulates token usage across API calls and prices it."""

    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    calls: int = 0
    notes: list[str] = field(default_factory=list)

    def add(self, usage) -> None:
        self.calls += 1
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens
        self.cache_read_tokens += getattr(usage, "cache_read_input_tokens", 0) or 0

    @property
    def usd(self) -> float:
        inp, out = PRICES.get(self.model, (0.0, 0.0))
        cost = (self.input_tokens * inp + self.output_tokens * out) / 1_000_000
        cost += self.cache_read_tokens * inp * 0.1 / 1_000_000
        return cost

    def summary(self) -> dict:
        return {
            "model": self.model,
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "usd": round(self.usd, 6),
        }
