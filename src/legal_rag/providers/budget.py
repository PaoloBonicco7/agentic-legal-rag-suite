"""Persistent reservations across providers and process restarts."""

import fcntl
import json
import math
import os
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from uuid import uuid4

from legal_rag.artifacts import utc_now
from legal_rag.contracts import ProviderConfig
from .errors import BudgetExceeded, ProviderError


class BudgetLedger:
    def __init__(self, path: str | Path, config: ProviderConfig, *, purpose: str = "experiment"):
        self.path, self.config, self.purpose = Path(path), config, purpose
        if purpose not in {"experiment", "validation"}:
            raise ValueError("purpose must be experiment or validation")
        if config.budget_eur > 20 or config.budget_reserve_eur >= config.budget_eur:
            raise ValueError("Budget must not exceed 20 EUR and reserve must be below the cap")
        self._check_policy()

    def _check_policy(self):
        config = self.config
        if config.input_usd_per_million is None or config.output_usd_per_million is None or not all((config.price_date, config.price_source_url, config.fx_date, config.fx_source_url)):
            raise ProviderError("prices_missing", "OpenAI needs explicit prices, verification dates and source URLs for prices and EUR/USD.")
        try:
            price_date, fx_date = date.fromisoformat(config.price_date), date.fromisoformat(config.fx_date)
        except ValueError:
            raise ProviderError("prices_invalid", "Pricing verification dates must be ISO dates.") from None
        if price_date != date.today() or not 0 <= (date.today() - fx_date).days <= 4:
            raise ProviderError("prices_stale", "Verify today's prices and a EUR/USD rate from the last four days before paid calls.")
        if not all(math.isfinite(value) for value in (config.input_usd_per_million, config.output_usd_per_million, config.eur_per_usd)):
            raise ProviderError("prices_invalid", "Pricing values must be finite.")
        if not all(url.startswith(("https://", "http://")) for url in (config.price_source_url, config.fx_source_url)):
            raise ProviderError("prices_invalid", "Pricing and FX sources must be explicit HTTP URLs.")

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens * self.config.input_usd_per_million + output_tokens * self.config.output_usd_per_million) * self.config.eur_per_usd / 1_000_000

    def preflight(self) -> dict:
        self._check_policy()
        with self._locked() as data:
            if any(row["state"] == "bound_exceeded" for row in data["entries"]):
                raise ProviderError("budget_review_required", "A previous request exceeded its cost bound.")
        return self.snapshot()

    @contextmanager
    def _locked(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.with_suffix(self.path.suffix + ".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                data = json.loads(self.path.read_text()) if self.path.exists() else {"schema_version": "budget-v1", "entries": []}
                yield data
                temporary = self.path.with_suffix(self.path.suffix + ".tmp")
                temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
                os.replace(temporary, self.path)
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def reserve(self, amount: float, *, model: str, request_hash: str) -> str:
        self._check_policy()
        if not math.isfinite(amount) or amount < 0:
            raise ValueError("Reservation must be finite and nonnegative")
        with self._locked() as data:
            if any(row["state"] == "bound_exceeded" for row in data["entries"]):
                raise ProviderError("budget_review_required", "A previous request exceeded its cost bound.")
            total = sum(row["amount_eur"] for row in data["entries"])
            experiments = sum(row["amount_eur"] for row in data["entries"] if row["purpose"] == "experiment")
            validations = sum(row["amount_eur"] for row in data["entries"] if row["purpose"] == "validation")
            experiment_cap = self.config.budget_eur - self.config.budget_reserve_eur
            if total + amount > self.config.budget_eur or (self.purpose == "experiment" and experiments + amount > experiment_cap) or (self.purpose == "validation" and validations + amount > self.config.budget_reserve_eur):
                raise BudgetExceeded()
            reservation = uuid4().hex
            data["entries"].append({"id": reservation, "purpose": self.purpose, "amount_eur": amount, "reserved_eur": amount,
                "state": "reserved", "model": model, "request_hash": request_hash, "created_at": utc_now(),
                "price_date": self.config.price_date, "eur_per_usd": self.config.eur_per_usd,
                "price_source_url": self.config.price_source_url, "fx_date": self.config.fx_date, "fx_source_url": self.config.fx_source_url,
                "input_usd_per_million": self.config.input_usd_per_million, "output_usd_per_million": self.config.output_usd_per_million})
        return reservation

    def settle(self, reservation: str, actual: float | None) -> None:
        if actual is not None and (not math.isfinite(actual) or actual < 0):
            raise ValueError("Actual cost must be finite and nonnegative")
        with self._locked() as data:
            entry = next(row for row in data["entries"] if row["id"] == reservation)
            if actual is not None:
                if actual > entry["reserved_eur"] + 1e-9:
                    entry.update(state="bound_exceeded", amount_eur=actual)
                else:
                    entry.update(state="settled", amount_eur=actual)
            else:
                entry["state"] = "uncertain"

    def snapshot(self) -> dict:
        with self._locked() as data:
            return {"spent_or_reserved_eur": sum(row["amount_eur"] for row in data["entries"]),
                    "cap_eur": self.config.budget_eur, "reserve_eur": self.config.budget_reserve_eur,
                    "entries": len(data["entries"])}
