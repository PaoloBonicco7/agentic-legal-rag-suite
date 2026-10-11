"""Provider failures with secret-free diagnostics."""

from legal_rag.contracts import Usage

NOT_ATTEMPTED_CODES = {"budget_exceeded", "budget_review_required", "prices_stale", "prices_missing", "prices_invalid", "credentials_missing", "context_unverified"}
TERMINAL_CODES = NOT_ATTEMPTED_CODES | {"http_401", "http_403", "http_404", "cost_bound_exceeded", "model_mismatch"}


class ProviderError(RuntimeError):
    def __init__(self, code: str, message: str, *, usage: list[Usage] | None = None, raw_payload: dict | None = None):
        self.code = code
        self.usage = usage or []
        self.raw_payload = raw_payload
        super().__init__(f"{code}: {message}")


class BudgetExceeded(ProviderError):
    def __init__(self):
        super().__init__("budget_exceeded", "The persistent OpenAI spending limit would be exceeded.")
