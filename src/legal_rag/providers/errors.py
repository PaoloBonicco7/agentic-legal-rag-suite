"""Provider failures with secret-free diagnostics."""

from legal_rag.contracts import Usage


class ProviderError(RuntimeError):
    def __init__(self, code: str, message: str, *, usage: list[Usage] | None = None, raw_payload: dict | None = None):
        self.code = code
        self.usage = usage or []
        self.raw_payload = raw_payload
        super().__init__(f"{code}: {message}")


class BudgetExceeded(ProviderError):
    def __init__(self):
        super().__init__("budget_exceeded", "The persistent OpenAI spending limit would be exceeded.")
