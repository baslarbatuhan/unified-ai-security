"""Output Guard — analyzes model responses for PII leaks, leaked credentials,
unsafe instructions the model might smuggle back, downstream injection, and
hallucinated redirects to untrusted destinations.

The module is intentionally content-only: it never sees the original prompt,
which keeps the attack surface small and makes it trivial to slot into the
fusion pipeline as the last stage.
"""
