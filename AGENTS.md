# Working agreement

- Explain work in Italian; write code, comments and specifications in English.
- Read the relevant numbered specification before changing behavior.
- Keep core logic in Python modules; notebooks and UI only call those modules.
- Use strict Pydantic artifact contracts. Never export credentials.
- Preserve source snapshots and the original evaluation CSVs.
- Runtime retrieval and generation must never read gold answers or qrel files.
- Modify gold only through an explicitly approved, versioned review.
- Use the shared context builder and fixed budgets for every controlled comparison.
- Record unsuccessful experiments and incomplete runs honestly.
- Graph relationships and status filters require local source evidence.
- Do not infer validity from publication date or the words "Testo vigente".
- One agent owns shared contracts; agree contract changes before implementation.
- Never run concurrent writers on an index, run checkpoint, or budget ledger.
