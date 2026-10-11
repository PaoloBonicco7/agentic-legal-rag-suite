# 04 — Immutable local index

## Purpose
Build a reproducible Qdrant local index from a verified corpus snapshot.

## Inputs
Corpus manifest and chunks.jsonl with verified source hashes.

## Outputs
A new index directory containing Qdrant storage and manifest.json. Existing directories are never overwritten.
The manifest records source hashes, embedding identity, dimensions, sparse
identity, code identity, count and readiness. Secrets are never serialized.
Interrupted output stays incomplete and cannot be resumed in place. Rebuild
under a fresh index path; verified model/text cache entries avoid repeated
embedding calls. An optional progress callback emits state initially, every
approximately 100 processed chunks and after readiness verification. CLI emits
these events to stderr; stdout is the final manifest. The monitoring contract
passed the offline suite on 2026-10-11; readiness still requires a completed
build with reconciled payloads and counts.

## Pipeline
- Verify recorded source file hashes and unique nonempty chunk identities.
- Embed embedding_text in batches of two. Default: pinned BGE-M3, CPU, float32,
  dense and native sparse in one forward pass. Cache is model/text addressed.
  MPS and float16 are explicit alternatives; float16 requires MPS. Device and
  precision are part of the identity, so cached representations cannot mix.
  MPS encoding pads to a multiple of 32 tokens; CPU keeps default padding.
  padding_multiple is part of the scientific identity, preventing reuse of
  representations produced with another padding policy. This limits sequence
  shapes to at most 16 lengths under the 512-token limit, as a hypothesis for
  reducing MPS driver allocation growth. Hardware profiling must verify it;
  no retrieval or answer-quality improvement is claimed from padding alone.
  BGE identity includes the adapter schema version and installed FlagEmbedding,
  transformers and torch package versions. BM25 records its adapter version and
  fastembed version. Metadata lookup never imports these optional libraries or
  loads weights; unavailable packages are recorded as unavailable. Library
  upgrades therefore require separate cache identities and indexes. The locked
  environment in uv.lock remains the record of all transitive dependencies.
  The manifest records upload batch_size and configured embedding_batch_size
  separately; neither splits model/text cache identity. Floating-point
  differences across batch sizes remain a limitation. An explicit experiment
  may group 64 uploads with inference batches of 8; this configuration requires
  measured memory stability and is not a portable default.
  After each MPS batch, release unused allocator buffers once outputs have been
  converted to CPU arrays. This cleanup does not change representations or retry
  inference. Runtime failures, including masked FlagEmbedding batch-adjustment
  errors, stop indexing with local_inference_error; the application does not
  retry inference or change the batch or device. FlagEmbedding itself attempts
  batch adaptation during its initial forward pass; configured batch size is
  not a guarantee about that internal behavior. Validate it over long runs and
  maximum-length inputs; a short throughput smoke does not establish stability.
- Alternatives: OpenAI or Utopia dense plus independent Italian BM25. BM25
  uses document/query-specific encoding and Qdrant IDF; BGE sparse does not.
  OpenAI validates nonempty strings, at most 2048 inputs, a conservative bound
  of at most 8192 tokens per input and 300000 for the request before touching
  the ledger or API. Each bound uses UTF-8 bytes plus the existing 2048-token
  allowance; lower configured input limits also apply. The batch allowance is
  added once to the sum of input bytes. These checks can reject valid inputs
  conservatively; they never truncate them. See the official
  [embedding request contract](https://developers.openai.com/api/reference/resources/embeddings/methods/create).
- Store named dense/sparse vectors and the exact Chunk payload under UUID5 IDs.
- Retain ceased and unknown content. Exclusion is a query-time policy.
- Mark ready only after finite-vector, dimension, payload and count checks.

## Contract
IndexManifest is a strict Pydantic record. The input limit is 512 BGE tokens
including special tokens. Retrieval verifies the dense and sparse identities;
opening the store verifies dimensions, named vectors and exact point count.
All point UUID5 IDs and payloads are checked against the frozen source chunks.
Vector cache v2 records hash model identity, source text and canonical vectors.
Legacy entries are re-embedded by default; explicit legacy reuse is recorded.
The migration helper verifies persisted points against source payloads and both
float32 cosine projections (absolute tolerance 1e-6, relative tolerance zero),
then writes v2 entries and a checksummed audit companion. Partial stores require
allow_incomplete=True; cache entries without persisted points remain unverified.

## Quality Gates
No duplicate IDs, zero-vector dense representations, nonfinite values, missing
hashes, mixed embedding identities or partial indexes are accepted. Changing
representation requires a separate index. Qdrant local payload indexes do not
accelerate searches; no HNSW claims or ANN sweeps are made.

## Notebook Role
Notebooks call the same Python entry points as the CLI. Offline tests use
synthetic embeddings and Qdrant memory mode.

## Acceptance Criteria
Real model/provider runs require
explicit commands; API errors cannot trigger another provider.

Implementation references for these acceptance criteria:
[BGE-M3](https://arxiv.org/abs/2402.03216),
[Qdrant hybrid queries](https://qdrant.tech/documentation/search/hybrid-queries/),
[FastEmbed BM25](https://github.com/qdrant/fastembed/blob/main/fastembed/sparse/bm25.py).
