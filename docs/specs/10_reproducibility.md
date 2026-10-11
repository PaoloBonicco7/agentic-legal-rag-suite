# 10 — Portable reproduction

## Purpose
Restore the experimental source bytes instead of downloading a different edition of the website.

## Inputs
Complete source snapshot, verified corpus, optional saved evaluation runs, package source and benchmark CSVs.

## Outputs
A gzip archive with strict `BundleManifest`, member SHA-256 values and a verified restoration directory.

## Pipeline
Verify snapshot/catalog/raw identities, corpus artifact hashes and run provenance. Package source and artifacts; exclude credentials, runtime environments, symlinks and model weights. Compute code identity from the selected project and archived Python source hashes. Every included run declaring `inputs.code.source_sha256` must match that packaged source; a mismatch requires the correct experimental checkout. Reopen and verify every archive member before promotion. Restore only into a new directory after rejecting unsafe paths, duplicates and checksum mismatches.
Verification decompresses gzip once into a temporary seekable file. Restore reads
the same verified tar instance; temporary storage closes on success or failure.

## Contract
An archive restores the frozen bytes and LLM outputs. Rebuilding indexes requires the pinned model revision and explicit device/dtype. New LLM generations may differ. Paths may change after restoration; CLI paths are explicit and source hashes remain unchanged. Indexes are rebuilt rather than included by default.
The source fingerprint uses the same sorted `src/legal_rag/**/*.py` path/hash
list as run code identity. Verification and restore recompute it from verified
archive members and check both the bundle identity and declared run identities,
even if outer checksums were recomposed. Preserved results alone do not establish
that the packaged code produced them. Legacy runs without a declared source
fingerprint remain compatible with explicit unverified-code-provenance warnings;
their data and output checks remain mandatory. This is integrity checking, not
authentication against an independently trusted experimental hash.

## Quality Gates
Internal input manifests agree with packaged files; archive checksums verified; declared code fingerprints match archived source bytes; missing legacy code provenance is visible; no secret fields; no traversal or duplicate members; restored checksums match.

## Notebook Role
Load manifests or saved results only; no packaging logic.

## Acceptance Criteria
A local bundle restores the exact snapshot, source and benchmark. Altered inputs and credential-bearing artifacts are refused. A second restoration cannot overwrite existing data.
Focused tests cover a different selected project, matching/mismatching run code,
recomposed code checksums during verification and restore, and explicit legacy
provenance warnings without weakening existing artifact checks.
