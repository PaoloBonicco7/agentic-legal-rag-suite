# 10 — Portable reproduction

## Purpose
Restore the experimental source bytes instead of downloading a different edition of the website.

## Inputs
Complete source snapshot, verified corpus, optional saved evaluation runs, package source and benchmark CSVs.

## Outputs
A gzip archive with strict `BundleManifest`, member SHA-256 values and a verified restoration directory.

## Pipeline
Verify snapshot/catalog/raw identities, corpus artifact hashes and run provenance. Package source and artifacts; exclude credentials, runtime environments, symlinks and model weights. Reopen and verify every archive member before promotion. Restore only into a new directory after rejecting unsafe paths, duplicates and checksum mismatches.
Verification decompresses gzip once into a temporary seekable file. Restore reads
the same verified tar instance; temporary storage closes on success or failure.

## Contract
An archive restores the frozen bytes and LLM outputs. Rebuilding indexes requires the pinned model revision and explicit device/dtype. New LLM generations may differ. Paths may change after restoration; CLI paths are explicit and source hashes remain unchanged. Indexes are rebuilt rather than included by default.

## Quality Gates
Internal input manifests agree with packaged files; archive checksums verified; no secret fields; no traversal or duplicate members; restored checksums match.

## Notebook Role
Load manifests or saved results only; no packaging logic.

## Acceptance Criteria
A local bundle restores the exact snapshot, source and benchmark. Altered inputs and credential-bearing artifacts are refused. A second restoration cannot overwrite existing data.
