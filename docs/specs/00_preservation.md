# 00 — Preservation and boundaries

## Purpose
Start a clean implementation without losing the working research checkout or its untracked legal sources.

## Inputs
Original working tree, Git history, raw HTML, benchmark CSVs and thesis documentation.

## Outputs
Isolated `feature/legal-rag-rebuild` checkout, verified private source archive and file-level checksum manifest.

## Pipeline
Archive source, raw corpus and research documentation; verify every archived file against its SHA-256; preserve the original checkout. Build a single new package in the isolated checkout. Generated indexes and runs stay outside version control. Existing evaluation CSVs remain byte-identical.

## Contract
A branch alone does not preserve ignored data. The private archive and its verified checksum manifest preserve those assets. The new source package must not serialize environment credentials. Historical outputs remain historical and cannot be presented as controlled new experiments.

## Quality Gates
Original checkout intact; verified source count; source/evaluation hashes retained; no credentials in portable exports; clear ownership of shared contracts.

## Notebook Role
None. Preservation is an explicit operation, not a notebook dependency.

## Acceptance Criteria
The original work and untracked legal HTML remain recoverable. New source, data and generated artifacts have separate locations. No benchmark gold is automatically revised.
