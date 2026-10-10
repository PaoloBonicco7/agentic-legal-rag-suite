# Local embedding execution

The common tokenizer and embedder use `BAAI/bge-m3` revision
`5617a9f61b028005a4858fdac845db406aefb181`. CPU/float32/batch 2 remains the
portable default. The execution configuration used for the current index is
explicit MPS/float16, inference batch 8 and upload groups of 64. Model identity
records dtype, device, padding policy and library versions; the index manifest
also records both batch settings.

On the available M1 with 8 GB, an earlier long MPS run failed after 3,496
documents. FlagEmbedding's internal batch adaptation subsequently surfaced an
`IndexError`. A single failed batch reproduced successfully in a fresh process,
so that observation did not establish malformed input as the cause. Short speed
tests alone were insufficient to establish stability.

An instrumented 400-document run, including 80 of the longest inputs, measured
6.29 GB of MPS driver allocation with unrounded padding. Padding to multiples
of 32 limits the number of tensor shapes; a separate 400-document run measured
2.75 GB at completion, with about 1.14 GB of live tensors and 71.25 seconds of
elapsed execution. These are local observations, not a general memory bound.
The driver and live-tensor values are distinct measurements. Calling
`empty_cache()` does not by itself establish that all allocations are released.

The adapter rejects inputs over 512 tokens instead of truncating them. Local
inference failures stop the job without changing provider or device. Verified
cache records permit rebuilding an interrupted index into a new directory;
incomplete indexes cannot serve ordinary retrieval. Old cache formats require
explicit validation against persisted vectors before migration.

The complete-corpus result and exact artifact paths are recorded in the
[delivery note](../results/00_delivery.md). Numerical embeddings are preserved
in the index/cache; byte-identical preprocessing does not imply bitwise-identical
floating-point inference across hardware or library versions.
