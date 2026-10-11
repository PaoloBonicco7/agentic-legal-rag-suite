# Primary sources and bounded claims

| Source | What motivates this implementation | What is not established |
|---|---|---|
| [BGE-M3](https://arxiv.org/abs/2402.03216) | Multilingual dense and lexical representations from one model. | Improvement on Valle d’Aosta laws; optimal chunk length. |
| [Legal Chunking](https://journals.sagepub.com/doi/10.3233/FAIA241255) | Evaluate segmentation as a legal retrieval choice. | Our 384/512-token defaults are hypotheses, not the paper’s proven local optimum. |
| [RRF](https://doi.org/10.1145/1571941.1572114) and [Qdrant Query API](https://qdrant.tech/documentation/search/hybrid-queries/) | Fuse ranks without comparing incompatible dense/sparse scores. | Hybrid has equal cost to dense; only the final candidate/context budget is equal. |
| [DMQR-RAG](https://arxiv.org/abs/2411.13154) | Treat query variation as an isolated retrieval experiment. | Our fixed original-plus-two policy is a full reproduction of its adaptive method. |
| [Lost in the Middle](https://arxiv.org/abs/2307.03172) | Test evidence placement and context composition as separate choices. | The observed position effects establish a failure mode or effect size for Luna on our corpus. |
| [BAAI BGE reranker model card](https://huggingface.co/BAAI/bge-reranker-v2-m3) | Optional multilingual query-passage scoring with the existing FlagEmbedding dependency. | Local M1 feasibility, Italian legal accuracy or superiority to the current LLM reranker. |
| [SAT-Graph RAG](https://journals.sagepub.com/doi/10.3233/FAIA251598) | Preserve structural relations and explicit temporal evidence. | Quantitative improvement here; our one-hop graph implements its full ontology/version system. |
| [ARES](https://aclanthology.org/2024.naacl-long.20/) | Separate relevance, answer correctness and faithfulness. | We reproduce its trained judges or prediction-powered inference; our judges are diagnostic LLM judgments. |
| [Legal RAG Bench](https://arxiv.org/abs/2603.01710) | Separate retrieval and generation errors in controlled comparisons. | Its Australian-domain findings transfer to Italian law; references constitute exhaustive relevance judgments. |
| [OpenAI structured outputs](https://developers.openai.com/api/docs/guides/structured-outputs) | Parse strict response contracts through Responses. | Schema compliance establishes substantive legal correctness. |

Primary pages were checked on 2026-10-10. Method defaults are declared in specifications and measured locally through ablations. The initial 100 intents are reused development material; bootstrap intervals quantify variation conditional on this benchmark, not population-wide legal accuracy. Unjudged passages are not automatically irrelevant. A reranker changes selection/order within an existing pool.

The strategy review rechecked DMQR-RAG, SAT-Graph RAG and the Qdrant Query API,
and added the two sources above on 2026-10-11. These motivate the
[planned experiments](../specs/11_retrieval_optimization.md), not implemented gains.
Native weighted RRF in Qdrant 1.17 differs from linear weighted outer fusion
between query rankings; freeze and test the selected formula explicitly.
