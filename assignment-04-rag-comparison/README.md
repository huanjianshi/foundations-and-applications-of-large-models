# Assignment 04 — Vector RAG, GraphRAG, and WikiRAG

This assignment compares three retrieval-augmented generation knowledge organization strategies on the same 15-document corpus and 20-question benchmark:

- **Vector RAG**: fixed-length overlapping chunks ranked by multilingual sentence embeddings.
- **GraphRAG**: entity-relation triples, graph traversal, and separate Local/Global retrieval scopes.
- **WikiRAG**: curated topic entries mapped back to their source documents.
- **Hybrid extension**: reciprocal-rank fusion (RRF) of character-bigram lexical retrieval and vector retrieval.

The repository contains the retrieval and evaluation code plus the reproducibility artifacts used in the report. Answer generation did not call an LLM API; answers were produced separately from the retrieved contexts. The optional extraction mode is provided only for audited triple extraction experiments.

## Environment

- Python 3.10+
- PyTorch 2.x; CPU execution is sufficient
- sentence-transformers
- networkx
- numpy

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
```

The default embedding model is `paraphrase-multilingual-MiniLM-L12-v2`. Its weights are downloaded on first use. The program does not silently fall back to a lexical approximation when the model is unavailable.

## Usage

Run the complete experiment suite:

```bash
python run_experiments.py
```

Run representative individual queries and evaluations:

```bash
python hw4_rag.py --mode graph --q Q01 --scope local
python hw4_rag.py --mode graph --q Q01 --scope global
python hw4_rag.py --mode eval --set builtin15 --k 2 --scope local
python hw4_rag.py --mrr_check 2,1,4
```

The optional `--mode extract` reads `LLM_API_BASE`, `LLM_API_KEY`, and `LLM_MODEL` from environment variables and requests JSON triples from an OpenAI-compatible endpoint. Its output must be reviewed before replacing the checked preset triples. This mode was not used for the reported retrieval benchmark.

## Reproducibility artifacts

- [Per-question rankings and evidence hits](artifacts/per_question.csv)
- [Aggregated metrics](artifacts/summary.csv)
- [Environment and graph metadata](artifacts/metadata.json)
- [Fidelity scores](artifacts/fidelity_scores.csv)
- [Representative Vector RAG console output](artifacts/vector_q01_console.txt)
- [Representative GraphRAG Local console output](artifacts/graph_q01_local_console.txt)
- [Figures](figures/)

The report document, assignment guide, template, model weights, caches, and document-generation utilities are intentionally excluded from this public source repository.
