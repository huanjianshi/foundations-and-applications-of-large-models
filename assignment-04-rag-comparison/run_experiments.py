"""一次加载嵌入模型，完成作业四全部必做检索实验。"""
import json
import platform
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np
import sentence_transformers
import torch

from hw4_rag import (COMMUNITY_SUMMARIES, CORPUS, PRE_ENTRIES, PRE_TRIPLES,
                     QUESTIONS, Embedder, GraphRAG, HybridRAG, VectorRAG,
                     WikiRAG, evaluate, summary, write_csv)


def main():
    start = time.time()
    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    model = "paraphrase-multilingual-MiniLM-L12-v2"
    emb = Embedder(model)
    vec = VectorRAG(emb)
    graph = GraphRAG(emb)
    systems = {"Vector RAG": vec, "GraphRAG": graph, "WikiRAG": WikiRAG(emb),
               "Hybrid RAG": HybridRAG(vec)}
    rows = []
    rows += evaluate(systems, QUESTIONS, 5, "local")
    rows += evaluate({"GraphRAG": graph}, QUESTIONS, 5, "global")
    for k in (1, 2, 3, 5):
        rows += evaluate(systems, QUESTIONS[:15], k, "local")
        rows += evaluate({"GraphRAG": graph}, QUESTIONS[:15], k, "global")
    write_csv(out / "per_question.csv", rows)
    write_csv(out / "summary.csv", summary(rows))
    meta = dict(model=model, python=sys.version, os=platform.platform(), cpu=platform.processor(),
                sentence_transformers=sentence_transformers.__version__, torch=torch.__version__,
                networkx=nx.__version__, numpy=np.__version__, documents=len(CORPUS),
                questions_builtin=15, questions_added=5, triples=len(PRE_TRIPLES),
                graph_nodes=graph.graph.number_of_nodes(), detected_communities=len(graph.communities),
                preset_summaries=len(COMMUNITY_SUMMARIES), preset_entries=len(PRE_ENTRIES),
                chunk_size=120, chunk_overlap=20, rrf_c=60,
                runtime_seconds=round(time.time() - start, 2))
    (out / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    for s in summary(rows):
        if s["question_set"] == "all20" and s["k"] == 5:
            print(s)


if __name__ == "__main__":
    main()
