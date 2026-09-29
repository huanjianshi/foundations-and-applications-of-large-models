# -*- coding: utf-8 -*-
"""课程作业四：Vector RAG、GraphRAG、WikiRAG 与混合检索对比。

默认使用真实 sentence-transformers 嵌入；未安装或下载失败时直接报错。
预置三元组、条目、社区摘要来自课程指南，并非本程序现场 LLM 生成。
"""
import argparse
import csv
import json
import math
import os
import platform
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np

CORPUS = {
    "doc01": "云山大学位于岭南省云州市，建于 1958 年，是省属重点大学，现有全日制本科生约 2.1 万人。校训为「格物致知」。",
    "doc02": "人工智能学院成立于 2018 年，首任及现任院长为王启明教授。学院下设机器学习系、智能科学系、认知计算系三个系。",
    "doc03": "计算机学院前身为 1985 年成立的计算机系，现任院长陈国峰教授，设计算机科学与技术、软件工程两个本科专业。",
    "doc04": "人工智能学院教师：李文瀚，教授，2019 年入职，研究方向为检索增强生成，主讲《大模型通识课》；赵婉晴，副教授，研究方向为知识图谱，主讲《知识图谱导论》；孙浩然，讲师，研究方向为强化学习。",
    "doc05": "计算机学院教师：周天宇，教授，研究方向为分布式系统；林小雨，副教授，研究方向为数据库系统。",
    "doc06": "《大模型通识课》课程代码 AI2101，3 学分、32 学时，春季学期开设，授课教师李文瀚，面向全校本科生，无先修课程要求。",
    "doc07": "《知识图谱导论》课程代码 AI3305，2 学分，秋季学期开设，授课教师赵婉晴，面向人工智能学院研究生。",
    "doc08": "选课规则：本科生每学期最多修 30 学分；GPA 低于 2.0 给予学术警告。先修要求：《知识图谱导论》需先修《数据结构》，《大模型通识课》无先修要求。",
    "doc09": "奖学金体系：国家奖学金 8000 元/年，评定比例约 2%；校长奖学金 20000 元/年，全校每年 10 人；云山一等奖学金 3000 元/年，比例约 5%。",
    "doc10": "科研平台：认知计算全国重点实验室依托人工智能学院建设；岭南超算中心由云山大学与省科技厅共建，计算机学院参与运行管理。",
    "doc11": "学生社团：AI 协会，指导教师李文瀚，每周三晚组织论文研读；机器人战队，指导教师孙浩然，每年参加全国机器人大赛。",
    "doc12": "校历：春季学期 3 月 2 日开学、7 月 5 日放暑假；秋季学期 9 月 1 日开学、次年 1 月 15 日放寒假。",
    "doc13": "图书馆藏书 380 万册，开放时间为每天 7:00-22:00；人工智能分馆位于理科楼 B 座 3 层，收藏大模型与智能体专题图书。",
    "doc14": "校园交通：地铁 3 号线「云大站」距东门 200 米；校内校巴共 5 条线路，10 分钟一班。",
    "doc15": "国际交流：学校与 12 个国家的 47 所高校签有交换协议；人工智能学院与新加坡南洋理工大学有本科联合培养项目。",
}

QUESTIONS = [
    ("Q01", "multi", "《大模型通识课》授课教师所在学院的行政负责人是谁？", ["doc04", "doc06", "doc02"]),
    ("Q02", "multi", "AI 协会的指导教师主讲课程的课程代码是什么？", ["doc11", "doc04", "doc06"]),
    ("Q03", "multi", "认知计算全国重点实验室依托的学院成立于哪一年？", ["doc10", "doc02"]),
    ("Q04", "multi", "《知识图谱导论》授课教师的研究方向是什么？", ["doc07", "doc04"]),
    ("Q05", "multi", "机器人战队的指导教师的研究方向是什么？", ["doc11", "doc04"]),
    ("Q06", "fact", "云山大学的校训是什么？", ["doc01"]),
    ("Q07", "fact", "国家奖学金的金额是多少？", ["doc09"]),
    ("Q08", "fact", "图书馆每天几点到几点开放？", ["doc13"]),
    ("Q09", "fact", "人工智能学院成立于哪一年？", ["doc02"]),
    ("Q10", "fact", "春季学期什么时候开学？", ["doc12"]),
    ("Q11", "global", "云山大学有哪些科研平台？分别依托谁建设？", ["doc10", "doc02", "doc03"]),
    ("Q12", "global", "学校的学生奖励体系包含哪些项目？", ["doc09"]),
    ("Q13", "global", "人工智能学院有哪些教师？各自做什么研究方向？", ["doc04"]),
    ("Q14", "global", "本科生选课有哪些限制和先修要求？", ["doc08"]),
    ("Q15", "global", "对想深入学习大模型的学生，学校提供哪些课程和课外活动？", ["doc06", "doc11"]),
    ("Q16", "fact", "校内校巴每隔多少分钟一班？", ["doc14"]),
    ("Q17", "fact", "云山一等奖学金每年的金额是多少？", ["doc09"]),
    ("Q18", "multi", "《大模型通识课》的主讲教师指导哪个学生社团？", ["doc06", "doc04", "doc11"]),
    ("Q19", "multi", "机器人战队指导教师所在学院的院长是谁？", ["doc11", "doc04", "doc02"]),
    ("Q20", "global", "学校提供哪些国际交流机会？人工智能学院参与了什么项目？", ["doc15"]),
]
QUESTIONS = [dict(id=i, type=t, q=q, evidence=e) for i, t, q, e in QUESTIONS]

# 原指南的 40 条预置三元组，保留来源文档号。
PRE_TRIPLES = [
    ("李文瀚", "任职于", "人工智能学院", "doc04"), ("李文瀚", "职称", "教授", "doc04"),
    ("李文瀚", "入职年份", "2019 年", "doc04"), ("李文瀚", "研究方向", "检索增强生成", "doc04"),
    ("李文瀚", "主讲", "大模型通识课", "doc06"), ("李文瀚", "指导", "AI 协会", "doc11"),
    ("赵婉晴", "任职于", "人工智能学院", "doc04"), ("赵婉晴", "研究方向", "知识图谱", "doc04"),
    ("赵婉晴", "主讲", "知识图谱导论", "doc07"), ("孙浩然", "任职于", "人工智能学院", "doc04"),
    ("孙浩然", "研究方向", "强化学习", "doc04"), ("孙浩然", "指导", "机器人战队", "doc11"),
    ("人工智能学院", "院长", "王启明", "doc02"), ("人工智能学院", "成立于", "2018 年", "doc02"),
    ("人工智能学院", "下设系", "机器学习系", "doc02"), ("人工智能学院", "下设系", "智能科学系", "doc02"),
    ("人工智能学院", "下设系", "认知计算系", "doc02"), ("认知计算全国重点实验室", "依托", "人工智能学院", "doc10"),
    ("岭南超算中心", "参与共建", "计算机学院", "doc10"), ("计算机学院", "院长", "陈国峰", "doc03"),
    ("计算机学院", "前身", "计算机系", "doc03"), ("周天宇", "任职于", "计算机学院", "doc05"),
    ("周天宇", "研究方向", "分布式系统", "doc05"), ("林小雨", "任职于", "计算机学院", "doc05"),
    ("林小雨", "研究方向", "数据库系统", "doc05"), ("大模型通识课", "课程代码", "AI2101", "doc06"),
    ("大模型通识课", "学分", "3 学分", "doc06"), ("大模型通识课", "开设学期", "春季学期", "doc06"),
    ("知识图谱导论", "课程代码", "AI3305", "doc07"), ("知识图谱导论", "先修课程", "数据结构", "doc08"),
    ("AI 协会", "指导教师", "李文瀚", "doc11"), ("机器人战队", "指导教师", "孙浩然", "doc11"),
    ("国家奖学金", "金额", "8000 元/年", "doc09"), ("校长奖学金", "金额", "20000 元/年", "doc09"),
    ("云山一等奖学金", "金额", "3000 元/年", "doc09"), ("云山大学", "校训", "格物致知", "doc01"),
    ("云山大学", "建校于", "1958 年", "doc01"), ("云山大学", "位于", "岭南省云州市", "doc01"),
    ("图书馆", "藏书量", "380 万册", "doc13"), ("春季学期", "开学日期", "3 月 2 日", "doc12"),
]

# 参考指南手工提供的摘要，与运行时贪心模块度发现的社区不是同一次产物。
COMMUNITY_SUMMARIES = [
    ("社区 C1 教学与课程：人工智能学院设三个系，开设《大模型通识课》（AI2101，李文瀚主讲）与《知识图谱导论》（AI3305，赵婉晴主讲），AI3305 需先修《数据结构》。", ["doc02", "doc04", "doc06", "doc07", "doc08"]),
    ("社区 C2 科研与国际：认知计算全国重点实验室依托人工智能学院，岭南超算中心由计算机学院参与共建；学院与新加坡南洋理工大学有联合培养项目。", ["doc02", "doc03", "doc10", "doc15"]),
    ("社区 C3 学生生活：奖学金分国家（8000 元）、校长（20000 元）、云山一等（3000 元）三级；社团有 AI 协会与机器人战队，图书馆藏书 380 万册、7:00-22:00 开放。", ["doc09", "doc11", "doc13"]),
]

PRE_ENTRIES = [
    ("云山大学", "云山大学位于岭南省云州市，建于 1958 年，省属重点大学，本科生约 2.1 万人，校训「格物致知」。地铁 3 号线「云大站」距东门 200 米，校巴 5 条线路。校历：春季学期 3 月 2 日开学。", ["doc01", "doc12", "doc14"]),
    ("人工智能学院", "人工智能学院成立于 2018 年，院长王启明教授，下设机器学习系、智能科学系、认知计算系。认知计算全国重点实验室依托本院建设。与新加坡南洋理工大学有本科联合培养项目。", ["doc02", "doc10", "doc15"]),
    ("计算机学院", "计算机学院前身是 1985 年成立的计算机系，院长陈国峰教授，设计算机科学与技术、软件工程两个专业。参与岭南超算中心运行管理。", ["doc03", "doc10"]),
    ("人工智能学院教师", "李文瀚：教授，2019 年入职，研究检索增强生成，主讲《大模型通识课》，指导 AI 协会。赵婉晴：副教授，研究知识图谱，主讲《知识图谱导论》。孙浩然：讲师，研究强化学习，指导机器人战队。", ["doc04", "doc11"]),
    ("计算机学院教师", "周天宇：教授，研究分布式系统。林小雨：副教授，研究数据库系统。", ["doc05"]),
    ("大模型通识课", "《大模型通识课》（AI2101）：3 学分 32 学时，春季学期开设，李文瀚主讲，面向全校本科生，无先修要求。", ["doc06"]),
    ("知识图谱导论", "《知识图谱导论》（AI3305）：2 学分，秋季学期，赵婉晴主讲，面向人工智能学院研究生，需先修《数据结构》。", ["doc07", "doc08"]),
    ("奖学金体系", "国家奖学金 8000 元/年（约 2%）；校长奖学金 20000 元/年（全校 10 人）；云山一等奖学金 3000 元/年（约 5%）。", ["doc09"]),
    ("学生社团与校园生活", "AI 协会：指导教师李文瀚，每周三晚论文研读。机器人战队：指导教师孙浩然，参加全国机器人大赛。图书馆藏书 380 万册，每天 7:00-22:00 开放，人工智能分馆在理科楼 B 座 3 层。", ["doc11", "doc13"]),
]


def chunk_docs(size=120, overlap=20):
    if size <= overlap or overlap < 0:
        raise ValueError("分块要求 size > overlap >= 0")
    out = []
    for doc, txt in CORPUS.items():
        for i in range(0, len(txt), size - overlap):
            out.append((doc, txt[i:i + size]))
            if i + size >= len(txt):
                break
    return out


class Embedder:
    def __init__(self, model_name):
        from sentence_transformers import SentenceTransformer
        self.name = model_name
        self.model = SentenceTransformer(model_name, device="cpu")

    def encode(self, texts):
        return np.asarray(self.model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False))


def dedup_scored(items):
    best = {}
    for doc, score, detail in items:
        if doc not in best or score > best[doc][0]:
            best[doc] = (float(score), detail)
    return [(d, *best[d]) for d in sorted(best, key=lambda d: (-best[d][0], d))]


class VectorRAG:
    name = "Vector RAG"
    def __init__(self, emb):
        self.emb = emb
        self.chunks = chunk_docs()
        self.matrix = emb.encode([x[1] for x in self.chunks])

    def rank(self, query):
        scores = self.matrix @ self.emb.encode([query])[0]
        return dedup_scored([(d, scores[i], txt) for i, (d, txt) in enumerate(self.chunks)])


class GraphRAG:
    name = "GraphRAG"
    def __init__(self, emb):
        self.emb = emb
        self.graph = nx.MultiDiGraph()
        for h, rel, t, src in PRE_TRIPLES:
            self.graph.add_edge(h, t, rel=rel, src=src)
        self.communities = list(nx.community.greedy_modularity_communities(nx.Graph(self.graph)))
        self.summary_matrix = emb.encode([s for s, _ in COMMUNITY_SUMMARIES])

    def entities(self, query):
        norm = query.replace("《", "").replace("》", "")
        matches = [n for n in self.graph.nodes if n in norm]
        return sorted(matches, key=lambda n: (-len(n), n))[:2]

    def rank_local(self, query):
        frontier = self.entities(query)
        if not frontier:
            return self.rank_global(query)
        seen_edges, triples = set(), []
        for _ in range(3):
            next_nodes = []
            for entity in frontier:
                edges = list(self.graph.out_edges(entity, keys=True, data=True)) + list(self.graph.in_edges(entity, keys=True, data=True))
                for h, t, key, data in edges:
                    edge_id = (h, t, key)
                    if edge_id not in seen_edges:
                        seen_edges.add(edge_id)
                        triples.append((data["src"], f"{h}—{data['rel']}→{t}"))
                    other = t if h == entity else h
                    if other not in next_nodes:
                        next_nodes.append(other)
            frontier = sorted(next_nodes, key=lambda n: (-self.graph.out_degree(n), n))[:6]
        if not triples:
            return self.rank_global(query)
        scores = self.emb.encode([t for _, t in triples]) @ self.emb.encode([query])[0]
        return dedup_scored([(d, scores[i], txt) for i, (d, txt) in enumerate(triples)])

    def rank_global(self, query):
        scores = self.summary_matrix @ self.emb.encode([query])[0]
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:2]
        out, seen = [], set()
        for i in order:
            for d in COMMUNITY_SUMMARIES[i][1]:
                if d not in seen:
                    out.append((d, float(scores[i]), COMMUNITY_SUMMARIES[i][0]))
                    seen.add(d)
        return out

    def rank(self, query, scope="local"):
        return self.rank_local(query) if scope == "local" else self.rank_global(query)


class WikiRAG:
    name = "WikiRAG"
    def __init__(self, emb):
        self.emb = emb
        self.matrix = emb.encode([title + "。" + txt for title, txt, _ in PRE_ENTRIES])

    def rank(self, query):
        scores = self.matrix @ self.emb.encode([query])[0]
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:3]
        out, seen = [], set()
        for i in order:
            title, _, sources = PRE_ENTRIES[i]
            for d in sources:
                if d not in seen:
                    out.append((d, float(scores[i]), title))
                    seen.add(d)
        return out


def char_bigrams(text):
    compact = re.sub(r"\s+", "", text.lower())
    return {compact[i:i + 2] for i in range(max(0, len(compact) - 1))}


class HybridRAG:
    name = "Hybrid RAG"
    def __init__(self, vector, rrf_c=60):
        self.vector = vector
        self.rrf_c = rrf_c
        self.lexical = [(d, char_bigrams(t)) for d, t in vector.chunks]

    def rank(self, query):
        vec = self.vector.rank(query)
        qg = char_bigrams(query)
        lex_chunk = sorted(((d, len(qg & grams) / max(1, len(qg | grams))) for d, grams in self.lexical), key=lambda x: -x[1])
        lex, seen = [], set()
        for d, _ in lex_chunk:
            if d not in seen:
                seen.add(d)
                lex.append(d)
        vr = {d: i + 1 for i, (d, _, _) in enumerate(vec)}
        lr = {d: i + 1 for i, d in enumerate(lex)}
        return [(d, 1 / (self.rrf_c + vr[d]) + 1 / (self.rrf_c + lr[d]), f"vector={vr[d]}, lexical={lr[d]}") for d in sorted(vr, key=lambda d: (-(1 / (self.rrf_c + vr[d]) + 1 / (self.rrf_c + lr[d])), d))]


def ranked(system, query, scope="local"):
    return system.rank(query, scope) if isinstance(system, GraphRAG) else system.rank(query)


def evaluate(systems, questions, k, scope="local"):
    rows = []
    for name, system in systems.items():
        for q in questions:
            hits = ranked(system, q["q"], scope)[:k]
            retrieved = [d for d, _, _ in hits]
            evidence = set(q["evidence"])
            rank = next((i + 1 for i, d in enumerate(retrieved) if d in evidence), 0)
            rows.append(dict(system=name, scope=scope if name == "GraphRAG" else "na", k=k,
                             question_set="builtin15" if len(questions) == 15 else "all20",
                             id=q["id"], type=q["type"], question=q["q"], evidence=";".join(q["evidence"]),
                             retrieved=";".join(retrieved), first_hit_rank=rank,
                             recall=len(evidence.intersection(retrieved)) / len(evidence),
                             reciprocal_rank=1 / rank if rank else 0.0,
                             complete_hit=int(evidence.issubset(retrieved)),
                             hits=json.dumps([dict(doc=d, score=round(s, 6), detail=t, evidence=d in evidence) for d, s, t in hits], ensure_ascii=False)))
    return rows


def summary(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["system"], row["scope"], row["question_set"], row["k"], row["type"])].append(row)
    return [dict(system=key[0], scope=key[1], question_set=key[2], k=key[3], type=key[4], n=len(group),
                 recall=round(sum(r["recall"] for r in group) / len(group), 6),
                 mrr=round(sum(r["reciprocal_rank"] for r in group) / len(group), 6),
                 complete=sum(r["complete_hit"] for r in group)) for key, group in sorted(groups.items())]


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


EXTRACT_PROMPT = """从下面的大学介绍文档中抽取5至10条[头实体, 关系, 尾实体]三元组。
只输出JSON数组，不要Markdown代码块或说明文字。关系使用简短动词短语。
文档：\n{doc}"""


def parse_triples(raw, doc_id):
    """严格解析现场抽取结果；拒绝不完整或非字符串的三元组。"""
    value = json.loads(raw.strip())
    if not isinstance(value, list):
        raise ValueError("顶层必须是JSON数组")
    triples = []
    for item in value:
        if not isinstance(item, list) or len(item) != 3 or any(not isinstance(x, str) or not x.strip() for x in item):
            raise ValueError(f"非法三元组：{item!r}")
        triples.append((item[0].strip(), item[1].strip(), item[2].strip(), doc_id))
    return triples


def extract_triples(output=None):
    base, key, model = (os.getenv(x) for x in ("LLM_API_BASE", "LLM_API_KEY", "LLM_MODEL"))
    if not all((base, key, model)):
        raise SystemExit("现场抽取须配置 LLM_API_BASE、LLM_API_KEY、LLM_MODEL；正式检索评测不需要")
    from openai import OpenAI
    client = OpenAI(base_url=base, api_key=key)
    results, failures = [], []
    for doc_id, document in CORPUS.items():
        response = client.chat.completions.create(model=model, temperature=0,
            messages=[{"role": "user", "content": EXTRACT_PROMPT.format(doc=document)}])
        raw = response.choices[0].message.content or ""
        try:
            triples = parse_triples(raw, doc_id)
            results.extend(triples)
            print(f"{doc_id}: {len(triples)}条")
        except (json.JSONDecodeError, ValueError) as exc:
            failures.append(dict(doc=doc_id, error=str(exc), raw=raw))
            print(f"{doc_id}: 解析失败：{exc}", file=sys.stderr)
    print(f"合计{len(results)}条；解析失败{len(failures)}/{len(CORPUS)}篇。须人工核对实体与来源后方可替换PRE_TRIPLES。")
    if output:
        output.mkdir(parents=True, exist_ok=True)
        (output / "extracted_triples.json").write_text(json.dumps(dict(triples=results, failures=failures), ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["eval", "vector", "graph", "wiki", "hybrid", "extract"], default="eval")
    p.add_argument("--q", default="Q01")
    p.add_argument("--scope", choices=["local", "global"], default="local")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--set", choices=["builtin15", "all20"], default="all20")
    p.add_argument("--emb", default="paraphrase-multilingual-MiniLM-L12-v2")
    p.add_argument("--out", type=Path)
    p.add_argument("--mrr_check")
    args = p.parse_args()
    if args.mrr_check:
        ranks = [int(x) for x in args.mrr_check.split(",")]
        if not ranks or any(x < 0 for x in ranks):
            p.error("排名须为非负整数；0 表示未命中")
        val = sum(1 / r if r else 0 for r in ranks) / len(ranks)
        print("MRR = (" + " + ".join("0" if r == 0 else f"1/{r}" for r in ranks) + f") / {len(ranks)} = {val:.6f}")
        return
    if args.k < 1:
        p.error("--k 至少为 1")
    if args.mode == "extract":
        extract_triples(args.out)
        return
    started = time.time()
    emb = Embedder(args.emb)
    vector = VectorRAG(emb)
    systems = {"Vector RAG": vector, "GraphRAG": GraphRAG(emb), "WikiRAG": WikiRAG(emb), "Hybrid RAG": HybridRAG(vector)}
    qs = QUESTIONS[:15] if args.set == "builtin15" else QUESTIONS
    if args.mode == "eval":
        rows = evaluate(systems, qs, args.k, args.scope)
        sums = summary(rows)
        for row in sums:
            print(f"{row['system']:12} {row['scope']:6} {row['type']:6} R@{args.k}={row['recall']:.3f} MRR={row['mrr']:.3f} full={row['complete']}/{row['n']}")
        if args.out:
            args.out.mkdir(parents=True, exist_ok=True)
            write_csv(args.out / "per_question.csv", rows)
            write_csv(args.out / "summary.csv", sums)
            (args.out / "metadata.json").write_text(json.dumps(dict(model=args.emb, python=sys.version, platform=platform.platform(), networkx=nx.__version__, numpy=np.__version__, seconds=round(time.time()-started, 2), question_set=args.set, k=args.k, scope=args.scope, triples=len(PRE_TRIPLES), nodes=systems["GraphRAG"].graph.number_of_nodes(), detected_communities=len(systems["GraphRAG"].communities), preset_summaries=len(COMMUNITY_SUMMARIES)), ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        q = next((q for q in QUESTIONS if q["id"].lower() == args.q.lower()), None)
        if q is None:
            p.error("未知题号")
        name = {"vector": "Vector RAG", "graph": "GraphRAG", "wiki": "WikiRAG", "hybrid": "Hybrid RAG"}[args.mode]
        print(f"{q['id']} {q['q']} evidence={','.join(q['evidence'])}")
        for i, (d, s, detail) in enumerate(ranked(systems[name], q["q"], args.scope)[:args.k], 1):
            print(f"{i}. {d} score={s:.6f} {'[evidence]' if d in q['evidence'] else ''} {detail}")
            print(f"   {CORPUS[d]}")


if __name__ == "__main__":
    main()
