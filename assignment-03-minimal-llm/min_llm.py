# -*- coding: utf-8 -*-
"""作业三：在 CPU 上训练字符级最小 GPT，并复现实验指南中的全部实验。"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import math
import os
import platform
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F


POEMS = [
    "床前明月光，疑是地上霜。举头望明月，低头思故乡。",
    "春眠不觉晓，处处闻啼鸟。夜来风雨声，花落知多少。",
    "白日依山尽，黄河入海流。欲穷千里目，更上一层楼。",
    "锄禾日当午，汗滴禾下土。谁知盘中餐，粒粒皆辛苦。",
    "离离原上草，一岁一枯荣。野火烧不尽，春风吹又生。",
    "远芳侵古道，晴翠接荒城。又送王孙去，萋萋满别情。",
    "千山鸟飞绝，万径人踪灭。孤舟蓑笠翁，独钓寒江雪。",
    "鹅，鹅，鹅，曲项向天歌。白毛浮绿水，红掌拨清波。",
    "两个黄鹂鸣翠柳，一行白鹭上青天。窗含西岭千秋雪，门泊东吴万里船。",
    "朝辞白帝彩云间，千里江陵一日还。两岸猿声啼不住，轻舟已过万重山。",
    "故人西辞黄鹤楼，烟花三月下扬州。孤帆远影碧空尽，唯见长江天际流。",
    "李白乘舟将欲行，忽闻岸上踏歌声。桃花潭水深千尺，不及汪伦送我情。",
    "日照香炉生紫烟，遥看瀑布挂前川。飞流直下三千尺，疑是银河落九天。",
    "天门中断楚江开，碧水东流至此回。两岸青山相对出，孤帆一片日边来。",
    "月落乌啼霜满天，江枫渔火对愁眠。姑苏城外寒山寺，夜半钟声到客船。",
    "清明时节雨纷纷，路上行人欲断魂。借问酒家何处有，牧童遥指杏花村。",
    "独在异乡为异客，每逢佳节倍思亲。遥知兄弟登高处，遍插茱萸少一人。",
    "烟笼寒水月笼沙，夜泊秦淮近酒家。商女不知亡国恨，隔江犹唱后庭花。",
    "折戟沉沙铁未销，自将磨洗认前朝。东风不与周郎便，铜雀春深锁二乔。",
    "巴山楚水凄凉地，二十三年弃置身。怀旧空吟闻笛赋，到乡翻似烂柯人。",
    "沉舟侧畔千帆过，病树前头万木春。今日听君歌一曲，暂凭杯酒长精神。",
    "朱雀桥边野草花，乌衣巷口夕阳斜。旧时王谢堂前燕，飞入寻常百姓家。",
    "湖光秋月两相和，潭面无风镜未磨。遥望洞庭山水翠，白银盘里一青螺。",
    "杨柳青青江水平，闻郎江上踏歌声。东边日出西边雨，道是无晴却有晴。",
    "自古逢秋悲寂寥，我言秋日胜春朝。晴空一鹤排云上，便引诗情到碧霄。",
    "前不见古人，后不见来者。念天地之悠悠，独怆然而涕下。",
    "葡萄美酒夜光杯，欲饮琵琶马上催。醉卧沙场君莫笑，古来征战几人回。",
    "黄河远上白云间，一片孤城万仞山。羌笛何须怨杨柳，春风不度玉门关。",
    "秦时明月汉时关，万里长征人未还。但使龙城飞将在，不教胡马度阴山。",
    "月黑雁飞高，单于夜遁逃。欲将轻骑逐，大雪满弓刀。",
    "好雨知时节，当春乃发生。随风潜入夜，润物细无声。",
    "晓看红湿处，花重锦官城。野径云俱黑，江船火独明。",
    "国破山河在，城春草木深。感时花溅泪，恨别鸟惊心。",
    "烽火连三月，家书抵万金。白头搔更短，浑欲不胜簪。",
    "细草微风岸，危樯独夜舟。星垂平野阔，月涌大江流。",
    "风急天高猿啸哀，渚清沙白鸟飞回。无边落木萧萧下，不尽长江滚滚来。",
    "花近高楼伤客心，万方多难此登临。锦江春色来天地，玉垒浮云变古今。",
    "独怜幽草涧边生，上有黄鹂深树鸣。春潮带雨晚来急，野渡无人舟自横。",
    "天街小雨润如酥，草色遥看近却无。最是一年春好处，绝胜烟柳满皇都。",
    "昔人已乘黄鹤去，此地空余黄鹤楼。黄鹤一去不复返，白云千载空悠悠。",
    "晴川历历汉阳树，芳草萋萋鹦鹉洲。日暮乡关何处是，烟波江上使人愁。",
    "客舍青青柳色新，渭城朝雨浥轻尘。劝君更尽一杯酒，西出阳关无故人。",
    "寒雨连江夜入吴，平明送客楚山孤。洛阳亲友如相问，一片冰心在玉壶。",
    "山光忽西落，池月渐东上。散发乘夕凉，开轩卧闲敞。",
    "荷笠带斜阳，青山独归远。苍苍竹林寺，杳杳钟声晚。",
    "空山不见人，但闻人语响。返景入深林，复照青苔上。",
    "人闲桂花落，夜静春山空。月出惊山鸟，时鸣春涧中。",
    "红豆生南国，春来发几枝。愿君多采撷，此物最相思。",
    "独坐幽篁里，弹琴复长啸。深林人不知，明月来相照。",
    "君自故乡来，应知故乡事。来日绮窗前，寒梅著花未。",
    "山中相送罢，日暮掩柴扉。春草明年绿，王孙归不归。",
    "花间一壶酒，独酌无相亲。举杯邀明月，对影成三人。",
    "小时不识月，呼作白玉盘。又疑瑶台镜，飞在青云端。",
    "长安一片月，万户捣衣声。秋风吹不尽，总是玉关情。",
    "弃我去者，昨日之日不可留。乱我心者，今日之日多烦忧。",
    "抽刀断水水更流，举杯消愁愁更愁。人生在世不称意，明朝散发弄扁舟。",
    "千里黄云白日曛，北风吹雁雪纷纷。莫愁前路无知己，天下谁人不识君。",
    "慈母手中线，游子身上衣。临行密密缝，意恐迟迟归。谁言寸草心，报得三春晖。",
    "山重水复疑无路，柳暗花明又一村。莫笑农家腊酒浑，丰年留客足鸡豚。",
    "纸上得来终觉浅，绝知此事要躬行。古人学问无遗力，少壮功夫老始成。",
    "死去元知万事空，但悲不见九州同。王师北定中原日，家祭无忘告乃翁。",
    "小荷才露尖尖角，早有蜻蜓立上头。泉眼无声惜细流，树阴照水爱晴柔。",
    "接天莲叶无穷碧，映日荷花别样红。毕竟西湖六月中，风光不与四时同。",
    "胜日寻芳泗水滨，无边光景一时新。等闲识得东风面，万紫千红总是春。",
    "半亩方塘一鉴开，天光云影共徘徊。问渠那得清如许，为有源头活水来。",
    "郁孤台下清江水，中间多少行人泪。西北望长安，可怜无数山。",
    "人生自古谁无死，留取丹心照汗青。辛苦遭逢起一经，干戈寥落四周星。",
    "咬定青山不放松，立根原在破岩中。千磨万击还坚劲，任尔东西南北风。",
    "千锤万凿出深山，烈火焚烧若等闲。粉骨碎身浑不怕，要留清白在人间。",
    "浩荡离愁白日斜，吟鞭东指即天涯。落红不是无情物，化作春泥更护花。",
    "九州生气恃风雷，万马齐喑究可哀。我劝天公重抖擞，不拘一格降人才。",
    "力微任重久神疲，再竭衰庸定不支。苟利国家生死以，岂因祸福避趋之。",
]


def build_corpus(extra_file: str | None = None) -> str:
    text = "\n".join(POEMS)
    if extra_file and Path(extra_file).exists():
        text += "\n" + Path(extra_file).read_text(encoding="utf-8")
    return text


class CharTokenizer:
    def __init__(self, text: str):
        self.chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(self.chars)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[ch] for ch in text if ch in self.stoi]

    def decode(self, ids: Iterable[int]) -> str:
        return "".join(self.itos[int(i)] for i in ids)


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int):
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError("n_embd 必须能被 n_head 整除")
        self.n_head = n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, steps, channels = x.shape
        q, k, v = self.qkv(x).split(channels, dim=2)
        q = q.view(batch, steps, self.n_head, -1).transpose(1, 2)
        k = k.view(batch, steps, self.n_head, -1).transpose(1, 2)
        v = v.view(batch, steps, self.n_head, -1).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        att = att.masked_fill(self.mask[:, :, :steps, :steps] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(batch, steps, channels)
        return self.proj(y)


class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class MiniGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_embd: int = 128,
        n_head: int = 4,
        n_layer: int = 2,
        block_size: int = 128,
        use_pos: bool = True,
    ):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd) if use_pos else None
        self.blocks = nn.ModuleList(
            [Block(n_embd, n_head, block_size) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, steps = idx.shape
        x = self.tok_emb(idx)
        if self.pos_emb is not None:
            x = x + self.pos_emb(torch.arange(steps, device=idx.device))
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 200,
        temperature: float = 1.0,
        top_k: int | None = None,
        no_repeat_ngram: int = 0,
    ) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.block_size :])
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            if no_repeat_ngram >= 2 and idx.size(0) == 1 and idx.size(1) >= no_repeat_ngram - 1:
                seq = idx[0].tolist()
                prefix = tuple(seq[-(no_repeat_ngram - 1) :])
                forbidden = {
                    seq[i + no_repeat_ngram - 1]
                    for i in range(len(seq) - no_repeat_ngram + 1)
                    if tuple(seq[i : i + no_repeat_ngram - 1]) == prefix
                }
                if forbidden:
                    logits[:, list(forbidden)] = float("-inf")
            if top_k is not None and top_k < logits.size(-1):
                threshold = torch.topk(logits, max(1, top_k))[0][:, -1:]
                logits = logits.masked_fill(logits < threshold, float("-inf"))
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx


@dataclass(frozen=True)
class TrainConfig:
    run_id: str
    dimension: str
    iters: int
    batch_size: int = 32
    block_size: int = 128
    n_embd: int = 128
    n_head: int = 4
    n_layer: int = 2
    lr: float = 1e-3
    seed: int = 42
    use_pos: bool = True

    @property
    def curve_name(self) -> str:
        if not self.use_pos:
            return "loss_curve_no_pos.png"
        if self.block_size != 128:
            return f"loss_curve_T{self.block_size}.png"
        return f"loss_curve_L{self.n_layer}_E{self.n_embd}_lr{self.lr:g}.png"


def configure_logging(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("min_llm")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    for handler in (
        logging.FileHandler(log_path, mode="w", encoding="utf-8"),
        logging.StreamHandler(),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def get_batch(
    data: torch.Tensor, block_size: int, batch_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    starts = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in starts])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in starts])
    return x, y


def plot_loss(losses: list[float], cfg: TrainConfig, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, len(losses) + 1), losses, linewidth=0.9, color="#1f5a94")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title(
        f"Training Loss ({cfg.run_id}: L={cfg.n_layer}, E={cfg.n_embd}, "
        f"T={cfg.block_size}, lr={cfg.lr:g}, pos={'on' if cfg.use_pos else 'off'})"
    )
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def generate_samples(
    model: MiniGPT,
    tok: CharTokenizer,
    prompt: str,
    seeds: Iterable[int],
    temperature: float,
    top_k: int | None,
    max_new_tokens: int,
    no_repeat_ngram: int = 0,
) -> list[str]:
    outputs = []
    for seed in seeds:
        set_seed(seed)
        start = torch.tensor([tok.encode(prompt)], dtype=torch.long)
        result = model.generate(
            start,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            no_repeat_ngram=no_repeat_ngram,
        )
        outputs.append(tok.decode(result[0].tolist()))
    return outputs


def save_checkpoint(path: Path, model: MiniGPT, tok: CharTokenizer, cfg: TrainConfig) -> None:
    torch.save(
        {"model_state": model.state_dict(), "chars": tok.chars, "config": asdict(cfg)},
        path,
    )


def train_one(
    cfg: TrainConfig,
    text: str,
    output_dir: Path,
    logger: logging.Logger,
    save_model: bool = False,
) -> tuple[dict[str, object], list[float], MiniGPT, CharTokenizer]:
    set_seed(cfg.seed)
    tok = CharTokenizer(text)
    data = torch.tensor(tok.encode(text), dtype=torch.long)
    model = MiniGPT(
        tok.vocab_size,
        cfg.n_embd,
        cfg.n_head,
        cfg.n_layer,
        cfg.block_size,
        cfg.use_pos,
    )
    params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    losses: list[float] = []
    started = time.perf_counter()
    status = "completed"
    logger.info(
        "%s 开始 | params=%s V=%s L=%s H=%s E=%s T=%s lr=%s pos=%s iters=%s",
        cfg.run_id,
        f"{params:,}",
        tok.vocab_size,
        cfg.n_layer,
        cfg.n_head,
        cfg.n_embd,
        cfg.block_size,
        cfg.lr,
        cfg.use_pos,
        cfg.iters,
    )
    for step in range(1, cfg.iters + 1):
        xb, yb = get_batch(data, cfg.block_size, cfg.batch_size)
        _, loss = model(xb, yb)
        assert loss is not None
        value = float(loss.item())
        losses.append(value)
        if not math.isfinite(value):
            status = "diverged"
            logger.info("%s 在 step=%s 出现非有限 loss=%s，保留证据并停止", cfg.run_id, step, value)
            break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == cfg.iters:
            elapsed = time.perf_counter() - started
            speed = step / elapsed
            eta = (cfg.iters - step) / speed if speed else 0
            logger.info(
                "%s step %4d/%d | loss %.4f | %.2f it/s | ETA %.0fs",
                cfg.run_id,
                step,
                cfg.iters,
                value,
                speed,
                eta,
            )
    elapsed = time.perf_counter() - started
    plot_loss(losses, cfg, output_dir / cfg.curve_name)
    if save_model:
        save_checkpoint(output_dir / "model_base.pt", model, tok, cfg)
    result = {
        "run_id": cfg.run_id,
        "dimension": cfg.dimension,
        "iters_planned": cfg.iters,
        "iters_completed": len(losses),
        "batch_size": cfg.batch_size,
        "block_size": cfg.block_size,
        "n_embd": cfg.n_embd,
        "n_head": cfg.n_head,
        "n_layer": cfg.n_layer,
        "learning_rate": cfg.lr,
        "use_pos": cfg.use_pos,
        "seed": cfg.seed,
        "vocab_size": tok.vocab_size,
        "corpus_chars": len(text),
        "parameters": params,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "train_seconds": elapsed,
        "iterations_per_second": len(losses) / elapsed,
        "status": status,
        "curve_file": cfg.curve_name,
    }
    logger.info(
        "%s 完成 | status=%s initial=%.4f final=%s time=%.1fs curve=%s",
        cfg.run_id,
        status,
        losses[0],
        f"{losses[-1]:.4f}",
        elapsed,
        cfg.curve_name,
    )
    return result, losses, model, tok


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def repetition_metrics(text: str) -> tuple[float, float]:
    trigrams = [text[i : i + 3] for i in range(max(0, len(text) - 2))]
    if not trigrams:
        return 0.0, 0.0
    unique_ratio = len(set(trigrams)) / len(trigrams)
    repeated_ratio = 1.0 - unique_ratio
    return unique_ratio, repeated_ratio


def run_sampling_suite(
    model: MiniGPT, tok: CharTokenizer, output_dir: Path, max_new_tokens: int
) -> list[dict[str, object]]:
    configs = [
        ("sample_1", 1.0, 20, 0),
        ("sample_2", 0.5, 20, 0),
        ("sample_3", 1.5, 20, 0),
        ("sample_4", 1.0, 5, 0),
        ("sample_5", 1.0, tok.vocab_size, 0),
        ("bonus_base", 1.0, 20, 0),
        ("bonus_no_repeat_3gram", 1.0, 20, 3),
    ]
    rows: list[dict[str, object]] = []
    text_blocks: list[str] = []
    for group, temperature, top_k, no_repeat in configs:
        samples = generate_samples(
            model,
            tok,
            "春",
            (1101, 1102, 1103),
            temperature,
            top_k,
            max_new_tokens,
            no_repeat,
        )
        text_blocks.append(
            f"## {group} | temperature={temperature} | top_k={top_k} | "
            f"no_repeat_ngram={no_repeat}\n"
            + "\n\n".join(f"[{i}] {sample}" for i, sample in enumerate(samples, 1))
        )
        for index, sample in enumerate(samples, 1):
            unique_ratio, repeated_ratio = repetition_metrics(sample)
            rows.append(
                {
                    "group": group,
                    "temperature": temperature,
                    "top_k": top_k,
                    "no_repeat_ngram": no_repeat,
                    "sample_index": index,
                    "seed": 1100 + index,
                    "prompt": "春",
                    "text": sample,
                    "unique_3gram_ratio": unique_ratio,
                    "repeated_3gram_ratio": repeated_ratio,
                }
            )
    write_rows(output_dir / "sampling_results.csv", rows)
    (output_dir / "generated_sampling.txt").write_text(
        "\n\n".join(text_blocks) + "\n", encoding="utf-8"
    )
    return rows


def baseline_samples(
    model: MiniGPT, tok: CharTokenizer, output_dir: Path, max_new_tokens: int
) -> None:
    blocks = []
    for prompt in ("春", "月"):
        samples = generate_samples(
            model, tok, prompt, (1001, 1002), 1.0, 20, max_new_tokens
        )
        blocks.append(
            f"提示词「{prompt}」\n" + "\n\n".join(
                f"样例 {index}\n{sample}" for index, sample in enumerate(samples, 1)
            )
        )
    (output_dir / "generated_base.txt").write_text(
        "\n\n".join(blocks) + "\n", encoding="utf-8"
    )


def suite_configs() -> list[TrainConfig]:
    return [
        TrainConfig("baseline", "基线", 2000),
        TrainConfig("exp1_lr_1e-2", "学习率", 1000, lr=1e-2),
        TrainConfig("exp1_lr_1e-4", "学习率", 1000, lr=1e-4),
        TrainConfig("exp2_layer_1", "层数", 1000, n_layer=1),
        TrainConfig("exp2_layer_4", "层数", 1000, n_layer=4),
        TrainConfig("exp3_embd_64", "嵌入维度", 1000, n_embd=64, n_head=2),
        TrainConfig("exp3_embd_256", "嵌入维度", 1000, n_embd=256, n_head=8),
        TrainConfig("exp4_context_32", "上下文长度", 1000, block_size=32),
        TrainConfig("exp5_no_pos", "位置编码", 1000, use_pos=False),
    ]


def run_suite(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(output_dir / "run.log")
    text = build_corpus(args.extra_corpus)
    logger.info(
        "环境 | Python=%s | PyTorch=%s | OS=%s | device=cpu | threads=%s",
        platform.python_version(),
        torch.__version__,
        platform.platform(),
        torch.get_num_threads(),
    )
    results: list[dict[str, object]] = []
    histories: list[dict[str, object]] = []
    variant_samples: list[dict[str, object]] = []
    baseline_model = None
    baseline_tok = None
    for cfg in suite_configs():
        result, losses, model, tok = train_one(
            cfg, text, output_dir, logger, save_model=cfg.run_id == "baseline"
        )
        results.append(result)
        histories.extend(
            {"run_id": cfg.run_id, "step": step, "loss": loss}
            for step, loss in enumerate(losses, 1)
        )
        samples = generate_samples(
            model,
            tok,
            "春",
            (1201, 1202),
            1.0,
            20,
            min(120, args.max_new_tokens),
        )
        for sample_index, sample in enumerate(samples, 1):
            unique_ratio, repeated_ratio = repetition_metrics(sample)
            variant_samples.append(
                {
                    "run_id": cfg.run_id,
                    "sample_index": sample_index,
                    "seed": 1200 + sample_index,
                    "prompt": "春",
                    "temperature": 1.0,
                    "top_k": 20,
                    "text": sample,
                    "unique_3gram_ratio": unique_ratio,
                    "repeated_3gram_ratio": repeated_ratio,
                }
            )
        write_rows(output_dir / "results.csv", results)
        write_rows(output_dir / "histories.csv", histories)
        write_rows(output_dir / "variant_samples.csv", variant_samples)
        if cfg.run_id == "baseline":
            baseline_model, baseline_tok = model, tok
            baseline_samples(model, tok, output_dir, args.max_new_tokens)
        else:
            del model, tok
            gc.collect()
    assert baseline_model is not None and baseline_tok is not None
    blocks = []
    for row in variant_samples:
        if int(row["sample_index"]) == 1:
            blocks.append(f"## {row['run_id']}\n{row['text']}")
    (output_dir / "generated_variants.txt").write_text(
        "\n\n".join(blocks) + "\n", encoding="utf-8"
    )
    run_sampling_suite(baseline_model, baseline_tok, output_dir, args.max_new_tokens)
    metadata = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "device": "cpu",
        "threads": torch.get_num_threads(),
        "poem_count": len(POEMS),
        "corpus_chars": len(text),
        "vocab_size": baseline_tok.vocab_size,
        "seed": 42,
        "source": "作业三实验指南内置唐诗语料",
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("全部训练与采样实验完成")


def run_single(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(output_dir / "run_single.log")
    cfg = TrainConfig(
        run_id="single",
        dimension="单次运行",
        iters=args.iters,
        batch_size=args.batch_size,
        block_size=args.block_size,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_layer=args.n_layer,
        lr=args.lr,
        seed=args.seed,
        use_pos=not args.no_pos,
    )
    result, _, model, tok = train_one(
        cfg, build_corpus(args.extra_corpus), output_dir, logger, save_model=True
    )
    samples = generate_samples(
        model,
        tok,
        args.prompt,
        (args.seed + 100,),
        args.temperature,
        args.top_k,
        args.max_new_tokens,
        args.no_repeat_ngram,
    )
    print(samples[0])
    (output_dir / "generated_single.txt").write_text(samples[0] + "\n", encoding="utf-8")
    write_rows(output_dir / "results_single.csv", [result])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", action="store_true", help="运行报告所需的全部实验")
    parser.add_argument("--iters", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--n_embd", type=int, default=128)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--n_layer", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--prompt", default="春")
    parser.add_argument("--no_pos", action="store_true")
    parser.add_argument("--no_repeat_ngram", type=int, default=0)
    parser.add_argument("--extra_corpus", default="corpus_extra.txt")
    parser.add_argument("--output_dir", default=".")
    parser.add_argument("--threads", type=int, default=min(16, os.cpu_count() or 1))
    args = parser.parse_args()
    torch.set_num_threads(max(1, args.threads))
    return args


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.suite:
        run_suite(arguments)
    else:
        run_single(arguments)
