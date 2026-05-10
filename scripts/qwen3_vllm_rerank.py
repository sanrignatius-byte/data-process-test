#!/usr/bin/env python3
"""Qwen3-Reranker vLLM runner for dense top-N reranking.

This follows the official Qwen3-Reranker vLLM recipe: build a yes/no prompt,
generate one token with logprobs, and use P(yes) as the relevance score.
It intentionally avoids sentence-transformers, whose import can hang in the
current minerU environment on the login node.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable


KS = (1, 5, 10, 100)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_corpus(path: Path) -> dict[str, str]:
    out = {}
    for row in load_jsonl(path):
        pid = row.get("passage_id") or row.get("pid") or row.get("id")
        out[pid] = row.get("text", "")
    return out


def load_queries(path: Path) -> dict[str, str]:
    out = {}
    for row in load_jsonl(path):
        qid = row.get("query_id") or row.get("qid") or row.get("id")
        out[qid] = row.get("text") or row.get("query") or ""
    return out


def load_qrels(path: Path) -> dict[str, set[str]]:
    out: dict[str, set[str]] = defaultdict(set)
    for row in load_jsonl(path):
        if row.get("relevance", 1) >= 1:
            out[row["query_id"]].add(row["passage_id"])
    return out


def compute_metrics(rankings: dict[str, list[str]], qrels: dict[str, set[str]]) -> dict:
    recall_at = {k: [] for k in KS}
    ndcg_at = {k: [] for k in KS}
    hit_at = {k: [] for k in KS}
    rrs = []
    for qid, gold in qrels.items():
        ranked = rankings.get(qid, [])
        rr = 0.0
        for idx, pid in enumerate(ranked, start=1):
            if pid in gold:
                rr = 1.0 / idx
                break
        rrs.append(rr)
        for k in KS:
            topk = ranked[:k]
            hits = sum(1 for pid in topk if pid in gold)
            recall_at[k].append(hits / len(gold) if gold else 0.0)
            hit_at[k].append(1.0 if hits else 0.0)
            dcg = sum(1.0 / math.log2(i + 2) for i, pid in enumerate(topk) if pid in gold)
            idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold), k)))
            ndcg_at[k].append(dcg / idcg if idcg else 0.0)

    def avg(xs: Iterable[float]) -> float:
        xs = list(xs)
        return sum(xs) / len(xs) if xs else 0.0

    out = {"num_queries": len(qrels), "mrr": avg(rrs)}
    for k in KS:
        out[f"recall@{k}"] = avg(recall_at[k])
        out[f"hit@{k}"] = avg(hit_at[k])
        out[f"ndcg@{k}"] = avg(ndcg_at[k])
    return out


def format_instruction(instruction: str, query: str, doc: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Judge whether the Document meets the requirements based on the "
                "Query and the Instruct provided. Note that the answer can only "
                "be \"yes\" or \"no\"."
            ),
        },
        {
            "role": "user",
            "content": f"<Instruct>: {instruction}\n\n<Query>: {query}\n\n<Document>: {doc}",
        },
    ]


def build_prompts(tokenizer, pairs: list[tuple[str, str]], instruction: str,
                  max_prompt_tokens: int, suffix_tokens: list[int]):
    from vllm.inputs.data import TokensPrompt

    messages = [format_instruction(instruction, query, doc) for query, doc in pairs]
    tokenized = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    tokenized = [ids[:max_prompt_tokens] + suffix_tokens for ids in tokenized]
    return [TokensPrompt(prompt_token_ids=ids) for ids in tokenized]


def score_pairs(model, tokenizer, sampling_params, pairs: list[tuple[str, str]],
                instruction: str, max_prompt_tokens: int, suffix_tokens: list[int],
                true_token: int, false_token: int) -> list[float]:
    prompts = build_prompts(tokenizer, pairs, instruction, max_prompt_tokens, suffix_tokens)
    outputs = model.generate(prompts, sampling_params, use_tqdm=False)
    scores = []
    for out in outputs:
        final_logprobs = out.outputs[0].logprobs[-1]
        true_logit = final_logprobs[true_token].logprob if true_token in final_logprobs else -10.0
        false_logit = final_logprobs[false_token].logprob if false_token in final_logprobs else -10.0
        true_score = math.exp(true_logit)
        false_score = math.exp(false_logit)
        scores.append(true_score / (true_score + false_score))
    return scores


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranking-in", required=True, type=Path)
    ap.add_argument("--corpus", required=True, type=Path)
    ap.add_argument("--queries", required=True, type=Path)
    ap.add_argument("--qrels", required=True, type=Path)
    ap.add_argument("--model", default="Qwen/Qwen3-Reranker-4B")
    ap.add_argument("--instruction", required=True)
    ap.add_argument("--rerank-topn", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--limit-queries", type=int, default=0)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.82)
    ap.add_argument("--ranking-out", required=True, type=Path)
    ap.add_argument("--metrics-out", required=True, type=Path)
    ap.add_argument("--scores-out", required=True, type=Path)
    args = ap.parse_args()

    print(f"[info] loading queries from {args.queries}", flush=True)
    queries = load_queries(args.queries)
    print(f"[info] loading corpus from {args.corpus}", flush=True)
    corpus = load_corpus(args.corpus)
    print(f"[info] loading qrels from {args.qrels}", flush=True)
    qrels = load_qrels(args.qrels)
    print(f"[info] loading dense ranking from {args.ranking_in}", flush=True)
    dense_rows = load_jsonl(args.ranking_in)
    if args.limit_queries > 0:
        dense_rows = dense_rows[: args.limit_queries]
        print(f"[info] dry-run limit: using first {len(dense_rows)} rows", flush=True)
    print(f"[info] queries={len(queries)} corpus={len(corpus)} qrels={len(qrels)} "
          f"dense_rows={len(dense_rows)}", flush=True)

    print(f"[info] importing tokenizer/vLLM for {args.model}", flush=True)
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.distributed.parallel_state import destroy_model_parallel

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)
    max_prompt_tokens = args.max_model_len - len(suffix_tokens)
    true_token = tokenizer("yes", add_special_tokens=False).input_ids[0]
    false_token = tokenizer("no", add_special_tokens=False).input_ids[0]
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=1,
        logprobs=20,
        allowed_token_ids=[true_token, false_token],
    )

    print(f"[info] loading vLLM model max_model_len={args.max_model_len} "
          f"gpu_mem={args.gpu_memory_utilization}", flush=True)
    model = LLM(
        model=args.model,
        tensor_parallel_size=max(1, torch.cuda.device_count()),
        max_model_len=args.max_model_len,
        enable_prefix_caching=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    reranked: dict[str, list[str]] = {}
    score_rows: dict[str, list[tuple[str, float]]] = {}
    t0 = time.time()
    n_pairs = 0
    for row_idx, row in enumerate(dense_rows, start=1):
        qid = row["query_id"]
        qtext = queries.get(qid, "")
        pids = (row.get("top_k") or row.get("ranking") or [])[: args.rerank_topn]
        pid_scores: list[tuple[str, float]] = []
        for start in range(0, len(pids), args.batch_size):
            chunk = pids[start:start + args.batch_size]
            pairs = [(qtext, corpus.get(pid, "")) for pid in chunk]
            scores = score_pairs(
                model, tokenizer, sampling_params, pairs, args.instruction,
                max_prompt_tokens, suffix_tokens, true_token, false_token
            )
            pid_scores.extend(zip(chunk, scores))
            n_pairs += len(chunk)
        pid_scores.sort(key=lambda kv: kv[1], reverse=True)
        reranked[qid] = [pid for pid, _ in pid_scores]
        score_rows[qid] = pid_scores
        if row_idx % 10 == 0 or row_idx == len(dense_rows):
            print(f"[info] {row_idx}/{len(dense_rows)} queries reranked; "
                  f"pairs={n_pairs}; elapsed={time.time() - t0:.1f}s", flush=True)

    args.ranking_out.parent.mkdir(parents=True, exist_ok=True)
    with args.ranking_out.open("w", encoding="utf-8") as f:
        for qid, pids in reranked.items():
            f.write(json.dumps({"query_id": qid, "top_k": pids}) + "\n")
    with args.scores_out.open("w", encoding="utf-8") as f:
        for qid, pid_scores in score_rows.items():
            f.write(json.dumps({
                "query_id": qid,
                "scores": [{"passage_id": pid, "score": score} for pid, score in pid_scores],
            }) + "\n")

    dense_rankings = {
        row["query_id"]: (row.get("top_k") or row.get("ranking") or [])
        for row in dense_rows
    }
    metrics = {
        "dense_baseline": compute_metrics(dense_rankings, qrels),
        "qwen3_vllm_reranker": compute_metrics(reranked, qrels),
    }
    args.metrics_out.write_text(json.dumps({
        "config": {
            "model": args.model,
            "rerank_topn": args.rerank_topn,
            "batch_size": args.batch_size,
            "max_model_len": args.max_model_len,
            "ranking_in": str(args.ranking_in),
        },
        "metrics": metrics,
    }, indent=2))

    print(f"[info] wrote ranking -> {args.ranking_out}", flush=True)
    print(f"[info] wrote scores -> {args.scores_out}", flush=True)
    print(f"[info] wrote metrics -> {args.metrics_out}", flush=True)
    for name, m in metrics.items():
        print(f"  {name:24s} R@10={m['recall@10']:.4f} "
              f"R@100={m['recall@100']:.4f} MRR={m['mrr']:.4f}", flush=True)

    destroy_model_parallel()
    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
