"""
Дообучение rubert-tiny2 как token classifier (NER, BIO) для поиска М.Видео.

Сущности: CATEGORY / BRAND / ATTR (7 меток BIO, см. labels.json).
Данные: data/train.jsonl, data/val.jsonl — построчно {"tokens": [...], "tags": [...]}
        (уже нормализованы, lowercase, сплит без утечки train/val).

Самодостаточно: только torch + transformers + seqeval (без HF Trainer/accelerate/datasets).
Устройство выбирается автоматически: CUDA -> MPS -> CPU (на GPU-боксе будет CUDA + fp16).

Запуск:
    pip install -r requirements.txt
    python train_ner.py                      # полное обучение
    python train_ner.py --max-train 500 --epochs 1   # быстрый смоук-тест

Результат:
    model/                — сохранённая модель + токенизатор (лучший чекпойнт по val entity-F1)
    model/metrics.json    — per-type precision/recall/F1 на val + p50/p95 латентности (CPU, batch=1)
"""
import os, json, time, argparse, random
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    DataCollatorForTokenClassification, get_linear_schedule_with_warmup,
)
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = "cointegrated/rubert-tiny2"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--train", default=os.path.join(HERE, "data", "train.jsonl"))
    p.add_argument("--val", default=os.path.join(HERE, "data", "val.jsonl"))
    p.add_argument("--labels", default=os.path.join(HERE, "labels.json"))
    p.add_argument("--out", default=os.path.join(HERE, "model"))
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--max-length", type=int, default=64)
    p.add_argument("--max-train", type=int, default=0, help="ограничить train (0=всё) — для смоук-теста")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-cuda", action="store_true")
    return p.parse_args()


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(no_cuda):
    if not no_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    if not no_cuda and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def read_jsonl(path, limit=0):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def encode(examples, tokenizer, label2id, max_length):
    """Токенизация pre-tokenized входа + выравнивание меток по сабвордам (первый сабворд = метка)."""
    enc = tokenizer(
        [ex["tokens"] for ex in examples],
        is_split_into_words=True, truncation=True, max_length=max_length,
    )
    features = []
    for i, ex in enumerate(examples):
        word_ids = enc.word_ids(i)
        tags = ex["tags"]
        labels, prev = [], None
        for wid in word_ids:
            if wid is None:
                labels.append(-100)
            elif wid != prev:
                labels.append(label2id[tags[wid]])
            else:
                labels.append(-100)  # маркируем только первый сабворд слова
            prev = wid
        features.append({
            "input_ids": enc["input_ids"][i],
            "attention_mask": enc["attention_mask"][i],
            "labels": labels,
        })
    return features


@torch.no_grad()
def evaluate(model, loader, device, id2label):
    model.eval()
    y_true, y_pred = [], []
    for batch in loader:
        labels = batch["labels"]
        inp = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        logits = model(**inp).logits.detach().cpu()
        preds = logits.argmax(-1)
        for p_seq, l_seq in zip(preds, labels):
            t, pr = [], []
            for p_i, l_i in zip(p_seq.tolist(), l_seq.tolist()):
                if l_i == -100:
                    continue
                t.append(id2label[l_i])
                pr.append(id2label[p_i])
            y_true.append(t)
            y_pred.append(pr)
    return y_true, y_pred


def latency_benchmark(model, tokenizer, val_rows, max_length, n=1000):
    """p50/p95 латентности инференса на CPU, batch=1 (ориентир под приёмку <100мс)."""
    cpu = torch.device("cpu")
    model.to(cpu).eval()
    times = []
    sample = val_rows[:n]
    with torch.no_grad():
        for ex in sample:  # прогрев + замер
            enc = tokenizer(ex["tokens"], is_split_into_words=True,
                            truncation=True, max_length=max_length, return_tensors="pt")
            t0 = time.perf_counter()
            model(**enc)
            times.append((time.perf_counter() - t0) * 1000.0)
    times = np.array(times[5:]) if len(times) > 5 else np.array(times)
    return {"p50_ms": float(np.percentile(times, 50)),
            "p95_ms": float(np.percentile(times, 95)),
            "mean_ms": float(times.mean()), "n": int(len(times))}


def main():
    args = parse_args()
    set_seed(args.seed)
    device = pick_device(args.no_cuda)
    use_amp = device.type == "cuda"
    print(f"device={device} amp={use_amp} model={args.model}")

    labels = json.load(open(args.labels, encoding="utf-8"))
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for i, l in enumerate(labels)}

    train_rows = read_jsonl(args.train, args.max_train)
    val_rows = read_jsonl(args.val)
    print(f"train={len(train_rows):,} val={len(val_rows):,} labels={labels}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model, num_labels=len(labels), id2label=id2label, label2id=label2id,
    ).to(device)

    collator = DataCollatorForTokenClassification(tokenizer)
    train_feats = encode(train_rows, tokenizer, label2id, args.max_length)
    val_feats = encode(val_rows, tokenizer, label2id, args.max_length)
    train_loader = DataLoader(train_feats, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_feats, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        opt, int(total_steps * args.warmup_ratio), total_steps)
    scaler = torch.amp.GradScaler(enabled=use_amp)

    best_f1, best_report = -1.0, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for step, batch in enumerate(train_loader, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            opt.zero_grad()
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                loss = model(**batch).loss
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            scheduler.step()
            running += loss.item()
            if step % 200 == 0:
                print(f"  epoch {epoch} step {step}/{len(train_loader)} loss={running/step:.4f}")

        y_true, y_pred = evaluate(model, val_loader, device, id2label)
        f1 = f1_score(y_true, y_pred)
        p = precision_score(y_true, y_pred)
        r = recall_score(y_true, y_pred)
        print(f"[epoch {epoch}] val entity  P={p:.4f} R={r:.4f} F1={f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            os.makedirs(args.out, exist_ok=True)
            model.save_pretrained(args.out)
            tokenizer.save_pretrained(args.out)
            print(f"  -> сохранён лучший чекпойнт (F1={f1:.4f}) в {args.out}")

    # финальный отчёт по типам на лучшей модели
    print("\n=== per-type report (best) ===")
    for k, v in best_report.items():
        if isinstance(v, dict):
            print(f"  {k:14} P={v['precision']:.3f} R={v['recall']:.3f} F1={v['f1-score']:.3f} (n={int(v['support'])})")

    # латентность на CPU (грузим сохранённую лучшую модель)
    best_model = AutoModelForTokenClassification.from_pretrained(args.out)
    best_tok = AutoTokenizer.from_pretrained(args.out)
    lat = latency_benchmark(best_model, best_tok, val_rows, args.max_length)
    print(f"\n=== латентность CPU batch=1 (n={lat['n']}) ===")
    print(f"  p50={lat['p50_ms']:.1f}ms  p95={lat['p95_ms']:.1f}ms  mean={lat['mean_ms']:.1f}ms  (приёмка <100мс)")

    metrics = {"best_val_f1": best_f1, "per_type": best_report, "latency_cpu_batch1": lat,
               "config": {k: getattr(args, k) for k in ("model", "epochs", "batch_size", "lr",
                                                         "max_length", "seed", "max_train")}}

    def to_native(o):  # seqeval кладёт numpy-типы (support/int64) — приводим к native для JSON
        if hasattr(o, "item"):
            return o.item()
        raise TypeError(f"not serializable: {type(o)}")

    with open(os.path.join(args.out, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2, default=to_native)
    print(f"\n[saved] {args.out}/  (model + tokenizer + metrics.json)")


if __name__ == "__main__":
    main()
