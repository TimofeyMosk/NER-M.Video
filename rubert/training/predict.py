"""
Инференс обученной NER-модели по датасету запросов -> предсказанные BIO-теги.

Назначение: получить model_pred.jsonl для честной оценки на gold-сете через labeling/08_eval_gold.py.
Вход — jsonl со строками, содержащими как минимум {query_text, tokens} (gold.jsonl подходит:
у него есть и tokens, и эталонные tags, но здесь используются только tokens).
Выход — jsonl {query_text, tokens, tags} c ПРЕДСКАЗАННЫМИ тегами (по одному тегу на входной токен).

Токены уже разбиты (та же токенизация, что в обучении). Модель токенизирует их с
is_split_into_words=True, а предсказания сабвордов сворачиваются к словам (тег первого сабворда).
Длина tags == длине tokens (обрезанные по max_length слова получают 'O'), чтобы 08_eval_gold.py
мог сматчить токены построчно.

Запуск (после обучения, модель в ./model):
    python predict.py                     # gold.jsonl -> model_pred.jsonl в labeling/output/
    python predict.py --input path.jsonl --output out.jsonl
"""
import os, json, argparse
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
DEF_INPUT = os.path.join(REPO, "labeling", "output", "gold.jsonl")
DEF_OUTPUT = os.path.join(REPO, "labeling", "output", "model_pred.jsonl")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=os.path.join(HERE, "model"))
    p.add_argument("--input", default=DEF_INPUT)
    p.add_argument("--output", default=DEF_OUTPUT)
    p.add_argument("--max-length", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--no-cuda", action="store_true")
    return p.parse_args()


def pick_device(no_cuda):
    if not no_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    if not no_cuda and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def predict_batch(rows, tokenizer, model, device, id2label, max_length):
    tokens_list = [r["tokens"] for r in rows]
    enc = tokenizer(tokens_list, is_split_into_words=True, truncation=True,
                    max_length=max_length, padding=True, return_tensors="pt")
    logits = model(**{k: v.to(device) for k, v in enc.items()}).logits.cpu()
    preds = logits.argmax(-1)
    out = []
    for i, r in enumerate(rows):
        word_ids = enc.word_ids(i)
        n_words = len(r["tokens"])
        tags = ["O"] * n_words
        prev = None
        for pos, wid in enumerate(word_ids):
            if wid is None or wid == prev:
                prev = wid
                continue
            if wid < n_words:
                tags[wid] = id2label[int(preds[i][pos])]
            prev = wid
        out.append({"query_text": r["query_text"], "tokens": r["tokens"], "tags": tags})
    return out


def main():
    args = parse_args()
    if not os.path.isdir(args.model):
        raise SystemExit(f"[STOP] нет модели в {args.model} — сначала обучите (train_ner.py).")
    if not os.path.exists(args.input):
        raise SystemExit(f"[STOP] нет входа {args.input} — сначала разметьте gold.jsonl "
                         f"(см. labeling/GOLD_GUIDELINES.md).")

    device = pick_device(args.no_cuda)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForTokenClassification.from_pretrained(args.model).to(device).eval()
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    print(f"device={device} model={args.model} labels={list(id2label.values())}")

    rows = [json.loads(l) for l in open(args.input, encoding="utf-8")]
    rows = [r for r in rows if r.get("tokens")]  # пропускаем пустые
    print(f"[in] {len(rows)} запросов из {args.input}")

    results, tagc = [], {}
    for i in range(0, len(rows), args.batch_size):
        for r in predict_batch(rows[i:i + args.batch_size], tokenizer, model, device, id2label, args.max_length):
            results.append(r)
            for t in r["tags"]:
                tagc[t] = tagc.get(t, 0) + 1

    with open(args.output, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[saved] {args.output}  ({len(results)} строк)")
    print(f"tag dist: { {k: tagc[k] for k in sorted(tagc)} }")
    print(f"\nДалее: python labeling/08_eval_gold.py --gold labeling/output/gold.jsonl "
          f"--pred {os.path.relpath(args.output, REPO)}")


if __name__ == "__main__":
    main()
