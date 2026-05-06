#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from transformers import CLIPModel, CLIPProcessor

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


PROMPTS = {
    "sleeve": {
        "long sleeve": [
            "a fashion product photo of a long sleeve top",
            "studio product image of a top with full-length sleeves",
            "upper-body clothing whose sleeves extend to the wrists",
        ],
        "short sleeve": [
            "a fashion product photo of a short sleeve top",
            "studio product image of a top with abbreviated sleeves",
            "upper-body clothing whose sleeves stop above the elbow",
        ],
        "sleeveless": [
            "a fashion product photo of a sleeveless top",
            "studio product image of a top with no sleeves",
            "upper-body clothing with the arms fully uncovered",
        ],
    },
    "pattern": {
        "floral": [
            "a fashion product photo of floral clothing",
            "clothing with a flower-inspired print",
            "garment with floral patterning",
        ],
        "graphic": [
            "a fashion product photo of graphic clothing",
            "clothing with a bold printed visual design",
            "garment with graphic patterning",
        ],
        "striped": [
            "a fashion product photo of striped clothing",
            "clothing with repeated stripe-like bands",
            "garment with striped patterning",
        ],
        "solid": [
            "a fashion product photo of solid-color clothing",
            "plain clothing with no visible pattern",
            "garment with a solid surface",
        ],
    },
    "material": {
        "denim": [
            "a fashion product photo of denim clothing",
            "garment made from sturdy denim fabric",
            "clothing with a heavy twill denim texture",
        ],
        "chiffon": [
            "a fashion product photo of chiffon clothing",
            "garment made from sheer chiffon fabric",
            "clothing with a light flowing chiffon texture",
        ],
        "cotton": [
            "a fashion product photo of cotton clothing",
            "garment made from soft cotton fabric",
            "clothing with a matte cotton texture",
        ],
        "leather": [
            "a fashion product photo of leather clothing",
            "garment made from smooth leather material",
            "clothing with a dense leather texture",
        ],
        "knit": [
            "a fashion product photo of knit clothing",
            "garment made from knitted yarn",
            "clothing with a sweater-like knit texture",
        ],
    },
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def norm(x, eps=1e-12):
    d = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(d, eps, None)


class ClipWrap:
    def __init__(self, model_id, device):
        self.device = device
        self.model = CLIPModel.from_pretrained(model_id).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.model.eval()

    @torch.no_grad()
    def encode_text(self, texts):
        inputs = self.processor(
            text=list(texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        out = self.model.text_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        feat = self.model.text_projection(out.pooler_output)
        return norm(feat.detach().cpu().numpy())

    @torch.no_grad()
    def encode_img(self, paths, batch_size):
        feats = []

        for i in range(0, len(paths), batch_size):
            batch = paths[i:i + batch_size]
            imgs = [Image.open(p).convert("RGB") for p in batch]

            inputs = self.processor(images=imgs, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            out = self.model.vision_model(pixel_values=inputs["pixel_values"])
            feat = self.model.visual_projection(out.pooler_output)
            feats.append(feat.detach().cpu().numpy())

        return norm(np.concatenate(feats, axis=0))


def load_data(csv_path, image_root=None):
    df = pd.read_csv(csv_path)

    if "image_path" not in df.columns and "original_image_path" in df.columns:
        df = df.rename(columns={"original_image_path": "image_path"})

    need = {"image_path", "label", "style_split"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"missing columns: {sorted(miss)}")

    df = df.dropna(subset=["image_path", "label", "style_split"]).copy()
    df["image_path"] = df["image_path"].astype(str)

    if image_root is None:
        image_root = "/content/deepfashion_data"

    root = Path(image_root)

    def fix_path(p):
        p = str(p)
        pp = Path(p)

        if pp.exists():
            return str(pp)

        q = root / p
        if q.exists():
            return str(q)

        q = root / pp.name
        if q.exists():
            return str(q)

        if "img_highres/" in p:
            tail = p.split("img_highres/", 1)[1]
            q = root / "img_highres" / tail
            if q.exists():
                return str(q)
            q = root / tail
            if q.exists():
                return str(q)

        return p

    df["image_path"] = df["image_path"].map(fix_path)
    df["ok"] = df["image_path"].map(lambda p: Path(p).exists())
    df = df[df["ok"]].drop(columns=["ok"]).reset_index(drop=True)

    if len(df) == 0:
        raise ValueError("no valid image paths")

    return df


def split_data(df, ratio, seed):
    rng = np.random.default_rng(seed)
    calib = []
    test = []

    for _, g in df.groupby(["label", "style_split"], dropna=False):
        ids = np.arange(len(g))
        rng.shuffle(ids)

        n = max(1, int(round(len(g) * ratio)))
        calib.append(g.iloc[ids[:n]])

        if len(ids[n:]) > 0:
            test.append(g.iloc[ids[n:]])

    calib_df = pd.concat(calib, ignore_index=True)
    test_df = pd.concat(test, ignore_index=True)
    return calib_df, test_df


def task_from_labels(labels):
    labels = set(labels)

    for task, prompt_map in PROMPTS.items():
        if labels == set(prompt_map.keys()):
            return task

    return None


def text_proto(clip, prompt_map):
    names = list(prompt_map.keys())
    protos = []

    for c in names:
        f = clip.encode_text(prompt_map[c])
        p = norm(f.mean(axis=0, keepdims=True))[0]
        protos.append(p)

    return names, np.stack(protos, axis=0)


def get_style_dir(calib_df, feats, class_names, balanced=True):
    tmp = calib_df.reset_index(drop=True).copy()
    tmp["idx"] = np.arange(len(tmp))

    dirs = []

    if balanced:
        for c in class_names:
            sub = tmp[tmp["label"] == c]
            cat = sub[sub["style_split"] == "catalog_like"]["idx"].to_numpy()
            less = sub[sub["style_split"] == "less_curated"]["idx"].to_numpy()

            if len(cat) == 0 or len(less) == 0:
                continue

            v = feats[cat].mean(axis=0) - feats[less].mean(axis=0)
            dirs.append(v)

    if len(dirs) == 0:
        cat = tmp[tmp["style_split"] == "catalog_like"]["idx"].to_numpy()
        less = tmp[tmp["style_split"] == "less_curated"]["idx"].to_numpy()

        if len(cat) == 0 or len(less) == 0:
            raise ValueError("need both style groups")

        v = feats[cat].mean(axis=0) - feats[less].mean(axis=0)
        dirs.append(v)

    return norm(np.mean(dirs, axis=0, keepdims=True))[0]


def remove_style(feats, v, alpha):
    v = norm(v[None, :])[0]
    part = np.outer(feats @ v, v)
    return norm(feats - alpha * part)


def img_proto(df, feats, class_names):
    tmp = df.reset_index(drop=True).copy()
    tmp["idx"] = np.arange(len(tmp))

    protos = []

    for c in class_names:
        ids = tmp[tmp["label"] == c]["idx"].to_numpy()

        if len(ids) == 0:
            raise ValueError(f"no calibration images for {c}")

        p = feats[ids].mean(axis=0, keepdims=True)
        protos.append(norm(p)[0])

    return np.stack(protos, axis=0)


def fuse(text_p, img_p, beta):
    return norm((1 - beta) * text_p + beta * img_p)


def style_protos(df, feats, class_names):
    tmp = df.reset_index(drop=True).copy()
    tmp["idx"] = np.arange(len(tmp))

    out = {}

    for style, sub in tmp.groupby("style_split"):
        ps = []

        for c in class_names:
            ids = sub[sub["label"] == c]["idx"].to_numpy()

            if len(ids) == 0:
                raise ValueError(f"no samples for {c} in {style}")

            p = feats[ids].mean(axis=0, keepdims=True)
            ps.append(norm(p)[0])

        out[style] = np.stack(ps, axis=0)

    return out


def style_clf(feats, styles, seed):
    y = np.array([1 if s == "less_curated" else 0 for s in styles])
    clf = LogisticRegression(max_iter=1000, random_state=seed)
    clf.fit(feats, y)
    return clf


def style_prior_adj(calib_df, class_names, eps=1e-6):
    global_p = (
        calib_df["label"]
        .value_counts(normalize=True)
        .reindex(class_names, fill_value=0.0)
        .to_numpy(dtype=float)
    )

    out = {}

    for style, sub in calib_df.groupby("style_split"):
        style_p = (
            sub["label"]
            .value_counts(normalize=True)
            .reindex(class_names, fill_value=0.0)
            .to_numpy(dtype=float)
        )
        out[style] = np.log(style_p + eps) - np.log(global_p + eps)

    return out


def predict_basic(feats, protos, class_names):
    logits = feats @ protos.T
    ids = logits.argmax(axis=1)
    yhat = np.array([class_names[i] for i in ids])
    return yhat, logits


def predict_conditioned(feats, probs, global_p, style_p, adj, class_names, gamma):
    cat_p = style_p["catalog_like"]
    less_p = style_p["less_curated"]

    cat_adj = adj["catalog_like"]
    less_adj = adj["less_curated"]

    preds = []
    all_logits = []

    for f, prob in zip(feats, probs):
        p_style = norm((1 - prob) * cat_p + prob * less_p)

        base_logits = f @ global_p.T
        style_logits = np.sum(p_style * f[None, :], axis=1)

        a = (1 - prob) * cat_adj + prob * less_adj

        logits = 0.5 * base_logits + 0.5 * style_logits + gamma * a
        idx = int(np.argmax(logits))

        preds.append(class_names[idx])
        all_logits.append(logits)

    return np.array(preds), np.stack(all_logits, axis=0)


def evaluate(df, yhat, class_names):
    y = df["label"].to_numpy()

    acc = accuracy_score(y, yhat)
    f1 = f1_score(y, yhat, labels=list(class_names), average="macro")

    tmp = df.copy()
    tmp["prediction"] = yhat

    style_rows = []
    worst = 1.0

    for style, g in tmp.groupby("style_split"):
        a = accuracy_score(g["label"], g["prediction"])
        m = f1_score(g["label"], g["prediction"], labels=list(class_names), average="macro")

        worst = min(worst, a)

        style_rows.append({
            "style_split": style,
            "n": int(len(g)),
            "accuracy": float(a),
            "macro_f1": float(m),
        })

    class_style_rows = []

    for (style, label), g in tmp.groupby(["style_split", "label"]):
        class_style_rows.append({
            "style_split": style,
            "label": label,
            "n": int(len(g)),
            "accuracy": float(accuracy_score(g["label"], g["prediction"])),
        })

    return {
        "overall_accuracy": float(acc),
        "overall_macro_f1": float(f1),
        "worst_style_accuracy": float(worst),
        "style_metrics": style_rows,
        "class_style_metrics": class_style_rows,
    }


def results_to_frames(results):
    rows = []

    for k in ["baseline", "style_invariant", "full_projection", "style_conditioned"]:
        if k in results:
            rows.append({
                "variant": k,
                "accuracy": results[k]["overall_accuracy"],
                "macro_f1": results[k]["overall_macro_f1"],
                "worst_style_accuracy": results[k]["worst_style_accuracy"],
            })

    frames = {
        "summary": pd.DataFrame(rows),
        "baseline_style": pd.DataFrame(results["baseline"]["style_metrics"]),
        "style_invariant_style": pd.DataFrame(results["style_invariant"]["style_metrics"]),
        "full_projection_style": pd.DataFrame(results["full_projection"]["style_metrics"]),
        "style_conditioned_style": pd.DataFrame(results["style_conditioned"]["style_metrics"]),
        "baseline_class_style": pd.DataFrame(results["baseline"]["class_style_metrics"]),
        "style_invariant_class_style": pd.DataFrame(results["style_invariant"]["class_style_metrics"]),
        "full_projection_class_style": pd.DataFrame(results["full_projection"]["class_style_metrics"]),
        "style_conditioned_class_style": pd.DataFrame(results["style_conditioned"]["class_style_metrics"]),
    }

    return frames


def run(args):
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print("device:", device)

    df = load_data(args.csv, args.image_root)

    if args.max_rows is not None and len(df) > args.max_rows:
        df = df.sample(args.max_rows, random_state=args.seed).reset_index(drop=True)

    task = args.task
    if task is None:
        task = task_from_labels(sorted(df["label"].unique()))

    if task not in PROMPTS:
        raise ValueError("cannot infer task. use --task sleeve, pattern, or material")

    print("task:", task)
    print("rows:", len(df))
    print("labels:")
    print(df["label"].value_counts())
    print("styles:")
    print(df["style_split"].value_counts())

    clip = ClipWrap(args.model_id, device)

    class_names, text_p = text_proto(clip, PROMPTS[task])

    calib_df, test_df = split_data(df, args.calib_ratio, args.seed)

    print("calib rows:", len(calib_df))
    print("test rows:", len(test_df))

    calib_f = clip.encode_img(calib_df["image_path"].tolist(), args.batch_size)
    test_f = clip.encode_img(test_df["image_path"].tolist(), args.batch_size)

    base_y, _ = predict_basic(test_f, text_p, class_names)
    base_eval = evaluate(test_df, base_y, class_names)

    v = get_style_dir(
        calib_df,
        calib_f,
        class_names,
        balanced=not args.unbalanced_style_direction,
    )

    calib_de = remove_style(calib_f, v, args.alpha)
    test_de = remove_style(test_f, v, args.alpha)

    img_p = img_proto(calib_df, calib_de, class_names)
    final_p = fuse(text_p, img_p, args.beta)

    de_y, _ = predict_basic(test_de, final_p, class_names)
    de_eval = evaluate(test_df, de_y, class_names)

    calib_full = remove_style(calib_f, v, 1.0)
    test_full = remove_style(test_f, v, 1.0)

    img_p_full = img_proto(calib_df, calib_full, class_names)
    final_p_full = fuse(text_p, img_p_full, args.beta)

    full_y, _ = predict_basic(test_full, final_p_full, class_names)
    full_eval = evaluate(test_df, full_y, class_names)

    sp = style_protos(calib_df, calib_de, class_names)
    adj = style_prior_adj(calib_df, class_names)

    clf = style_clf(calib_f, calib_df["style_split"].tolist(), args.seed)
    style_prob = clf.predict_proba(test_f)[:, 1]

    cond_y, _ = predict_conditioned(
        test_de,
        style_prob,
        final_p,
        sp,
        adj,
        class_names,
        args.gamma,
    )
    cond_eval = evaluate(test_df, cond_y, class_names)

    out = {
        "config": {
            "csv": str(args.csv),
            "image_root": str(args.image_root),
            "task": task,
            "model_id": args.model_id,
            "seed": args.seed,
            "device": device,
            "batch_size": args.batch_size,
            "calib_ratio": args.calib_ratio,
            "alpha": args.alpha,
            "beta": args.beta,
            "gamma": args.gamma,
            "max_rows": args.max_rows,
        },
        "dataset": {
            "total_rows": int(len(df)),
            "calibration_rows": int(len(calib_df)),
            "test_rows": int(len(test_df)),
            "class_names": class_names,
            "style_counts": df["style_split"].value_counts().to_dict(),
            "label_counts": df["label"].value_counts().to_dict(),
        },
        "baseline": base_eval,
        "style_invariant": de_eval,
        "full_projection": full_eval,
        "style_conditioned": cond_eval,
        "deltas": {
            "accuracy": de_eval["overall_accuracy"] - base_eval["overall_accuracy"],
            "macro_f1": de_eval["overall_macro_f1"] - base_eval["overall_macro_f1"],
            "worst_style_accuracy": de_eval["worst_style_accuracy"] - base_eval["worst_style_accuracy"],
        },
        "full_projection_deltas": {
            "accuracy": full_eval["overall_accuracy"] - base_eval["overall_accuracy"],
            "macro_f1": full_eval["overall_macro_f1"] - base_eval["overall_macro_f1"],
            "worst_style_accuracy": full_eval["worst_style_accuracy"] - base_eval["worst_style_accuracy"],
        },
        "conditioned_deltas": {
            "accuracy": cond_eval["overall_accuracy"] - base_eval["overall_accuracy"],
            "macro_f1": cond_eval["overall_macro_f1"] - base_eval["overall_macro_f1"],
            "worst_style_accuracy": cond_eval["worst_style_accuracy"] - base_eval["worst_style_accuracy"],
        },
        "style_classifier": {
            "coef_norm": float(np.linalg.norm(clf.coef_)),
            "mean_less_curated_prob": float(np.mean(style_prob)),
        },
        "style_direction_norm": float(np.linalg.norm(v)),
    }

    return out


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--image-root", type=Path, default="/content/deepfashion_data")

    p.add_argument("--task", type=str, default=None, choices=["sleeve", "pattern", "material"])
    p.add_argument("--model-id", type=str, default="openai/clip-vit-base-patch32")

    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--calib-ratio", type=float, default=0.2)

    p.add_argument("--alpha", type=float, default=0.7)
    p.add_argument("--beta", type=float, default=0.35)
    p.add_argument("--gamma", type=float, default=0.4)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-rows", type=int, default=None)

    p.add_argument("--cpu", action="store_true")
    p.add_argument("--unbalanced-style-direction", action="store_true")
    p.add_argument("--output-json", type=Path, default=None)

    return p.parse_args()


def main():
    args = parse_args()
    out = run(args)

    print(json.dumps(out, indent=2))

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
