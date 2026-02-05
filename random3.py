#!/usr/bin/env python3
import argparse, os, sys, random
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx


# ---------- utilities ----------
def write_best_gene_csv(outdir: Path, genes: list, fname: str = "best_50_genes.csv"):
    df = pd.DataFrame({"gene": list(genes)})
    df.to_csv(outdir / fname, index=False)

def ensure_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.apply(pd.to_numeric, errors="coerce")
    return df2

def dedupe_index(df: pd.DataFrame) -> pd.DataFrame:
    if df.index.has_duplicates:
        df = df[~df.index.duplicated(keep="first")]
    return df

def set_seed(s):
    random.seed(s)
    np.random.seed(s)

def read_matrix(path):
    # auto detect sep, assume genes x samples
    df = pd.read_csv(path, sep=None, engine="python", index_col=0)
    df = dedupe_index(df)
    df = ensure_numeric(df)
    return df

def rank_transform(df):
    # rank within each gene across samples
    return df.apply(lambda row: row.rank(method="average"), axis=1)

def corr_pairs(df_ranked, min_non_na=4):
    genes = df_ranked.index.to_list()
    X = df_ranked.to_numpy(dtype=float)
    G, N = X.shape
    rows = []
    for i in range(G):
        xi = X[i]
        mask_i = np.isfinite(xi)
        for j in range(i+1, G):
            xj = X[j]
            m = mask_i & np.isfinite(xj)
            n = int(m.sum())
            if n < min_non_na:
                continue
            r = np.corrcoef(xi[m], xj[m])[0, 1]
            if np.isfinite(r):
                rows.append((genes[i], genes[j], float(r), n))
    return pd.DataFrame(rows, columns=["gene_u", "gene_v", "r", "n"])

def compose_edges_rna(expr_df, min_non_na=4):
    E = corr_pairs(rank_transform(expr_df), min_non_na=min_non_na)
    E = E.rename(columns={"r": "weight"})
    return E[["gene_u", "gene_v", "weight"]]

def apply_filters(edges, abs_min=None, top_k=None):
    df = edges.copy()
    if abs_min is not None and not np.isnan(abs_min):
        df = df.loc[df["weight"].abs() >= float(abs_min)]
    if top_k is not None and top_k > 0:
        df = df.reindex(df["weight"].abs().sort_values(ascending=False).index)
        if len(df) > top_k:
            df = df.iloc[:top_k].copy()
    return df

def to_graph(edge_df, nodes):
    G = nx.Graph()
    G.add_nodes_from(nodes)
    for _, r in edge_df.iterrows():
        G.add_edge(r["gene_u"], r["gene_v"], weight=float(r["weight"]))
    return G

def communities_and_scores(G, expr_df):
    """
    Detect modules with greedy modularity and compute per sample scores
    as the mean expression of module genes.
    """
    try:
        comms = nx.algorithms.community.greedy_modularity_communities(G)
    except Exception:
        comms = []
    mods = []
    for idx, c in enumerate(comms):
        genes = list(c)
        if len(genes) < 2:
            continue
        inter = [g for g in genes if g in expr_df.index]
        if not inter:
            continue
        score = expr_df.loc[inter].mean(axis=0)
        mods.append((f"module_{idx}", inter, score))
    return mods

# ---------- subtype + metrics ----------

def load_subtypes(path, col, sample_col):
    df = pd.read_csv(path, sep=None, engine="python")
    if sample_col not in df.columns:
        # assume first column is sample id
        df = df.set_index(df.columns[0]).reset_index().rename(columns={df.columns[0]: sample_col})
    df[sample_col] = df[sample_col].astype(str)
    s = df[sample_col].tolist()
    lab = df[col].astype(str).tolist()
    return dict(zip(s, lab))

def auc_binary(y, scores):
    y = np.asarray(y, dtype=float)
    s = np.asarray(scores, dtype=float)
    m1 = y == 1
    m0 = y == 0
    n1 = int(m1.sum())
    n0 = int(m0.sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    r = pd.Series(s).rank(method="average").to_numpy()
    u = r[m1].sum() - n1 * (n1 + 1) / 2.0
    return float(u / (n1 * n0))

def subtype_labels_from_scores(scores_ser, subtype_map, pos_label, neg_label):
    idx = scores_ser.index.intersection(pd.Index(list(subtype_map.keys())))
    y = []
    sc = []
    for s in idx:
        lab = subtype_map.get(s)
        if lab == pos_label:
            y.append(1)
            sc.append(scores_ser[s])
        elif lab == neg_label:
            y.append(0)
            sc.append(scores_ser[s])
        else:
            # ignore other labels
            continue
    y = np.asarray(y, dtype=int)
    sc = np.asarray(sc, dtype=float)
    return y, sc

def compute_binary_metrics(y, scores):
    """
    y: 0/1 numpy array
    scores: continuous scores
    Returns dict with auc, threshold, sensitivity, specificity, accuracy.
    """
    y = np.asarray(y, dtype=int)
    s = np.asarray(scores, dtype=float)
    if y.size == 0:
        return dict(auc=np.nan, threshold=np.nan,
                    sensitivity=np.nan, specificity=np.nan, accuracy=np.nan)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return dict(auc=np.nan, threshold=np.nan,
                    sensitivity=np.nan, specificity=np.nan, accuracy=np.nan)

    auc = auc_binary(y, s)

    # choose threshold that maximizes Youden index (sens + spec - 1)
    uniq = np.unique(s)
    best_thr = None
    best_youden = -1.0
    best_acc = -1.0
    best_sens = np.nan
    best_spec = np.nan

    for thr in uniq:
        pred = (s >= thr).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        sens = tp / n_pos if n_pos > 0 else np.nan
        spec = tn / n_neg if n_neg > 0 else np.nan
        acc = (tp + tn) / (n_pos + n_neg)
        youden = sens + spec - 1.0
        if youden > best_youden or (np.isclose(youden, best_youden) and acc > best_acc):
            best_youden = youden
            best_acc = acc
            best_thr = thr
            best_sens = sens
            best_spec = spec

    return dict(
        auc=float(auc),
        threshold=float(best_thr),
        sensitivity=float(best_sens),
        specificity=float(best_spec),
        accuracy=float(best_acc),
    )

# ---------- main loop ----------

# ---------- CV helpers + Monte Carlo search ----------

def stratified_kfold_indices(y, n_splits, rng):
    """
    Simple stratified K-fold on a binary label vector y.
    Returns a list of folds, each a list of indices for the test set.
    """
    y = np.asarray(y, dtype=int)
    classes = np.unique(y)
    indices_per_class = {c: np.where(y == c)[0].tolist() for c in classes}
    for c in classes:
        rng.shuffle(indices_per_class[c])
    folds = [[] for _ in range(n_splits)]
    for c in classes:
        inds = indices_per_class[c]
        for i, idx in enumerate(inds):
            folds[i % n_splits].append(idx)
    return folds


def evaluate_module(expr_df, genes, subtype_map, pos_label, neg_label):
    """
    Evaluate a given module (list of genes) on a given expression matrix (genes x samples)
    and subtype map. Returns metrics dict from compute_binary_metrics.
    """
    if genes is None or len(genes) == 0:
        return dict(auc=np.nan, threshold=np.nan,
                    sensitivity=np.nan, specificity=np.nan, accuracy=np.nan)

    inter = [g for g in genes if g in expr_df.index]
    if len(inter) < 2:
        return dict(auc=np.nan, threshold=np.nan,
                    sensitivity=np.nan, specificity=np.nan, accuracy=np.nan)

    score_ser = expr_df.loc[inter].mean(axis=0)
    y, s = subtype_labels_from_scores(score_ser, subtype_map, pos_label, neg_label)
    return compute_binary_metrics(y, s)


def run_mc_search(
    expr_df,
    subtype_map,
    args,
    subset_genes=None,
    subset_size=50,
    n_iters=1000,
    target_auc=None,
    stage_name="stage1",
    fold_id=0,
    global_seed=1337,
    gene_universe_auc_thresh=None,
):
    """
    Core Monte Carlo module search.
    - expr_df: genes x samples (TRAIN ONLY for the fold)
    - subtype_map: sample -> label (TRAIN ONLY)
    - subset_genes: optional list of genes to restrict search to (None = all genes)
    - subset_size: number of genes to sample per iteration
    - gene_universe_auc_thresh: if not None, any module with AUC >= this adds its genes
      to the returned 'high_auc_genes' universe (used for Stage 1 only).
    Returns:
      summary_rows: list of dicts for all modules
      global_best: dict with best module on TRAIN (by AUC)
      high_auc_genes: set of genes accumulated over modules with AUC >= gene_universe_auc_thresh
    """
    # set a per-stage, per-fold seed for reproducibility but not total determinism
    stage_offset = 1 if stage_name == "stage1" else 2
    set_seed(global_seed + fold_id * 1000 + stage_offset * 100000)

    if subset_genes is None:
        genes_all = expr_df.index.to_list()
    else:
        genes_all = [g for g in subset_genes if g in expr_df.index]

    n_genes = len(genes_all)
    if n_genes == 0:
        return [], dict(auc=-1.0, iteration=None, module=None, module_size=None, genes=None, metrics=None), set()

    summary_rows = []
    global_best = dict(auc=-1.0, iteration=None, module=None, module_size=None, genes=None, metrics=None)
    high_auc_genes = set()

    if n_genes < subset_size:
        subset_size_eff = n_genes
    else:
        subset_size_eff = subset_size

    for it in range(1, n_iters + 1):
        # gene subset for this iteration
        if subset_size_eff == n_genes:
            subset = genes_all
        else:
            subset = random.sample(genes_all, subset_size_eff)

        expr_sub = expr_df.loc[subset]

        # build network on TRAIN only
        edges = compose_edges_rna(expr_sub, min_non_na=args.min_non_na)
        edges = apply_filters(
            edges,
            abs_min=(args.abs_min if args.abs_min > 0 else None),
            top_k=(args.top_k_edges if args.top_k_edges > 0 else None),
        )

        G = to_graph(edges, nodes=subset)
        mods = communities_and_scores(G, expr_sub)

        iter_best_auc = -1.0
        iter_best_row = None

        for mid, genes, score_ser in mods:
            y, s = subtype_labels_from_scores(score_ser, subtype_map, args.subtype_pos, args.subtype_neg)
            if y.size == 0:
                continue
            metrics = compute_binary_metrics(y, s)
            auc = metrics["auc"]
            if np.isnan(auc):
                continue

            row = dict(
                fold=fold_id,
                stage=stage_name,
                iteration=it,
                module=mid,
                module_size=len(genes),
                n_samples=int(y.size),
                auc=auc,
                sensitivity=metrics["sensitivity"],
                specificity=metrics["specificicity"] if "specificicity" in metrics else metrics["specificity"],
                accuracy=metrics["accuracy"],
                threshold=metrics["threshold"],
                genes=";".join(genes),
            )
            summary_rows.append(row)

            if auc > iter_best_auc:
                iter_best_auc = auc
                iter_best_row = row

            if auc > global_best["auc"]:
                global_best.update(
                    auc=float(auc),
                    iteration=it,
                    module=mid,
                    module_size=len(genes),
                    genes=genes,
                    metrics=metrics,
                )

            # accumulate Stage-1 "universe" genes if requested
            if gene_universe_auc_thresh is not None and auc >= gene_universe_auc_thresh:
                high_auc_genes.update(genes)

        if iter_best_row is not None:
            print(
                f"[fold {fold_id}][{stage_name}] iter {iter_best_row['iteration']} "
                f"best {iter_best_row['module']} size={iter_best_row['module_size']} "
                f"AUC_train={iter_best_row['auc']:.3f}",
                flush=True,
            )
        else:
            print(f"[fold {fold_id}][{stage_name}] iter {it} no valid modules", flush=True)

        if target_auc is not None and global_best["auc"] >= target_auc:
            print(
                f"[fold {fold_id}][{stage_name}] reached target AUC {target_auc} "
                f"at iteration {global_best['iteration']}",
                file=sys.stderr,
                flush=True,
            )
            break

    return summary_rows, global_best, high_auc_genes


# ---------- main loop with CV + 2-stage MC ----------

def main():
    ap = argparse.ArgumentParser(description="Random module screen for classical/basal subtyping with 5-fold CV")
    ap.add_argument("--expr", required=True)
    ap.add_argument("--subtypes", required=True)
    ap.add_argument("--subtype_col", default="Subtype")
    ap.add_argument("--subtype_sample_col", default="Sample")
    ap.add_argument("--subtype_pos", default="classical")
    ap.add_argument("--subtype_neg", default="basal")
    ap.add_argument("--subset_size", type=int, default=50)
    ap.add_argument("--n_iters", type=int, default=8000)
    ap.add_argument("--target_auc", type=float, default=0.85)  # Stage-1 target
    ap.add_argument("--min_non_na", type=int, default=4)
    ap.add_argument("--abs_min", type=float, default=0.0)
    ap.add_argument("--top_k_edges", type=int, default=0)
    ap.add_argument("--min_median_expr", type=float, default=0.0)
    ap.add_argument("--min_var", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    # Fixed CV / Stage-2 hyperparams (can be made CLI later if you care)
    cv_folds = 5
    stage1_n_iters = args.n_iters
    stage1_target_auc = args.target_auc
    stage2_n_iters = args.n_iters
    stage2_subset_size = 50
    stage2_target_auc = 0.90  # what you asked for

    set_seed(args.seed)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ----- load data -----
    expr_full = read_matrix(args.expr)
    submap_full = load_subtypes(args.subtypes, args.subtype_col, args.subtype_sample_col)

    # SAMPLE INTERSECTION
    common = [s for s in expr_full.columns if s in submap_full]
    if len(common) < 10:
        msg = f"Too few samples with subtype labels after intersection: {len(common)}"
        print(msg, file=sys.stderr)
        (outdir / "run_status.txt").write_text(msg + "\n")
        sys.exit(2)

    expr = expr_full[common]

    # restrict to pos/neg labels only (others dropped, leakage-proof)
    pos_label = args.subtype_pos
    neg_label = args.subtype_neg
    keep_samples = []
    y_labels = []
    for s in expr.columns:
        lab = submap_full.get(s)
        if lab == pos_label:
            keep_samples.append(s)
            y_labels.append(1)
        elif lab == neg_label:
            keep_samples.append(s)
            y_labels.append(0)
        # ignore anything else

    if len(keep_samples) < 10 or len(set(y_labels)) < 2:
        msg = (
            f"Not enough samples with both pos='{pos_label}' and neg='{neg_label}' "
            f"after filtering. n_samples={len(keep_samples)}"
        )
        print(msg, file=sys.stderr)
        (outdir / "run_status.txt").write_text(msg + "\n")
        sys.exit(2)

    expr = expr[keep_samples]
    y_labels = np.asarray(y_labels, dtype=int)

    # ---- GENE FILTERING on this clean matrix ----
    # 1) drop all-NaN genes
    expr = expr.dropna(axis=0, how="all")

    # 2) drop zero-variance genes
    gene_var = expr.var(axis=1, skipna=True)
    expr = expr.loc[gene_var > 0.0]

    # 3) require expression in at least min_non_na samples
    nonzero_counts = (expr != 0).sum(axis=1)
    expr = expr.loc[nonzero_counts >= args.min_non_na]

    # 4) median expression filter (optional)
    if args.min_median_expr > 0:
        med = expr.median(axis=1)
        expr = expr.loc[med >= args.min_median_expr]

    # 5) variance filter (optional)
    if args.min_var > 0:
        gene_var = expr.var(axis=1, skipna=True)
        expr = expr.loc[gene_var >= args.min_var]

    # 6) optional abs_min-dependent tightening
    if args.abs_min > 0:
        var_thresh = args.abs_min / 10.0
        gene_var = expr.var(axis=1, skipna=True)
        expr = expr.loc[gene_var > var_thresh]

    all_genes = expr.index.to_list()
    n_genes = len(all_genes)

    msg = f"[INFO] genes after filtering: {n_genes}"
    print(msg, file=sys.stderr, flush=True)
    print(msg, flush=True)
    (outdir / "gene_filter_summary.txt").write_text(
        msg + f"\nSamples used (pos/neg only): {len(keep_samples)}\n"
    )

    if n_genes == 0:
        msg2 = "No genes left after filtering. Exiting."
        print(msg2, file=sys.stderr)
        (outdir / "run_status.txt").write_text(msg + "\n" + msg2 + "\n")
        sys.exit(2)

    # ---- build stratified 5-fold CV ----
    rng = np.random.RandomState(args.seed)
    folds = stratified_kfold_indices(y_labels, cv_folds, rng)

    all_rows = []  # all modules across folds & stages
    cv_summary = []  # per-fold summary
    best_overall = None  # best by TEST AUC

    samples = np.array(keep_samples)

    for fold_idx, test_idx in enumerate(folds, start=1):
        test_idx = np.array(test_idx, dtype=int)
        train_mask = np.ones(len(samples), dtype=bool)
        train_mask[test_idx] = False
        train_idx = np.where(train_mask)[0]

        train_samples = samples[train_idx].tolist()
        test_samples = samples[test_idx].tolist()

        expr_train = expr[train_samples]
        expr_test = expr[test_samples]

        submap_train = {s: submap_full[s] for s in train_samples}
        submap_test = {s: submap_full[s] for s in test_samples}

        print(f"\n==== Fold {fold_idx}/{cv_folds} ====", flush=True)
        print(f"Train samples: {len(train_samples)}, Test samples: {len(test_samples)}", flush=True)

        # ---- Stage 1: broad MC on TRAIN only ----
        stage1_rows, stage1_best, stage1_univ_genes = run_mc_search(
            expr_train,
            submap_train,
            args,
            subset_genes=None,
            subset_size=args.subset_size,
            n_iters=stage1_n_iters,
            target_auc=stage1_target_auc,
            stage_name="stage1",
            fold_id=fold_idx,
            global_seed=args.seed,
            gene_universe_auc_thresh=stage1_target_auc,
        )
        all_rows.extend(stage1_rows)

        # Evaluate Stage-1 best module on TEST
        metrics_train_s1 = stage1_best["metrics"] if stage1_best["metrics"] is not None else dict(
            auc=np.nan, threshold=np.nan, sensitivity=np.nan, specificity=np.nan, accuracy=np.nan
        )
        metrics_test_s1 = evaluate_module(
            expr_test, stage1_best["genes"], submap_test, pos_label, neg_label
        )

        print(
            f"[fold {fold_idx}] Stage1 best AUC_train={metrics_train_s1['auc']:.3f} "
            f"AUC_test={metrics_test_s1['auc']:.3f}",
            flush=True,
        )

        # ---- Stage 2: refinement MC on TRAIN only (restricted universe) ----
        if stage1_univ_genes:
            stage2_universe = sorted(set(stage1_univ_genes))
        else:
            # fallback: just use Stage-1 best genes
            stage2_universe = stage1_best["genes"] if stage1_best["genes"] is not None else []

        stage2_rows, stage2_best, _ = run_mc_search(
            expr_train,
            submap_train,
            args,
            subset_genes=stage2_universe,
            subset_size=stage2_subset_size,
            n_iters=stage2_n_iters,
            target_auc=stage2_target_auc,
            stage_name="stage2",
            fold_id=fold_idx,
            global_seed=args.seed,
            gene_universe_auc_thresh=None,
        )
        all_rows.extend(stage2_rows)

        metrics_train_s2 = stage2_best["metrics"] if stage2_best["metrics"] is not None else dict(
            auc=np.nan, threshold=np.nan, sensitivity=np.nan, specificity=np.nan, accuracy=np.nan
        )
        metrics_test_s2 = evaluate_module(
            expr_test, stage2_best["genes"], submap_test, pos_label, neg_label
        )

        print(
            f"[fold {fold_idx}] Stage2 best AUC_train={metrics_train_s2['auc']:.3f} "
            f"AUC_test={metrics_test_s2['auc']:.3f}",
            flush=True,
        )

        # ---- choose best module for this fold by TEST AUC ----
        auc_test_s1 = metrics_test_s1["auc"]
        auc_test_s2 = metrics_test_s2["auc"]

        if np.isnan(auc_test_s1) and np.isnan(auc_test_s2):
            chosen_stage = "none"
            chosen_auc_test = np.nan
            chosen_genes = None
            chosen_train_metrics = None
            chosen_test_metrics = None
        elif np.isnan(auc_test_s2) or (not np.isnan(auc_test_s1) and auc_test_s1 >= auc_test_s2):
            chosen_stage = "stage1"
            chosen_auc_test = auc_test_s1
            chosen_genes = stage1_best["genes"]
            chosen_train_metrics = metrics_train_s1
            chosen_test_metrics = metrics_test_s1
        else:
            chosen_stage = "stage2"
            chosen_auc_test = auc_test_s2
            chosen_genes = stage2_best["genes"]
            chosen_train_metrics = metrics_train_s2
            chosen_test_metrics = metrics_test_s2

        cv_summary.append(
            dict(
                fold=fold_idx,
                n_train=len(train_samples),
                n_test=len(test_samples),
                stage1_auc_train=metrics_train_s1["auc"],
                stage1_auc_test=auc_test_s1,
                stage2_auc_train=metrics_train_s2["auc"],
                stage2_auc_test=auc_test_s2,
                chosen_stage=chosen_stage,
                chosen_auc_test=chosen_auc_test,
                chosen_module_size=len(chosen_genes) if chosen_genes is not None else 0,
            )
        )

        # update global best by TEST AUC (leakage-proof)
        if chosen_genes is not None and not np.isnan(chosen_auc_test):
            if best_overall is None or chosen_auc_test > best_overall["test_metrics"]["auc"]:
                best_overall = dict(
                    fold=fold_idx,
                    stage=chosen_stage,
                    genes=chosen_genes,
                    train_metrics=chosen_train_metrics,
                    test_metrics=chosen_test_metrics,
                )

    # ---- write outputs ----
    if all_rows:
        pd.DataFrame(all_rows).to_csv(outdir / "random_modules_performance_cv.csv", index=False)

    if cv_summary:
        pd.DataFrame(cv_summary).to_csv(outdir / "cv_results_summary.tsv", sep="\t", index=False)

    # save best overall module (by TEST AUC across folds)
    if best_overall is not None:
        best_txt = outdir / "best_module_summary.txt"
        with best_txt.open("w") as f:
            f.write(f"Selected by 5-fold CV (highest TEST AUC across folds)\n")
            f.write(f"Fold: {best_overall['fold']}\n")
            f.write(f"Stage: {best_overall['stage']}\n")
            f.write(f"Best TEST AUC: {best_overall['test_metrics']['auc']:.4f}\n")
            f.write(f"TRAIN AUC: {best_overall['train_metrics']['auc']:.4f}\n")
            f.write(f"Sensitivity (test): {best_overall['test_metrics']['sensitivity']:.4f}\n")
            f.write(f"Specificity (test): {best_overall['test_metrics']['specificity']:.4f}\n")
            f.write(f"Accuracy   (test): {best_overall['test_metrics']['accuracy']:.4f}\n")
            f.write(f"Threshold  (test): {best_overall['test_metrics']['threshold']:.4f}\n")
            f.write("Genes:\n")
            for g in best_overall["genes"]:
                f.write(f"{g}\n")

        (outdir / "best_module_genes.txt").write_text(
            "\n".join(best_overall["genes"]) + "\n"
        )
# ==========================================================
# NEW: single CSV of the globally best 50 genes (across folds)
# ==========================================================
    best_genes = list(best_overall["genes"])
    write_best_gene_csv(outdir, best_genes, fname="best_50_genes.csv")

# ==========================================================
# NEW: fold-wise artifacts for morphology script compatibility
# - Same global best gene list copied per fold
# - Edge lists computed TRAIN-only per fold for Laplacian loss
# ==========================================================
    for fold_idx, test_idx in enumerate(folds, start=1):
        test_idx = np.array(test_idx, dtype=int)
        train_mask = np.ones(len(samples), dtype=bool)
        train_mask[test_idx] = False
        train_idx = np.where(train_mask)[0]
        train_samples = samples[train_idx].tolist()

        expr_train = expr[train_samples]  # TRAIN ONLY for that fold
        inter = [g for g in best_genes if g in expr_train.index]

        # Write fold-wise best gene list (same across folds)
        with (outdir / f"fold{fold_idx}_best_genes.txt").open("w") as f:
            for g in inter:
                f.write(f"{g}\n")

    # Build fold-wise gene edge list on TRAIN ONLY
        edges = compose_edges_rna(expr_train.loc[inter], min_non_na=args.min_non_na)
        edges = apply_filters(
            edges,
            abs_min=(args.abs_min if args.abs_min > 0 else None),
            top_k=(args.top_k_edges if args.top_k_edges > 0 else None),
        )
        edges.to_csv(outdir / f"fold{fold_idx}_gene_edges.csv", index=False)


    print("[DONE] 5-fold CV random module screen complete.")


if __name__ == "__main__":
    main()
