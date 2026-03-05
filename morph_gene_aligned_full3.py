#!/usr/bin/env python3
"""
Gene–Morphology Alignment via Graph-Constrained Latent Modeling
==============================================================

Implements the full framework described in the proposal:

1) Load best 50-gene network per fold (from random2.py outputs)
2) Build fixed gene Laplacian (TRAIN-only)
3) Morphology-only model with:
   - patch encoder
   - optional spatial graph message passing
   - 50 gene heads (pseudo-expression bottleneck)
4) Alignment losses enforced DURING training:
   - classification
   - gene-graph smoothness
   - gene-head disentanglement
5) Cross-validation over provided folds
6) Saves gene–morphology coupling matrices for DM decomposition

Author: Alejandro Leyva
"""

import os
import json
import math
import glob
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import networkx as nx
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.neighbors import NearestNeighbors

# ============================================================
# ---------------------- CONFIG ------------------------------
# ============================================================

FOLDS_DIR = "/fs/scratch/PAS2942/Users/AbdulRehman/ResearchProjects/pancreaticcancer/folds_dir5"
EMB_ROOT  = "/fs/scratch/PAS2942/Users/AbdulRehman/ResearchProjects/pancreaticcancer/pancan_dataset"
GENE_NET_DIR = "/fs/scratch/PAS2942/Alejandro/networks/genetics/random2"

OUTDIR = "./morph_gene_aligned_outputs"
os.makedirs(OUTDIR, exist_ok=True)

NUM_GENES   = 50
PATCH_DIM  = 1536          # UNIv2 patch embeddings
HIDDEN_DIM = 256
KNN_K      = 8

BATCH_SIZE = 1             # slide-level
EPOCHS     = 30
LR         = 1e-4

LAMBDA_GRAPH = 1.0
LAMBDA_DIS   = 0.1

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LABEL_MAP = {"Basal-like": 1, "Classical": 0}

# ============================================================
# ------------------- DATA LOADING ----------------------------
# ============================================================

def load_slide_embedding(slide_filename):
    """
    Memory-safe loader:
    - loads one slide at a time
    - extracts patch embeddings + coordinates
    """
    path = os.path.join(EMB_ROOT, f"{slide_filename}.npy")
    data = np.load(path, allow_pickle=True).item()

    feats = []
    coords = []

    for (x, y), block in data.items():
        feats.append(block["patch_embeddings"])
        coords.append([x, y])

    return (
        torch.tensor(np.vstack(feats), dtype=torch.float32),
        torch.tensor(np.array(coords), dtype=torch.float32),
    )

def compute_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    auc = roc_auc_score(y_true, y_prob)

    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    sensitivity = tp / (tp + fn + 1e-8)   # recall / TPR
    specificity = tn / (tn + fp + 1e-8)   # TNR

    return auc, sensitivity, specificity
def evaluate(model, loader):
    ys, ps = [], []

    model.eval()
    with torch.no_grad():
        for X, coords, y, _ in loader:
            X = X.squeeze(0).to(DEVICE)
            _, y_hat, _ = model(X)
            ys.append(y.item())
            ps.append(y_hat.item())

    return compute_metrics(ys, ps)

def build_knn_edges(coords, k=KNN_K):
    """
    Build sparse adjacency for patch graph.
    """
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(coords.numpy())
    _, idx = nbrs.kneighbors(coords.numpy())

    edges = []
    for i in range(coords.shape[0]):
        for j in idx[i, 1:]:
            edges.append((i, j))
    return edges


def load_gene_laplacian(fold_id):
    genes_path = os.path.join(GENE_NET_DIR, f"fold{fold_id}_best_genes.txt")
    edges_path = os.path.join(GENE_NET_DIR, f"fold{fold_id}_gene_edges.csv")

    genes = [g.strip() for g in open(genes_path)]
    edges = pd.read_csv(edges_path)

    G = nx.Graph()
    G.add_nodes_from(genes)
    for _, r in edges.iterrows():
        G.add_edge(r["gene_u"], r["gene_v"], weight=r["weight"])

    L = nx.laplacian_matrix(G, nodelist=genes).astype(np.float32)  # scipy sparse
    L = L.toarray()  # -> (50,50) dense numpy

    return torch.from_numpy(L), genes



# ============================================================
# --------------------- DATASET -------------------------------
# ============================================================

class SlideDataset(Dataset):
    def __init__(self, csv_path, confidence_mode="high"):
        df = pd.read_csv(csv_path)

        # ---- keep only high confidence rows ----
        if "confidence" not in df.columns:
            raise ValueError(f"[SlideDataset] CSV missing required column 'confidence': {csv_path}")

        conf_series = df["confidence"].astype(str).str.lower()

        if confidence_mode == "high":
            df = df[conf_series == "high"].copy()
        elif confidence_mode == "low":
            df = df[conf_series == "low"].copy()
        elif confidence_mode == "all":
            df = df.copy()
        else:
            raise ValueError(f"Unknown confidence_mode: {confidence_mode}")

        # ---- slides ----
        if "slide_filename" not in df.columns:
            raise ValueError(
                f"[SlideDataset] CSV missing required column 'slide_filename': {csv_path}"
            )

        slides_all = df["slide_filename"].tolist()

        # ---- labels ----
        label_col = "subtype" if "subtype" in df.columns else "moffit_type_norm"

        uniques = set(df[label_col].astype(str).unique().tolist())
        unknown = sorted(list(uniques - set(LABEL_MAP.keys())))
        if unknown:
            raise ValueError(
                f"[SlideDataset] Unknown labels in column '{label_col}': {unknown}. "
                f"Expected one of: {sorted(list(LABEL_MAP.keys()))}"
            )

        labels_all = [LABEL_MAP[str(x)] for x in df[label_col].tolist()]

        # ---- FILTER MISSING EMBEDDINGS ----
        kept_slides = []
        kept_labels = []

        for slide, label in zip(slides_all, labels_all):
            path = os.path.join(EMB_ROOT, f"{slide}.npy")
            if os.path.exists(path):
                kept_slides.append(slide)
                kept_labels.append(label)

        dropped = len(slides_all) - len(kept_slides)
        if dropped > 0:
            print(f"[SlideDataset] Dropped {dropped} slides with missing embeddings")

        self.slides = kept_slides
        self.labels = kept_labels

    def __len__(self):
        return len(self.slides)

    def __getitem__(self, idx):
        X, coords = load_slide_embedding(self.slides[idx])
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return X, coords, y, self.slides[idx]




# ============================================================
# --------------------- MODEL --------------------------------
# ============================================================

class PatchEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(PATCH_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
        )

    def forward(self, x):
        return self.net(x)


class MorphGeneModel(nn.Module):
    """
    Morphology → gene-latent → subtype
    """
    def __init__(self, num_genes):
        super().__init__()

        self.encoder = PatchEncoder()

        self.gene_heads = nn.ModuleList(
            [nn.Linear(HIDDEN_DIM, 1) for _ in range(num_genes)]
        )

        self.classifier = nn.Linear(num_genes, 1)

    def forward(self, X):
        Z = self.encoder(X)                 # [N_patches, H]

        gene_vals = []
        for head in self.gene_heads:
            gene_vals.append(head(Z).mean(dim=0))

        G_hat = torch.cat(gene_vals, dim=0)          # [50]
        y_hat = torch.sigmoid(self.classifier(G_hat.unsqueeze(0))).squeeze(0)  # [1] -> scalar


        return G_hat, y_hat, Z


# ============================================================
# -------------------- LOSSES --------------------------------
# ============================================================

def graph_smoothness_loss(G_hat, L):
    g = G_hat.view(-1, 1)  # [50,1]
    return (g.t() @ L @ g).squeeze()



def disentanglement_loss(G_hat):
    G = F.normalize(G_hat, dim=0)
    return torch.norm(G.unsqueeze(1) @ G.unsqueeze(0) - torch.eye(G.shape[0], device=G.device))


# ============================================================
# -------------------- TRAINING -------------------------------
# ============================================================
for CONF_MODE in ["high", "low"]:
    print("\n" + "="*60)
    print(f"RUNNING CONFIDENCE MODE: {CONF_MODE.upper()}")
    print("="*60)

    for fold in range(5):
        gene_fold = fold + 1
        print(f"\n================= FOLD {fold} =================")

        train_csv = os.path.join(FOLDS_DIR, f"fold{fold}_train.csv")
        val_csv   = os.path.join(FOLDS_DIR, f"fold{fold}_val.csv")

        train_ds = SlideDataset(train_csv, confidence_mode=CONF_MODE)
        val_ds   = SlideDataset(val_csv, confidence_mode=CONF_MODE)
        

        train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
        val_loader   = DataLoader(val_ds, batch_size=1)
        test_csv = os.path.join(FOLDS_DIR, f"fold{fold}_test.csv")
        test_ds  = SlideDataset(test_csv, confidence_mode=CONF_MODE)
        test_loader = DataLoader(test_ds, batch_size=1)


        L_gene, gene_names = load_gene_laplacian(gene_fold)
        L_gene = L_gene.to(DEVICE)

        model = MorphGeneModel(NUM_GENES).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
        best_val_auc = -1.0
        best_state = None
        best_epoch = -1


        for epoch in range(EPOCHS):

            model.train()
            epoch_loss = 0.0

            for X, coords, y, slide_id in train_loader:
                X = X.squeeze(0).to(DEVICE)
                y = y.to(DEVICE)

                G_hat, y_hat, Z = model(X)

                loss_cls = F.binary_cross_entropy(y_hat.view(-1), y.view(-1))
                loss_graph = graph_smoothness_loss(G_hat, L_gene)
                loss_dis   = disentanglement_loss(G_hat)

                loss = (
                    loss_cls +
                    LAMBDA_GRAPH * loss_graph +
                    LAMBDA_DIS   * loss_dis
                )

                opt.zero_grad()
                loss.backward()
                opt.step()

                epoch_loss += loss.item()

            # ---------------- VALIDATION ----------------
            model.eval()
            ys, ps = [], []

            with torch.no_grad():
                for X, coords, y, _ in val_loader:
                    X = X.squeeze(0).to(DEVICE)
                    G_hat, y_hat, _ = model(X)
                    ys.append(y.item())
                    ps.append(y_hat.item())

            auc = roc_auc_score(ys, ps)
            print(f"Fold {fold} | Epoch {epoch:02d} | Val AUC = {auc:.4f}")
            if auc > best_val_auc:
                best_val_auc = auc
                best_epoch = epoch
                best_state = {
                    k: v.cpu().clone() for k, v in model.state_dict().items()
                }
        print(f"Fold {fold} | Best Epoch = {best_epoch} | Best Val AUC = {best_val_auc:.4f}")
        model.load_state_dict(best_state)
        model.to(DEVICE)

        # ================= FINAL EVALUATION =================

        train_auc, train_sens, train_spec = evaluate(model, train_loader)
        val_auc,   val_sens,   val_spec   = evaluate(model, val_loader)
        test_auc,  test_sens,  test_spec  = evaluate(model, test_loader)

        print(
            f"\nFOLD {fold} FINAL METRICS\n"
            f"Train | AUC {train_auc:.3f} | Sens {train_sens:.3f} | Spec {train_spec:.3f}\n"
            f"Val   | AUC {val_auc:.3f} | Sens {val_sens:.3f} | Spec {val_spec:.3f}\n"
            f"Test  | AUC {test_auc:.3f} | Sens {test_sens:.3f} | Spec {test_spec:.3f}"
        )
        # ---------------- SAVE METRICS ----------------
        fold_dir = os.path.join(OUTDIR, f"{CONF_MODE}_confidence", f"fold{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        metrics = {
            "fold": fold,
            "confidence_mode": CONF_MODE,
            "best_epoch": best_epoch,
            "best_val_auc": best_val_auc,

            "train_auc": train_auc,
            "train_sensitivity": train_sens,
            "train_specificity": train_spec,
            "val_auc": val_auc,
            "val_sensitivity": val_sens,
            "val_specificity": val_spec,
            "test_auc": test_auc,
            "test_sensitivity": test_sens,
            "test_specificity": test_spec,
        }


        pd.DataFrame([metrics]).to_csv(
            os.path.join(fold_dir, "metrics.csv"),
            index=False
        )


        # ---------------- SAVE MODEL ----------------
        fold_dir = os.path.join(OUTDIR, f"{CONF_MODE}_confidence", f"fold{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        torch.save(model.state_dict(), os.path.join(fold_dir, "model.pt"))

        # ---------------- SAVE COUPLINGS ----------------
        model.eval()
        couplings = {}

        with torch.no_grad():
            for X, coords, _, slide_id in train_loader:
                X = X.squeeze(0).to(DEVICE)
                Z = model.encoder(X)
                W = torch.stack([head.weight.squeeze() for head in model.gene_heads])
                C = (Z @ W.T).cpu().numpy()   # [patch, gene]
                couplings[slide_id[0]] = C

        np.save(os.path.join(fold_dir, "gene_morph_couplings.npy"), couplings)

    print("\nALL FOLDS COMPLETE.")
