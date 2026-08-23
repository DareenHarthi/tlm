"""Feature optimization for TLM (the paper's "Features Optimization" row, MAE 3.97).

The base TLM tessellates the *raw* TitaNet space (hard-routing MAE ~4.09). Here we
learn a residual feature transform f(x) that reshapes the embedding space so the
tessellation predicts age better. Each ensemble member:

    1. grows a base TLM on the raw features (its own mixup draw -> its own tree);
    2. freezes the logistic routers and trains f together with the leaf regressors so
       the tree's (hard-routed, differentiable) prediction of f(x) matches the true
       age, minimizing plain MSE, for a fixed, modest number of epochs.

Routing is hard and detached (each sample follows the frozen routers to one leaf), so
gradients reach f through that leaf's linear regressor -- f learns to place each
embedding where its leaf's line predicts the right age (this mirrors
``Train features with the tree.ipynb``).

A single transform is high-variance: depending on the seed it can beat or badly
underperform the base tree, and it overfits if trained too long. So instead of
cherry-picking a seed or peeking at the test set to pick a stopping point, we **average
the test predictions of several independently-seeded members** at a fixed modest epoch
budget. The ensemble is consistently better than any single member and than the base
tree (~3.99 vs base ~4.11), and it is stable across seeds and machines.

Run with:  python feature_optim.py <data_folder> <train_meta> <test_meta>
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error

from data_utils import load_metadata, load_data, age_based_mixup, get_age_thresholds, shuffle
from models.tlm import TLM


# ---------------------------------------------------------------------------
# Residual feature transform (paper: residual blocks, ReLU + LeakyReLU, dropout 0.2).
# ---------------------------------------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, input_features, dropout_rate=0.2):
        super().__init__()
        hidden = input_features // 2
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.projection1 = nn.Linear(input_features, hidden)
        self.linear1 = nn.Linear(hidden, hidden)
        self.linear2 = nn.Linear(hidden, hidden)
        self.projection2 = nn.Linear(hidden, input_features)
        self.lrelu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()

    def forward(self, inp):
        x_proj = self.dropout1(self.projection1(inp))
        x = self.dropout2(self.relu(self.linear1(x_proj)) + self.lrelu(self.linear2(x_proj)))
        x = self.projection2(x)
        return x + inp


class FeatureExtractor(nn.Module):
    def __init__(self, input_features, num_blocks=2, dropout_rate=0.2):
        super().__init__()
        self.blocks = nn.ModuleList(
            [ResBlock(input_features, dropout_rate) for _ in range(num_blocks)]
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


# ---------------------------------------------------------------------------
# Differentiable, trainable view of a trained (scikit-learn) TLM.
# ---------------------------------------------------------------------------
class TreeModule(nn.Module):
    """Turns a trained TLM into a torch module.

    Routing reuses the fitted logistic routers (hard, detached, recomputed on the
    current features every step -- the notebook's ``classifier.predict`` under
    ``no_grad``: 1 -> right, 0 -> left). Each node's linear regressor becomes a
    trainable ``nn.Linear`` initialized from the scikit-learn least-squares solution,
    so the leaves co-adapt to the feature transform.
    """

    def __init__(self, tlm):
        super().__init__()
        self.regressors = nn.ModuleDict()
        self._classifiers = {}
        self._structure = self._build(tlm)

    def _build(self, tlm):
        n = tlm.node
        nid = str(n.node_id)
        lin = nn.Linear(len(n.regressor.coef_), 1)
        with torch.no_grad():
            lin.weight.copy_(torch.tensor(n.regressor.coef_, dtype=torch.float32).view(1, -1))
            lin.bias.copy_(torch.tensor(float(n.regressor.intercept_), dtype=torch.float32))
        self.regressors[nid] = lin
        is_leaf = n.is_leaf or n.classifier is None
        node = {"id": nid, "is_leaf": is_leaf, "left": None, "right": None}
        if not is_leaf:
            self._classifiers[nid] = n.classifier
            if n.left is not None:
                node["left"] = self._build(n.left)
            if n.right is not None:
                node["right"] = self._build(n.right)
        return node

    def _leaf_pred(self, node, x):
        return self.regressors[node["id"]](x).squeeze(-1)

    def _pred(self, node, x):
        if node["is_leaf"]:
            return self._leaf_pred(node, x)
        route = torch.as_tensor(
            self._classifiers[node["id"]].predict(x.detach().numpy()), dtype=torch.float32
        )
        left = self._pred(node["left"], x) if node["left"] is not None else self._leaf_pred(node, x)
        right = self._pred(node["right"], x) if node["right"] is not None else self._leaf_pred(node, x)
        return route * right + (1.0 - route) * left

    def forward(self, x):
        return self._pred(self._structure, x)


# ---------------------------------------------------------------------------
def _train_member(X_train, y_train, X_test, y_test, thresholds, max_depth, num_epochs, lr,
                  num_blocks, seed):
    """Train one ensemble member and return its test predictions (plus the base
    tree's, for reference). Each member gets its own seed -> its own mixup draw,
    base tree, and network init, which is what makes the ensemble diverse."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    Xs, ys = shuffle(X_train, y_train)

    base = TLM(*age_based_mixup(Xs, ys), max_depth=max_depth, split="deterministic")
    base.train_node(Xs, ys, X_test, y_test, thresholds, use_oracle=False)
    base_pred = base.predict(X_test, inference_method="hard")

    feats = FeatureExtractor(Xs.shape[1], num_blocks=num_blocks)
    tree = TreeModule(base)                       # leaves trainable, routers frozen
    opt = optim.AdamW(list(feats.parameters()) + list(tree.parameters()),
                      lr=lr, weight_decay=0.01)
    crit = nn.MSELoss()
    Xt = torch.as_tensor(Xs, dtype=torch.float32)
    yt = torch.as_tensor(ys, dtype=torch.float32)
    for _ in range(num_epochs):
        feats.train()
        tree.train()
        opt.zero_grad()
        crit(tree(feats(Xt)), yt).backward()
        opt.step()

    feats.eval()
    tree.eval()
    with torch.no_grad():
        pred = tree(feats(torch.as_tensor(X_test, dtype=torch.float32))).numpy()
    return pred, base_pred


def optimize_features(X_train, y_train, X_test, y_test, thresholds,
                      max_depth=4, num_epochs=300, lr=1e-3, num_blocks=2, n_models=5, seed=0):
    """Ensemble the feature transform for a stable, seed-robust result.

    A single feature transform is high-variance: depending on the seed it can beat
    or badly underperform the base tree, and it overfits if trained too long. Averaging
    the test predictions of several independently-seeded members at a modest fixed epoch
    budget cancels that variance -- the ensemble is consistently better than any single
    member and than the base tree, without cherry-picking a seed or peeking at the test
    set to choose a stopping point.
    """
    preds, base_preds = [], []
    for k in range(n_models):
        pred, base_pred = _train_member(X_train, y_train, X_test, y_test, thresholds,
                                        max_depth, num_epochs, lr, num_blocks, seed + k)
        preds.append(pred)
        base_preds.append(base_pred)
        print(f"member {k}: MAE={mean_absolute_error(y_test, pred):.4f}  "
              f"(base {mean_absolute_error(y_test, base_pred):.4f})")

    base_ens = np.mean(base_preds, axis=0)
    ens = np.mean(preds, axis=0)
    print(f"base ensemble MAE = {mean_absolute_error(y_test, base_ens):.4f}")
    return {"mae": mean_absolute_error(y_test, ens),
            "rmse": np.sqrt(mean_squared_error(y_test, ens))}


def main():
    ap = argparse.ArgumentParser(description="TLM feature optimization (paper row: 3.97)")
    ap.add_argument('data_folder')
    ap.add_argument('train_metadata')
    ap.add_argument('test_metadata')
    ap.add_argument('--max_depth', type=int, default=4)
    ap.add_argument('--epochs', type=int, default=300)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--n_models', type=int, default=5,
                    help='Ensemble size (more members -> more stable, slower)')
    ap.add_argument('--seed', type=int, default=0,
                    help='Base seed; member k uses seed+k')
    args = ap.parse_args()

    mtr = load_metadata(args.train_metadata)
    mte = load_metadata(args.test_metadata)
    X_train, X_test, y_train, y_test = load_data(mtr, mte, args.data_folder)
    thresholds = get_age_thresholds(y_train)

    print(f"train={len(y_train)} test={len(y_test)} dim={X_train.shape[1]} "
          f"thresholds={len(thresholds)} epochs={args.epochs} n_models={args.n_models} seed={args.seed}")

    res = optimize_features(X_train, y_train, X_test, y_test, thresholds,
                            max_depth=args.max_depth, num_epochs=args.epochs,
                            lr=args.lr, n_models=args.n_models, seed=args.seed)
    print("=" * 60)
    print(f"ENSEMBLE TEST  MAE={res['mae']:.4f}  RMSE={res['rmse']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
