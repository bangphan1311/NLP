import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from eval_words import MIN_FREQ, RELATED_PAIRS
from evaluate_compare import MODELS, ROOT, find_input_embedding, load_vocab

FIG_DIR = ROOT / "results/figures"
TSNE_MODELS = {"main_ns_30ep"}
TSNE_TOP_WORDS = 3000
SEED = 42

PAIR_WORDS = [w for pair in RELATED_PAIRS for w in pair]


def pca_fit(X):
    mean = X.mean(dim=0, keepdim=True)
    _, S, Vh = torch.linalg.svd(X - mean, full_matrices=False)
    ratio = (S ** 2) / (S ** 2).sum()
    return mean, Vh[:2], ratio[:2]


def pca_project(X, mean, components):
    return (X - mean) @ components.T


def draw_pairs(ax, coords, title, background=None):
    cmap = plt.get_cmap("tab10")
    if background is not None:
        ax.scatter(background[:, 0], background[:, 1], s=4, color="lightgray", alpha=0.5, zorder=1)
    for i, (a, b) in enumerate(RELATED_PAIRS):
        color = cmap(i)
        (xa, ya), (xb, yb) = coords[a], coords[b]
        ax.plot([xa, xb], [ya, yb], color=color, linewidth=1.3, alpha=0.8, zorder=2)
        for word, x, y in ((a, xa, ya), (b, xb, yb)):
            ax.scatter([x], [y], color=color, s=36, zorder=3)
            ax.annotate(word, (x, y), textcoords="offset points", xytext=(4, 4),
                        fontsize=8, color=color, zorder=4)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.2)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    vocab, freq = load_vocab()
    frequent = torch.nonzero(freq >= MIN_FREQ).squeeze(1)
    pair_ids = torch.tensor([vocab[w] for w in PAIR_WORDS])

    for name, ckpt_path, _, _ in MODELS:
        if not ckpt_path.exists():
            print(f"[BỎ QUA] {name}: chưa có {ckpt_path.name}")
            continue

        print(f"[{name}]")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        M = find_input_embedding(ckpt["model_state_dict"], len(vocab))

        mean_g, comp_g, ratio_g = pca_fit(M[frequent])
        background = pca_project(M[frequent], mean_g, comp_g)
        coords_g = dict(zip(PAIR_WORDS, pca_project(M[pair_ids], mean_g, comp_g).tolist()))

        mean_l, comp_l, ratio_l = pca_fit(M[pair_ids])
        coords_l = dict(zip(PAIR_WORDS, pca_project(M[pair_ids], mean_l, comp_l).tolist()))

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        draw_pairs(
            axes[0], coords_g,
            f"{name}: PCA fit trên các từ tần suất >= {MIN_FREQ} "
            f"(PC1 {ratio_g[0]:.1%}, PC2 {ratio_g[1]:.1%})",
            background,
        )
        draw_pairs(
            axes[1], coords_l,
            f"{name}: PCA fit chỉ trên 20 từ của 10 cặp "
            f"(PC1 {ratio_l[0]:.1%}, PC2 {ratio_l[1]:.1%})",
        )
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"pca_{name}.png", dpi=150)
        plt.close(fig)
        print(f"   đã lưu pca_{name}.png")

        if name in TSNE_MODELS:
            try:
                from sklearn.manifold import TSNE
            except ImportError:
                print("   thiếu scikit-learn: chạy 'uv add scikit-learn' rồi chạy lại để có t-SNE")
                continue

            top = frequent[torch.argsort(freq[frequent], descending=True)[:TSNE_TOP_WORDS]]
            ids = torch.unique(torch.cat([top, pair_ids]))
            X = F.normalize(M[ids], dim=1).numpy()
            emb = TSNE(
                n_components=2, perplexity=30, init="pca",
                learning_rate="auto", random_state=SEED,
            ).fit_transform(X)
            index = {int(i): k for k, i in enumerate(ids)}
            coords_t = {w: emb[index[vocab[w]]].tolist() for w in PAIR_WORDS}

            fig, ax = plt.subplots(figsize=(10, 8))
            draw_pairs(
                ax, coords_t,
                f"{name}: t-SNE (cosine, top {TSNE_TOP_WORDS} từ + 20 từ của 10 cặp, perplexity 30)",
                emb,
            )
            fig.tight_layout()
            fig.savefig(FIG_DIR / f"tsne_{name}.png", dpi=150)
            plt.close(fig)
            print(f"   đã lưu tsne_{name}.png")


if __name__ == "__main__":
    main()
