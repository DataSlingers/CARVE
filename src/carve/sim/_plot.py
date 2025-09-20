import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def _plot_simulation(X: np.ndarray, y: np.ndarray, *, random_state: int | None):
    pca = PCA(n_components=2, random_state=random_state)
    X_pca = pca.fit_transform(X)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10', edgecolors='k', alpha=0.8)
    plt.title("PCA projection of simulated clusters")
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.legend(*scatter.legend_elements(), title="Clusters", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout(); plt.show()