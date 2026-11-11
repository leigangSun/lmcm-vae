"""Synthetic data generator for the latent mediator causal model.

The generative process is a three-layer chain X -> Z -> Y where Z is the
(unobserved) latent mediator. X feeds into Z through the matrix A, and Z
feeds into the observed variables Y through the matrix B. Each observed Y
has a single Z parent (single-parent assumption).
"""

import numpy as np


class LMCMDataGenerator:
    def __init__(self, n_samples, dx, dy, dz,
                 p_edge_xz=0.5,
                 min_children_per_z=3,
                 noise_scale=1.0,
                 seed=None):
        self.n_samples = n_samples
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.p_edge_xz = p_edge_xz
        self.min_children_per_z = min_children_per_z
        self.noise_scale = noise_scale

        if seed is not None:
            np.random.seed(seed)

        assert dy >= dz * min_children_per_z, \
            f"dy ({dy}) must be >= dz * min_children_per_z ({dz * min_children_per_z})"

        self._build_graph()

    def _build_graph(self):
        # A: dx x dz, weights of the X -> Z edges.
        self.A = np.zeros((self.dx, self.dz))
        for k in range(self.dz):
            for i in range(self.dx):
                if np.random.rand() < self.p_edge_xz:
                    self.A[i, k] = np.random.uniform(0.3, 1.0) * np.random.choice([-1, 1])

        # B: dz x dy, weights of the Z -> Y edges.
        # Every Y has exactly one Z parent; each Z gets at least
        # min_children_per_z Y children, the rest are assigned at random.
        self.B = np.zeros((self.dz, self.dy))
        self.cluster_labels = np.zeros(self.dy, dtype=int)

        y_assigned = 0
        for k in range(self.dz):
            for _ in range(self.min_children_per_z):
                j = y_assigned
                self.B[k, j] = np.random.uniform(0.3, 1.0) * np.random.choice([-1, 1])
                self.cluster_labels[j] = k
                y_assigned += 1

        for j in range(y_assigned, self.dy):
            k = np.random.randint(0, self.dz)
            self.B[k, j] = np.random.uniform(0.3, 1.0) * np.random.choice([-1, 1])
            self.cluster_labels[j] = k

        self.X_cov = np.eye(self.dx) * 0.5 + 0.5 * np.ones((self.dx, self.dx))
        self.X_mu = np.zeros(self.dx)

    def generate(self):
        X = np.random.multivariate_normal(self.X_mu, self.X_cov, self.n_samples)

        noise_Z = np.random.normal(0, self.noise_scale, (self.n_samples, self.dz))
        Z = X @ self.A + noise_Z

        noise_Y = np.random.normal(0, self.noise_scale, (self.n_samples, self.dy))
        Y = Z @ self.B + noise_Y

        return X, Y, Z

    def get_ground_truth(self):
        return {
            'A': self.A.copy(),
            'B': self.B.copy(),
            'cluster_labels': self.cluster_labels.copy(),
            'dz': self.dz,
            'dx': self.dx,
            'dy': self.dy,
        }


def generate_lmcm_data(n_samples, dx, dy, dz, seed=0, **kwargs):
    """Generate one LMCM dataset and return (X, Y, Z, ground_truth)."""
    dg = LMCMDataGenerator(n_samples, dx, dy, dz, seed=seed, **kwargs)
    X, Y, Z = dg.generate()
    gt = dg.get_ground_truth()
    return X, Y, Z, gt
