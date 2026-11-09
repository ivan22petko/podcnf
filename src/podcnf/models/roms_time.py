"""
Time-dependent extension of PODcnf: generates whole trajectories ("movies").

The CNF is trained on pointwise pairs ([mu, t_k], c(t_k)), so the time t is the
last column of the conditioning vector x = [mu, t]. Sampling each instant with
a fresh latent z gives independent realizations at every instant (a flickering
movie). Here one latent z is drawn per movie and kept fixed for all instants.
"""
from typing import Optional

import torch

from .roms import PODcnf


class PODcnfTime(PODcnf):
    """
    Same constructor as PODcnf: PODcnfTime(pod_matrix, cnf_model, mu_scaler, c_scaler).

    Shapes (n = number of reduced coefficients, N_h = full-order dofs):
        mu : (p,)    physical parameters, WITHOUT time (p = cnf.dim_x - 1)
        t  : (N_t,)  time instants
    """

    @classmethod
    def from_podcnf(cls, rom: PODcnf) -> "PODcnfTime":
        """Wraps an existing PODcnf (e.g. the one returned by PODcnf.load)."""
        return cls(rom.V, rom.cnf, rom.mu_scaler, rom.c_scaler)

    def sample_movie_latent(self, mu, t, n_movies: int = 1, seed: Optional[int] = None) -> torch.Tensor:
        """
        Returns the reduced coefficients c of n_movies movies, shape (n_movies, N_t, n).
        Each movie uses a single latent z, fixed for all the N_t instants.
        """
        dev = next(self.cnf.parameters()).device
        mu = torch.as_tensor(mu, dtype=torch.float32, device=dev).reshape(1, -1)   # (1, p)
        t = torch.as_tensor(t, dtype=torch.float32, device=dev).reshape(-1, 1)     # (N_t, 1)
        N_t = t.shape[0]

        if mu.shape[1] != self.cnf.dim_x - 1:
            raise ValueError(f"mu must have {self.cnf.dim_x - 1} entries (time excluded), got {mu.shape[1]}")

        # Conditioning of ONE movie: (N_t, p+1), same mu on every row, t in the last column
        X_one = self.mu_scaler.transform(torch.cat([mu.repeat(N_t, 1), t], dim=1))
        # All movies stacked: [t1..tN, t1..tN, ...]  -> (n_movies*N_t, p+1)
        X = X_one.repeat(n_movies, 1)

        # One latent per movie, repeated on its N_t rows: [z1..z1, z2..z2, ...] -> (n_movies*N_t, n)
        gen = None if seed is None else torch.Generator(device=dev).manual_seed(seed)
        Z0 = torch.randn(n_movies, self.cnf.dim_y, device=dev, generator=gen)
        Z = Z0.repeat_interleave(N_t, dim=0)

        self.cnf.eval()
        c_tilde = self.cnf.sample(X, z=Z)                      # (n_movies*N_t, n), scaled
        c = self.c_scaler.inverse_transform(c_tilde)           # physical coefficients
        return c.reshape(n_movies, N_t, -1)

    def sample_movie(self, mu, t, n_movies: int = 1, seed: Optional[int] = None) -> torch.Tensor:
        """Full-order movies u = c V^T, shape (n_movies, N_t, N_h)."""
        return self.decode(self.sample_movie_latent(mu, t, n_movies, seed))
