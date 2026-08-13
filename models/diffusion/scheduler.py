"""
signrl_diff.models.diffusion.scheduler
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Standard Denoising Diffusion Probabilistic Model (DDPM) noise scheduler.

Reference
---------
Ho et al., "Denoising Diffusion Probabilistic Models", NeurIPS 2020.

The scheduler manages the noise schedule (betas), derives all required
quantities (alphas, alpha-bars, sigmas), and provides the two core
operations needed during training and inference:

* :py:meth:`add_noise`  — sample ``x_t`` given ``x_0`` and timestep ``t``
* :py:meth:`step`       — one reverse diffusion step ``x_t → x_{t-1}``
"""

from __future__ import annotations

from typing import Tuple, Union

import torch
import torch.nn as nn


class DDPMScheduler(nn.Module):
    """DDPM noise scheduler with linear or cosine beta schedule.

    All schedule tensors are registered as **buffers** so they are
    automatically moved to the correct device alongside the model and
    are included in ``state_dict`` without being treated as trainable
    parameters.

    Parameters
    ----------
    num_train_steps : int, default 1000
        Total number of diffusion steps *T*.
    beta_schedule : {"linear", "cosine"}, default "linear"
        Strategy used to construct the beta schedule.
    beta_start : float, default 1e-4
        Starting value of beta (only used for ``"linear"`` schedule).
    beta_end : float, default 0.02
        Ending value of beta (only used for ``"linear"`` schedule).
    """

    def __init__(
        self,
        num_train_steps: int = 1000,
        beta_schedule: str = "linear",
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ) -> None:
        super().__init__()

        self.num_train_steps: int = num_train_steps
        self.beta_schedule: str = beta_schedule

        # ------------------------------------------------------------------
        # Build beta schedule
        # ------------------------------------------------------------------
        betas: torch.Tensor = self._make_betas(
            num_train_steps, beta_schedule, beta_start, beta_end
        )

        # Derived quantities
        alphas: torch.Tensor = 1.0 - betas
        alphas_cumprod: torch.Tensor = torch.cumprod(alphas, dim=0)

        # For the reverse-step formula we also need alphas_cumprod shifted
        # by one position (alpha_bar_{t-1}).  We prepend 1.0 so that
        # ``alphas_cumprod_prev[0] = 1.0`` (corresponding to t=0).
        alphas_cumprod_prev: torch.Tensor = torch.cat(
            [torch.tensor([1.0]), alphas_cumprod[:-1]]
        )

        # Posterior variance: beta_tilde_t = beta_t * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)
        posterior_variance: torch.Tensor = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )

        # ------------------------------------------------------------------
        # Register everything as buffers (non-trainable, device-aware)
        # ------------------------------------------------------------------
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.sqrt(1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "sqrt_recip_alphas", torch.sqrt(1.0 / alphas)
        )
        self.register_buffer("posterior_variance", posterior_variance)
        # Clamp the first element to avoid log(0) / div-by-zero
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(
                torch.cat(
                    [posterior_variance[1:2], posterior_variance[1:]]
                ).clamp(min=1e-20)
            ),
        )
        # Posterior mean coefficients
        self.register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev)
            * torch.sqrt(alphas)
            / (1.0 - alphas_cumprod),
        )

    # ------------------------------------------------------------------
    # Beta schedule constructors
    # ------------------------------------------------------------------

    @staticmethod
    def _make_betas(
        num_steps: int,
        schedule: str,
        beta_start: float,
        beta_end: float,
    ) -> torch.Tensor:
        """Build the beta schedule tensor.

        Parameters
        ----------
        num_steps : int
            Number of diffusion steps.
        schedule : str
            ``"linear"`` or ``"cosine"``.
        beta_start, beta_end : float
            Endpoints for the linear schedule.

        Returns
        -------
        torch.Tensor
            1-D tensor of shape ``(num_steps,)`` with dtype ``float64``
            (cast to ``float32`` later for numerical safety).
        """
        if schedule == "linear":
            betas = torch.linspace(
                beta_start, beta_end, num_steps, dtype=torch.float64
            )
        elif schedule == "cosine":
            # Nichol & Dhariwal, "Improved DDPM", 2021
            steps = torch.arange(num_steps + 1, dtype=torch.float64)
            s = 0.008
            alpha_bar = torch.cos(
                ((steps / num_steps) + s) / (1.0 + s) * (torch.pi / 2.0)
            ) ** 2
            betas = 1.0 - (alpha_bar[1:] / alpha_bar[:-1])
            betas = betas.clamp(max=0.999)
        else:
            raise ValueError(
                f"Unknown beta_schedule '{schedule}'. "
                "Supported: 'linear', 'cosine'."
            )
        return betas.float()

    # ------------------------------------------------------------------
    # Indexing helper
    # ------------------------------------------------------------------

    @staticmethod
    def _extract(
        tensor: torch.Tensor, t: torch.Tensor, shape: torch.Size
    ) -> torch.Tensor:
        """Gather values from *tensor* at indices *t* and reshape for
        broadcasting with a batch tensor of the given *shape*.

        Parameters
        ----------
        tensor : torch.Tensor
            1-D buffer of length ``T``.
        t : torch.Tensor
            1-D integer tensor of length ``B``.
        shape : torch.Size
            Target shape (typically ``x_0.shape``).

        Returns
        -------
        torch.Tensor
            Reshaped tensor broadcastable to *shape*.
        """
        gathered = tensor.gather(0, t.long().to(tensor.device))
        # Reshape to (B, 1, 1, ...) so it broadcasts over channel / spatial
        return gathered.reshape(-1, *([1] * (len(shape) - 1)))

    # ------------------------------------------------------------------
    # Forward process: q(x_t | x_0)
    # ------------------------------------------------------------------

    def add_noise(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample ``x_t`` from the forward diffusion process.

        Implements:

            ``x_t = sqrt(ᾱ_t) · x_0 + sqrt(1 - ᾱ_t) · ε``

        Parameters
        ----------
        x_0 : torch.Tensor
            Clean data, arbitrary shape ``(B, ...)``.
        t : torch.Tensor
            Integer timestep tensor of shape ``(B,)`` with values in
            ``[0, T-1]``.
        noise : torch.Tensor, optional
            Pre-sampled Gaussian noise with the same shape as *x_0*.
            If ``None``, fresh noise is drawn.

        Returns
        -------
        x_t : torch.Tensor
            Noisy sample at timestep *t*.
        noise : torch.Tensor
            The Gaussian noise that was added.
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha_bar = self._extract(
            self.sqrt_alphas_cumprod, t, x_0.shape  # type: ignore[arg-type]
        )
        sqrt_one_minus_alpha_bar = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_0.shape  # type: ignore[arg-type]
        )

        x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise
        return x_t, noise

    # ------------------------------------------------------------------
    # Reverse process: p(x_{t-1} | x_t)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def step(
        self,
        model_output: torch.Tensor,
        t: int,
        x_t: torch.Tensor,
    ) -> torch.Tensor:
        """Perform one reverse diffusion step.

        Implements the DDPM reverse step:

            ``x_{t-1} = (1/√α_t)(x_t - (1-α_t)/√(1-ᾱ_t) · ε̂) + σ_t · ε``

        When ``t == 0`` no noise is added (σ_0 = 0).

        Parameters
        ----------
        model_output : torch.Tensor
            Predicted noise ``ε̂`` from the denoising network, same shape
            as *x_t*.
        t : int
            Current timestep (scalar integer, ``0 ≤ t < T``).
        x_t : torch.Tensor
            Current noisy sample.

        Returns
        -------
        torch.Tensor
            Denoised sample ``x_{t-1}``.
        """
        # Build a batch index tensor — every element in the batch shares
        # the same timestep ``t``.
        batch_size = x_t.shape[0]
        t_tensor = torch.full(
            (batch_size,), t, device=x_t.device, dtype=torch.long
        )

        alpha_t = self._extract(self.alphas, t_tensor, x_t.shape)  # type: ignore[arg-type]
        alpha_bar_t = self._extract(self.alphas_cumprod, t_tensor, x_t.shape)  # type: ignore[arg-type]
        sqrt_recip_alpha_t = self._extract(
            self.sqrt_recip_alphas, t_tensor, x_t.shape  # type: ignore[arg-type]
        )

        # Predicted x_0 from the epsilon prediction
        # x_0_pred = (x_t - sqrt(1 - alpha_bar_t) * eps) / sqrt(alpha_bar_t)
        sqrt_one_minus_alpha_bar_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t_tensor, x_t.shape  # type: ignore[arg-type]
        )
        sqrt_alpha_bar_t = self._extract(
            self.sqrt_alphas_cumprod, t_tensor, x_t.shape  # type: ignore[arg-type]
        )

        # Mean: mu_tilde(x_t, x_0)
        # = (1 / sqrt(alpha_t)) * (x_t - (1 - alpha_t) / sqrt(1 - alpha_bar_t) * eps_hat)
        coef = (1.0 - alpha_t) / sqrt_one_minus_alpha_bar_t
        mean = sqrt_recip_alpha_t * (x_t - coef * model_output)

        if t == 0:
            # No noise at the final step
            return mean

        # σ_t = sqrt(beta_t)  where beta_t = 1 - alpha_t
        # We use the posterior standard deviation for better quality:
        # sigma_t = sqrt(posterior_variance_t)
        posterior_var = self._extract(
            self.posterior_variance, t_tensor, x_t.shape  # type: ignore[arg-type]
        )
        sigma_t = torch.sqrt(posterior_var)

        noise = torch.randn_like(x_t)
        return mean + sigma_t * noise

    # ------------------------------------------------------------------
    # Full sampling loop (convenience)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape: torch.Size,
        device: torch.device,
        text_condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the full reverse diffusion loop from pure noise to data.

        Parameters
        ----------
        model : nn.Module
            Denoising network.  Called as ``model(x_t, t_tensor, text_condition)``.
        shape : torch.Size
            Desired output shape, e.g. ``(B, T, C, H, W)``.
        device : torch.device
            Device on which to allocate tensors.
        text_condition : torch.Tensor, optional
            Text embedding tensor forwarded to the model.

        Returns
        -------
        torch.Tensor
            Fully denoised sample ``x_0``.
        """
        x = torch.randn(shape, device=device)
        for t in reversed(range(self.num_train_steps)):
            t_batch = torch.full(
                (shape[0],), t, device=device, dtype=torch.long
            )
            eps_hat = model(x, t_batch, text_condition)
            x = self.step(eps_hat, t, x)
        return x

    # ------------------------------------------------------------------
    # SNR helpers (useful for Min-SNR weighting in training)
    # ------------------------------------------------------------------

    def get_snr(self, t: torch.Tensor) -> torch.Tensor:
        """Return the signal-to-noise ratio at timestep(s) *t*.

        ``SNR(t) = ᾱ_t / (1 - ᾱ_t)``

        Parameters
        ----------
        t : torch.Tensor
            1-D integer tensor of timesteps.

        Returns
        -------
        torch.Tensor
            SNR values, same shape as *t*.
        """
        alpha_bar = self.alphas_cumprod.gather(0, t.long().to(self.alphas_cumprod.device))  # type: ignore[arg-type]
        return alpha_bar / (1.0 - alpha_bar)
