from const import OUTPUT_FOLDER
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

def plot_scale_space_comp(
    A: NDArray[np.float64],
    C: NDArray[np.float64],
    grid_half_extent: int,
    sigma: float,
    tau: float,
    output_path: Path,
) -> None:

    diff = A - C

    # Shared colour scale for A and C so they are visually comparable
    vmin = min(A.min(), C.min())
    vmax = max(A.max(), C.max())

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))

    extent = [-grid_half_extent, grid_half_extent, -grid_half_extent, grid_half_extent]

    im0 = axes[0].imshow(A, cmap="gray", vmin=vmin, vmax=vmax, extent=extent)
    axes[0].set_title(f"A = B(σ={sigma}) * G(τ={tau})")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(C, cmap="gray", vmin=vmin, vmax=vmax, extent=extent)
    axes[1].set_title("C = G(x,y, √(σ²+τ²))")
    axes[1].set_xlabel("x")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(
        diff, cmap="gray", vmin=vmin, vmax=vmax, extent=extent
    )
    axes[2].set_title("Difference A − C")
    axes[2].set_xlabel("x")
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    fig.suptitle(
        "Verification of G(σ) ∗ G(τ) = G(√(σ²+τ²))  [eq. 1]",
        y=1.02,
    )
    for i in range(3):
        # remove x and y axis ticks 
        axes[i].set_xticks([])
        axes[i].set_yticks([])
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()


def plot_tau_range(
    blob: NDArray[np.float64],
    taus: list[float],
    grid_half_width: int,
    scale_space_images: list[NDArray[np.float64]],
    sigma: float = 1.0,
) -> None:
    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(taus) + 1,
        figsize=(3.2 * (len(taus) + 1), 3.8),
    )

    axes[0].imshow(blob, cmap="gray", extent=[-grid_half_width, grid_half_width] * 2)
    axes[0].set_title(f"B(x,y)\nσ = {sigma}")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")

    for ax, tau, img in zip(axes[1:], taus, scale_space_images):
        ax.imshow(img, cmap="gray", extent=[-grid_half_width, grid_half_width] * 2)
        ax.set_title(f"I(x,y,τ={tau})")
        ax.set_xlabel("x")
        ax.axis("on")

    fig.suptitle(
        "Scale-space of blob B(x,y) = G(x,y,σ=1): I(x,y,τ) = B * G(x,y,τ)",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_FOLDER / "4_1_scale_space.png", bbox_inches="tight", dpi=150)
    plt.close()

if __name__ == "__main__":
    def H_analytical(tau: float, sigma: float = 1.0) -> float:
        """Closed-form H(0, 0, τ) for a Gaussian blob B(x,y,σ).

        Derived in 4.3 i/ii:
            H(0, 0, τ) = -τ² / (π (σ² + τ²)²)
        """
        return -(tau**2) / (np.pi * (sigma**2 + tau**2)**2)
    # Plot H(tau) for sigma 1:
    taus = np.linspace(-1, 5, 200)
    H_values = np.array([H_analytical(tau) for tau in taus])
    plt.figure(figsize=(6, 4))
    # Plot: plt.plot(taus, H_values,  label="H(τ) for σ=1")
    # But where values up to tau=0 are a dashed blue line,
    # and the rest of the curve is just a normal blue curve.
    plt.plot(taus[taus <= 0], H_values[taus <= 0], label="H(τ) for σ=1", color='blue', linestyle='--')
    plt.plot(taus[taus > 0], H_values[taus > 0], color='blue')
    plt.title("Scale-normalized Laplacian of H(τ)")
    plt.xlabel("τ")
    plt.ylabel("H(τ)")
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.grid()
    plt.savefig(OUTPUT_FOLDER / "4_3_iii.png", bbox_inches="tight", dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Reality-check plots
# ---------------------------------------------------------------------------

def plot_stack_slices(
    scales: NDArray[np.float64],
    stack: NDArray[np.float64],
    title_prefix: str,
    output_path,
    n_show: int = 6,
) -> None:
    """Visualise n_show evenly spaced slices of the Laplacian stack."""
    import matplotlib.pyplot as plt

    indices = np.linspace(0, len(scales) - 1, n_show, dtype=int)
    fig, axes = plt.subplots(1, n_show, figsize=(3.2 * n_show, 3.5))

    abs_max = np.abs(stack[indices]).max()   # shared colour scale

    for ax, idx in zip(axes, indices):
        im = ax.imshow(stack[idx], cmap="RdBu_r",
                       vmin=-abs_max, vmax=abs_max)
        ax.set_title(f"τ = {scales[idx]:.1f}")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    fig.suptitle(f"{title_prefix}  —  H(x, y, τ) slices", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()


def plot_blobs(
    image: NDArray[np.float64],
    coords: NDArray[np.float64],
    labels: NDArray[np.float64],
    taus: NDArray[np.float64],
    output_path,
) -> None:
    """Overlay detected blobs on the original colour image.

    Each blob is drawn as a dot at (col, row) and a circle of radius τ√2.
    Red = maximum, Blue = minimum.
    """
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(image, cmap="gray")
    ax.set_title(
        "Blob Detection — scale-normalised Laplacian\n"
        "Red = maxima, Blue = minima",
        fontsize=24,
    )
    ax.axis("off")

    color_map = {"max": "red", "min": "blue"}

    for (si, row, col), label, tau in zip(coords, labels, taus):
        c = color_map[label]
        radius = tau * np.sqrt(2)
        ax.plot(col, row, ".", color=c, markersize=4, alpha=0.85)
        circle = plt.Circle(
            (col, row), radius,
            color=c, fill=False, linewidth=1.0, alpha=0.7,
        )
        ax.add_patch(circle)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor="red",  edgecolor="red",  label=f"Maxima  (n={np.sum(labels=='max')})"),
        mpatches.Patch(facecolor="blue", edgecolor="blue", label=f"Minima  (n={np.sum(labels=='min')})"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved → {output_path}")

