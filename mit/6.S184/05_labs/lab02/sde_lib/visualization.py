"""
Visualization Utilities for SDE Simulations

This module provides plotting functions for visualizing:
- 1D trajectories over time
- 2D probability densities (heatmaps, contours)
- Distribution evolution (snapshots and animations)
- Particle dynamics with KDE estimation

These tools are essential for understanding and debugging SDE simulations,
as well as creating publication-quality figures.

Dependencies:
    - matplotlib: Core plotting library
    - seaborn: Statistical visualization (KDE)
    - celluloid: Animation creation
    - torch: Tensor operations

Author: MIT 6.S184 Course Staff
"""

from einops import rearrange
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.axes import Axes
from typing import Optional, Callable, Tuple
import seaborn as sns

from .base import Simulator, Sampleable, Density


def plot_trajectories_1d(
    x0: torch.Tensor,
    simulator: Simulator,
    ts: torch.Tensor,
    ax: Optional[Axes] = None
) -> None:
    """
    Plot 1D trajectories of SDE/ODE evolution over time.
    
    Visualizes how individual particles move through one-dimensional space
    as a function of time. Each trajectory is shown as a separate line.
    
    This is particularly useful for:
    - Brownian motion: Showing random walks
    - Ornstein-Uhlenbeck: Demonstrating mean reversion
    - Understanding variance growth or decay
    
    Args:
        x0 (torch.Tensor): Initial positions at time ts[0].
            Shape: (num_trajectories, 1)
            Each row is a starting position for one trajectory
        
        simulator (Simulator): Numerical simulator to use.
            Must implement simulate_with_trajectory method
        
        ts (torch.Tensor): Time discretization for simulation.
            Shape: (num_timesteps,)
            Sorted sequence from start time to end time
        
        ax (Optional[Axes]): Matplotlib axes to plot on.
            If None, uses current axes (plt.gca())
    
    Returns:
        None. Modifies the provided axes in-place.
    
    Implementation Details:
        - Simulates full trajectory: O(num_trajectories × num_timesteps)
        - Each trajectory plotted independently
        - Uses default matplotlib color cycle
        - Memory: Stores full (num_trajectories × num_timesteps × 1) tensor
    
    Visualization Tips:
        - Use 5-20 trajectories for clarity (too many → cluttered)
        - Add grid for easier reading: ax.grid(True, alpha=0.3)
        - Label axes appropriately: ax.set_xlabel('Time'), ax.set_ylabel('Position')
        - Add reference lines: ax.axhline(0, color='k', linestyle='--')
    
    Example - Brownian Motion:
        >>> # Show random walks
        >>> bm = BrownianMotion(sigma=1.0)
        >>> sim = EulerMaruyamaSimulator(bm)
        >>> x0 = torch.zeros(10, 1)  # 10 particles at origin
        >>> ts = torch.linspace(0, 5, 500)
        >>> 
        >>> fig, ax = plt.subplots(figsize=(10, 6))
        >>> plot_trajectories_1d(x0, sim, ts, ax)
        >>> ax.set_title('Brownian Motion Trajectories')
        >>> ax.set_xlabel('Time')
        >>> ax.set_ylabel('Position $X_t$')
        >>> ax.grid(True, alpha=0.3)
        >>> plt.show()
    
    Example - OU Process:
        >>> # Show mean reversion from different starts
        >>> ou = OUProcess(theta=0.5, sigma=1.0)
        >>> sim = EulerMaruyamaSimulator(ou)
        >>> x0 = torch.linspace(-10, 10, 15).view(-1, 1)
        >>> ts = torch.linspace(0, 20, 1000)
        >>> 
        >>> fig, ax = plt.subplots()
        >>> plot_trajectories_1d(x0, sim, ts, ax)
        >>> ax.axhline(0, color='r', linestyle='--', label='Equilibrium')
        >>> # Add theoretical variance bounds
        >>> var_inf = 1.0  # σ²/(2θ) = 1/(2*0.5)
        >>> ax.axhline(np.sqrt(var_inf), color='g', linestyle=':', alpha=0.5)
        >>> ax.axhline(-np.sqrt(var_inf), color='g', linestyle=':', alpha=0.5)
        >>> ax.legend()
        >>> plt.show()
    
    See Also:
        - plot_distribution_evolution: For 2D spatial evolution
        - Simulator.simulate_with_trajectory: Underlying simulation method
    """
    if ax is None:
        ax = plt.gca()
    
    # Simulate full trajectories: (num_trajectories, num_timesteps, 1)
    trajectories = simulator.simulate_with_trajectory(x0, ts)
    
    # Plot each trajectory as a line
    for trajectory_idx in range(trajectories.shape[0]):
        trajectory = trajectories[trajectory_idx, :, 0]  # Extract 1D position
        ax.plot(ts.cpu().numpy(), trajectory.cpu().numpy())


def hist2d_samples(
    samples: np.ndarray,
    ax: Optional[Axes] = None,
    bins: int = 200,
    scale: float = 5.0,
    percentile: int = 99,
    **kwargs
) -> None:
    """Plot 2D histogram of samples."""
    if ax is None:
        ax = plt.gca()
    
    H, xedges, yedges = np.histogram2d(
        samples[:, 0], samples[:, 1],
        bins=bins, range=[[-scale, scale], [-scale, scale]]
    )
    
    # Determine color normalization based on percentile
    cmax = np.percentile(H, percentile)
    cmin = 0.0
    norm = cm.colors.Normalize(vmax=cmax, vmin=cmin)
    
    # Plot using imshow
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    ax.imshow(H.T, extent=extent, origin='lower', norm=norm, **kwargs)


def hist2d_sampleable(
    sampleable: Sampleable,
    num_samples: int,
    ax: Optional[Axes] = None,
    bins: int = 200,
    scale: float = 5.0,
    percentile: int = 99,
    **kwargs
) -> None:
    """Sample from distribution and plot 2D histogram."""
    assert sampleable.dim == 2, "Only 2D distributions supported"
    samples = sampleable.sample(num_samples).detach().cpu().numpy()
    hist2d_samples(samples, ax, bins, scale, percentile, **kwargs)


def scatter_sampleable(
    sampleable: Sampleable,
    num_samples: int,
    ax: Optional[Axes] = None,
    **kwargs
) -> None:
    """Scatter plot of samples from distribution."""
    assert sampleable.dim == 2
    if ax is None:
        ax = plt.gca()
    samples = sampleable.sample(num_samples).detach().cpu().numpy()
    ax.scatter(samples[:, 0], samples[:, 1], **kwargs)


def kdeplot_sampleable(sampleable: Sampleable, num_samples: int, ax: Optional[Axes] = None, **kwargs):
    assert sampleable.dim == 2
    if ax is None:
        ax = plt.gca()
    samples = sampleable.sample(num_samples).detach().cpu().numpy()
    sns.kdeplot(x=samples[:, 0], y=samples[:, 1], ax=ax, **kwargs)


def imshow_density(
    density: Density,
    bins: int,
    scale: Optional[float] = None,
    device: str = 'cpu',
    x_bounds: Optional[list] = None,
    y_bounds: Optional[list] = None,
    ax: Optional[Axes] = None,
    x_offset: float = 0.0,
    **kwargs
) -> None:
    """
    Display 2D probability density as a heatmap.
    
    Creates a visual representation of the log density log p(x) over a
    spatial grid. Darker colors indicate higher probability regions.
    
    Args:
        density (Density): Probability distribution to visualize.
            Must implement log_density(x) method
        
        bins (int): Number of grid points per dimension.
            Total evaluations: bins²
            Typical values: 50-200 (higher = smoother but slower)
        
        scale (float, optional): Spatial extent [-scale, scale].
        
        x_bounds (list, optional): [xmin, xmax].
        y_bounds (list, optional): [ymin, ymax].
        
        device (str, optional): Device for computation ('cpu' or 'cuda').
            Default: 'cpu'
        
        ax (Optional[Axes]): Matplotlib axes to plot on.
            If None, uses current axes
            
        x_offset (float): Horizontal shift for the evaluation window.
        
        **kwargs: Additional arguments passed to ax.imshow():
            - vmin, vmax: Color scale limits
            - cmap: Colormap (e.g., 'Blues', 'viridis')
            - alpha: Transparency (0=invisible, 1=opaque)
            - interpolation: Smoothing method
    
    Returns:
        None. Modifies axes in-place.
    
    Computational Cost:
        - Grid creation: O(bins²)
        - Density evaluation: Depends on complexity of log_density
        - For Gaussians: Fast
        - For mixtures: O(bins² × num_modes)
    
    Visualization Details:
        - Uses 'lower' origin (bottom-left is minimum)
        - Extent set to [-scale, scale, -scale, scale]
        - Log density shown (NOT raw probability)
        - Typical range: -20 to 0 (log scale)
    
    Example - Single Gaussian:
        >>> # Visualize 2D Gaussian
        >>> gaussian = Gaussian(
        ...     mean=torch.zeros(2),
        ...     cov=torch.eye(2)
        ... )
        >>> 
        >>> fig, ax = plt.subplots(figsize=(8, 8))
        >>> imshow_density(
        ...     gaussian,
        ...     bins=100,
        ...     scale=5,
        ...     ax=ax,
        ...     cmap='Blues',
        ...     vmin=-10
        ... )
        >>> ax.set_title('2D Gaussian Density')
        >>> plt.colorbar(ax.images[0], label='log p(x)')
        >>> plt.show()
    
    Example - Gaussian Mixture:
        >>> # Visualize multimodal distribution
        >>> gmm = GaussianMixture.symmetric_2D(
        ...     nmodes=5,
        ...     std=0.5,
        ...     scale=8
        ... )
        >>> 
        >>> fig, ax = plt.subplots()
        >>> imshow_density(
        ...     gmm,
        ...     bins=200,  # High resolution for detail
        ...     scale=12,
        ...     ax=ax,
        ...     cmap='RdYlBu_r',
        ...     vmin=-15,
        ...     alpha=0.8
        ... )
        >>> plt.show()
    
    Common Use Cases:
        - Background for particle plots (set alpha < 1)
        - Standalone density visualization
        - Target distribution for Langevin dynamics
        - Comparing learned vs true distributions
    
    See Also:
        - contour_density: Alternative contour-based visualization
        - scatter_sampleable: Overlay samples on density
    """
    if ax is None:
        ax = plt.gca()
    
    # Determine bounds
    x_range = x_bounds if x_bounds else [-scale, scale]
    y_range = y_bounds if y_bounds else [-scale, scale]

    # Create spatial grid
    x = torch.linspace(x_range[0] + x_offset, x_range[1] + x_offset, bins).to(device)
    y = torch.linspace(y_range[0], y_range[1], bins).to(device)
    X, Y = torch.meshgrid(x, y, indexing='xy')
    
    # Flatten to (bins², 2) using einops for clarity
    xy = rearrange([X, Y], 'c h w -> (h w) c')
    
    # Evaluate log density
    with torch.no_grad():
        log_density_vals = density.log_density(xy).reshape(bins, bins)
    
    # Display as image
    ax.imshow(
        log_density_vals.cpu().numpy(),
        extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
        origin='lower',
        **kwargs
    )


def contour_density(
    density: Density,
    bins: int,
    scale: float,
    device: str = 'cpu',
    ax: Optional[Axes] = None,
    x_offset: float = 0.0,
    **kwargs
) -> None:
    """
    Plot 2D probability density as contour lines.
    
    Shows level sets (iso-probability curves) of the density function.
    Each contour line represents points with equal probability density.
    
    Contours are often clearer than heatmaps for:
    - Showing mode locations precisely
    - Comparing multiple distributions
    - Overlaying on other plots
    
    Args:
        density (Density): Probability distribution to visualize.
        bins (int): Grid resolution.
        scale (float): Spatial extent [-scale, scale]².
        device (str): Computation device.
        ax (Optional[Axes]): Matplotlib axes.
        x_offset (float): Horizontal shift for the evaluation window.
        **kwargs: Passed to ax.contour():
            - levels: Number or list of contour levels
            - colors: Line colors
            - linewidths: Line thickness
            - linestyles: Solid, dashed, etc.
            - alpha: Transparency
    
    Returns:
        None.
    
    Contour Interpretation:
        - Closely spaced lines: Steep gradient (rapid probability change)
        - Widely spaced lines: Gradual gradient
        - Closed loops: Local maxima (modes)
        - Concentric circles: Isotropic Gaussian
    
    Example - Overlay on Heatmap:
        >>> # Combine heatmap and contours
        >>> density = GaussianMixture.random_2D(nmodes=3, std=1.0)
        >>> 
        >>> fig, ax = plt.subplots()
        >>> # Background heatmap
        >>> imshow_density(density, 100, 10, ax=ax,
        ...                cmap='Blues', alpha=0.5, vmin=-15)
        >>> # Overlay contours
        >>> contour_density(density, 100, 10, ax=ax,
        ...                 colors='black', linewidths=1.5,
        ...                 levels=10)
        >>> plt.show()
    
    Example - Multiple Distributions:
        >>> # Compare source and target
        >>> source = Gaussian(torch.zeros(2), 10*torch.eye(2))
        >>> target = GaussianMixture.symmetric_2D(5, 0.8, 8)
        >>> 
        >>> fig, ax = plt.subplots()
        >>> contour_density(source, 100, 15, ax=ax,
        ...                 colors='blue', linestyles='--',
        ...                 levels=5, label='Source')
        >>> contour_density(target, 100, 15, ax=ax,
        ...                 colors='red', linestyles='-',
        ...                 levels=10, label='Target')
        >>> ax.legend()
        >>> plt.show()
    
    See Also:
        - imshow_density: Heatmap visualization
    """
    if ax is None:
        ax = plt.gca()
    
    # Create grid
    x = torch.linspace(-scale + x_offset, scale + x_offset, bins).to(device)
    y = torch.linspace(-scale, scale, bins).to(device)
    X, Y = torch.meshgrid(x, y, indexing='xy')
    # Flatten to (bins², 2) using einops
    xy = rearrange([X, Y], 'c h w -> (h w) c')
    
    # Evaluate density
    log_density_vals = density.log_density(xy).reshape(bins, bins)
    
    # Plot contours
    ax.contour(
        log_density_vals.cpu().numpy(),
        extent=[-scale, scale, -scale, scale],
        origin='lower',
        **kwargs
    )


def get_trajectory_snapshot_indices(
    num_timesteps: int,
    step_size: int
) -> torch.Tensor:
    """
    Compute indices for trajectory snapshots at regular intervals.
    
    Selects a subset of timesteps for visualization, ensuring we always
    include the first and last timesteps.
    
    Args:
        num_timesteps (int): Total number of timesteps.
        step_size (int): Interval between snapshots.
            - step_size=1: All timesteps
            - step_size=10: Every 10th timestep
    
    Returns:
        torch.Tensor: Indices to plot.
            Shape: (num_snapshots,)
            Always includes 0 and num_timesteps-1
    
    Example:
        >>> indices = get_trajectory_snapshot_indices(100, 25)
        >>> indices  # tensor([0, 25, 50, 75, 99])
        >>> 
        >>> # Edge case: step_size=1 returns all
        >>> indices = get_trajectory_snapshot_indices(5, 1)
        >>> indices  # tensor([0, 1, 2, 3, 4])
    """
    if step_size == 1:
        return torch.arange(num_timesteps)
    return torch.cat([
        torch.arange(0, num_timesteps - 1, step_size),
        torch.tensor([num_timesteps - 1])
    ])


def plot_distribution_evolution(
    num_samples: int,
    source_distribution: Sampleable,
    simulator: Simulator,
    target_density: Density,
    timesteps: torch.Tensor,
    plot_interval: int,
    bins: int,
    scale: float,
    device: str = 'cpu'
) -> None:
    """
    Visualize evolution of particle distribution over time (static snapshots).
    
    Creates a multi-panel figure showing how a collection of particles
    evolves from a source distribution toward a target distribution.
    For each time snapshot:
    - Top row: Scatter plot of particle positions
    - Bottom row: KDE estimate of empirical density
    
    This is the primary visualization for understanding Langevin dynamics
    and other distribution-transforming SDEs.
    
    Args:
        num_samples (int): Number of particles to simulate.
            More particles → smoother KDE estimate
            Typical: 1000-10000
        
        source_distribution (Sampleable): Initial distribution.
            Particles sampled from this at t=0
        
        simulator (Simulator): SDE/ODE simulator.
            Usually EulerMaruyamaSimulator for Langevin
        
        target_density (Density): Target distribution (for background).
            Shown as heatmap to compare with particle evolution
        
        timesteps (torch.Tensor): Full simulation time grid.
            Shape: (num_timesteps,)
        
        plot_interval (int): Timesteps between snapshots.
            Larger → fewer panels (e.g., 334 → 3 snapshots for 1000 steps)
        
        bins (int): Resolution for density heatmap.
            Higher → smoother background (but slower)
        
        scale (float): Spatial extent of plots.
        
        device (str): Computation device.
    
    Returns:
        None. Creates and displays a matplotlib figure.
    
    Figure Layout:
        For n snapshots:
        - Figure size: (8n, 16)
        - Grid: 2 rows × n columns
        - Row 0: Particle scatter plots
        - Row 1: KDE density estimates
    
    Interpretation Guide:
        - **Initial**: Particles follow source distribution
        - **Middle**: Transition shows drift toward high-probability regions
        - **Final**: Particles concentrate near target modes
        - **KDE vs Target**: Should match closely if converged
    
    Example - Langevin Dynamics:
        >>> # Setup target
        >>> target = GaussianMixture.random_2D(
        ...     nmodes=5, std=0.75, scale=15, seed=42
        ... )
        >>> 
        >>> # Setup Langevin SDE
        >>> langevin = LangevinSDE(sigma=0.8, density=target)
        >>> sim = EulerMaruyamaSimulator(langevin)
        >>> 
        >>> # Source: Broad Gaussian
        >>> source = Gaussian(torch.zeros(2), 20*torch.eye(2))
        >>> 
        >>> # Simulate and visualize
        >>> plot_distribution_evolution(
        ...     num_samples=10000,
        ...     source_distribution=source,
        ...     simulator=sim,
        ...     target_density=target,
        ...     timesteps=torch.linspace(0, 5, 1000),
        ...     plot_interval=334,  # 3 snapshots
        ...     bins=200,
        ...     scale=15,
        ...     device='cpu'
        ... )
    
    Common Issues:
        - **Particles not reaching target**: Increase simulation time
        - **Noisy KDE**: Increase num_samples
        - **Poor convergence**: Adjust sigma in Langevin
        - **Particles stuck**: May have local minima (multimodal target)
    
    Performance:
        - Time: O(num_samples × num_timesteps × cost_per_step)
        - Memory: O(num_samples × num_timesteps × dim)
        - For large simulations: Use lower plot_interval
    
    See Also:
        - animate_distribution_evolution: Animated version
        - imshow_density: Background density rendering
    """
    # Sample initial particles
    x0 = source_distribution.sample(num_samples).to(device)
    
    # Simulate full trajectory
    traj_full = simulator.simulate_with_trajectory(x0, timesteps)
    
    # Select snapshot times
    indices_to_plot = get_trajectory_snapshot_indices(len(timesteps), plot_interval)
    snapshot_timesteps = timesteps[indices_to_plot]
    snapshot_states = traj_full[:, indices_to_plot]
    
    # Create figure
    num_snapshots = len(snapshot_timesteps)
    fig, axes = plt.subplots(2, num_snapshots, figsize=(8 * num_snapshots, 16))
    axes = axes.reshape((2, num_snapshots))
    
    # Plot each snapshot
    for i in range(num_snapshots):
        t = snapshot_timesteps[i].item()
        current_samples = snapshot_states[:, i]
        
        # === Top row: Scatter plot ===
        ax_scatter = axes[0, i]
        # Background: target density
        imshow_density(
            target_density, bins, scale, device, ax_scatter,
            vmin=-15, alpha=0.25, cmap=plt.get_cmap('Blues')
        )
        # Particles
        ax_scatter.scatter(
            current_samples[:, 0].cpu().numpy(),
            current_samples[:, 1].cpu().numpy(),
            marker='x', color='black', alpha=0.5, s=10
        )
        ax_scatter.set_title(f'Particles at t={t:.1f}', fontsize=15)
        ax_scatter.set_xticks([])
        ax_scatter.set_yticks([])
        
        # === Bottom row: KDE estimate ===
        ax_kde = axes[1, i]
        # Background: target density
        imshow_density(
            target_density, bins, scale, device, ax_kde,
            vmin=-15, alpha=0.5, cmap=plt.get_cmap('Blues')
        )
        # KDE contours
        sns.kdeplot(
            x=current_samples[:, 0].cpu().numpy(),
            y=current_samples[:, 1].cpu().numpy(),
            alpha=0.5, ax=ax_kde, color='grey', fill=False
        )
        ax_kde.set_title(f'Estimated Density at t={t:.1f}', fontsize=15)
        ax_kde.set_xticks([])
        ax_kde.set_yticks([])
        ax_kde.set_xlabel("")
        ax_kde.set_ylabel("")
    
    plt.tight_layout()
    plt.show()


def animate_distribution_evolution(
    num_samples: int,
    source_distribution: Sampleable,
    simulator: Simulator,
    target_density: Density,
    timesteps: torch.Tensor,
    animate_interval: int,
    bins: int,
    scale: float,
    device: str = 'cpu',
    save_path: str = 'dynamics_animation.mp4'
):
    """
    Create animation of distribution evolution over time.
    
    Similar to plot_distribution_evolution but generates an MP4 video
    showing smooth temporal evolution. Each frame shows:
    - Left: Particle scatter plot
    - Right: KDE density estimate
    
    Requires:
        - ffmpeg installed (conda install -c conda-forge ffmpeg)
        - celluloid package (pip install celluloid)
    
    Args:
        num_samples (int): Number of particles.
        source_distribution (Sampleable): Initial distribution.
        simulator (Simulator): SDE simulator.
        target_density (Density): Target for background.
        timesteps (torch.Tensor): Simulation time grid.
        animate_interval (int): Frames between animation frames.
            Smaller → smoother (but larger file)
        bins (int): Density visualization resolution.
        scale (float): Spatial extent.
        device (str): Computation device.
        save_path (str): Output video filename.
    
    Returns:
        IPython.display.HTML: Embeddable HTML5 video (for notebooks).
    
    File Output:
        Saves MP4 video to save_path with H.264 encoding.
    
    Animation Tips:
        - Frame rate: ~10-30 fps (depends on animate_interval)
        - Duration: (num_timesteps / animate_interval) / fps seconds
        - File size: ~1-10 MB for typical animations
        - Quality: Increase bins for higher resolution
    
    Example:
        >>> from IPython.display import display
        >>> 
        >>> target = GaussianMixture.symmetric_2D(5, 0.6, 8)
        >>> langevin = LangevinSDE(sigma=0.7, density=target)
        >>> sim = EulerMaruyamaSimulator(langevin)
        >>> source = Gaussian(torch.zeros(2), 15*torch.eye(2))
        >>> 
        >>> video = animate_distribution_evolution(
        ...     num_samples=5000,
        ...     source_distribution=source,
        ...     simulator=sim,
        ...     target_density=target,
        ...     timesteps=torch.linspace(0, 10, 2000),
        ...     animate_interval=20,  # ~100 frames
        ...     bins=150,
        ...     scale=12,
        ...     save_path='langevin_demo.mp4'
        ... )
        >>> display(video)  # Show in notebook
    
    Troubleshooting:
        - "ffmpeg not found": Install via conda/mamba
        - "ImportError: celluloid": pip install celluloid
        - Video won't play: Try different save_path extension (.gif)
        - Too large: Increase animate_interval
    
    See Also:
        - plot_distribution_evolution: Static snapshot version
        - celluloid.Camera: Animation framework used
    """
    from celluloid import Camera
    from IPython.display import HTML
    
    # Sample and simulate
    x0 = source_distribution.sample(num_samples).to(device)
    traj_full = simulator.simulate_with_trajectory(x0, timesteps)
    
    # Select animation frames
    indices_to_animate = get_trajectory_snapshot_indices(len(timesteps), animate_interval)
    frame_timesteps = timesteps[indices_to_animate]
    frame_states = traj_full[:, indices_to_animate]
    
    # Create figure and camera
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    camera = Camera(fig)
    
    # Render each frame
    for i in range(len(frame_timesteps)):
        t = frame_timesteps[i].item()
        current_samples = frame_states[:, i]
        
        # Left: Scatter
        ax_scatter = axes[0]
        imshow_density(
            target_density, bins, scale, device, ax_scatter,
            vmin=-15, alpha=0.25, cmap=plt.get_cmap('Blues')
        )
        ax_scatter.scatter(
            current_samples[:, 0].cpu().numpy(),
            current_samples[:, 1].cpu().numpy(),
            marker='x', color='black', alpha=0.5, s=10
        )
        ax_scatter.set_title(f'Particles at t={t:.1f}', fontsize=15)
        ax_scatter.set_xticks([])
        ax_scatter.set_yticks([])
        
        # Right: KDE
        ax_kde = axes[1]
        imshow_density(
            target_density, bins, scale, device, ax_kde,
            vmin=-15, alpha=0.25, cmap=plt.get_cmap('Blues')
        )
        sns.kdeplot(
            x=current_samples[:, 0].cpu().numpy(),
            y=current_samples[:, 1].cpu().numpy(),
            alpha=0.5, ax=ax_kde, color='grey', fill=False
        )
        ax_kde.set_title(f'Estimated Density at t={t:.1f}', fontsize=15)
        ax_kde.set_xticks([])
        ax_kde.set_yticks([])
        ax_kde.set_xlabel("")
        ax_kde.set_ylabel("")
        
        # Capture frame
        camera.snap()
    
    # Generate animation
    animation = camera.animate()
    animation.save(save_pathanimation.save(save_path))
    plt.close(fig)
    
    return HTML(animation.to_html5_video())