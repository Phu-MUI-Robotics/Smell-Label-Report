
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Function visualize total with peaks and windows
def plot_total_with_peaks():
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    axes[0].plot(x, total_raw, alpha=0.3, label="Total raw")
    axes[0].plot(x, total_gauss, label=f"Gaussian Total σ ={GAUSS_SIGMA:g}")
    
    px = x.iloc[peaks] if use_time else x.to_numpy()[peaks]
    
    axes[0].scatter(px, total_gauss[peaks], s=50, label="Peaks")
    axes[0].legend()
    axes[0].set_title("Total Raw + Gaussian + Peaks")
    
    for cid, g in df_peak_window.groupby("cycle_id", sort=True):
        axes[1].plot(x.loc[g.index], g[TOTAL_COL], marker="o", markersize=3, label=f"cycle {cid}")
    axes[1].set_title("Peak-Window Raw per Cycle")
    axes[1].legend(ncol=4, fontsize=8)
    
    plt.tight_layout()
    return fig

# Function radar plot of peak-window mean per cycle
def radar_peak_window(
    df_peak_window,
    sensor_cols,
    max_radar=30,
    nrows=6,
    ncols=5,
    figsize=(22, 18),
    normalize=True,
    top=0.889,
    bottom=0.015,
    left=0.008,
    right=0.992,
    hspace=0.679,
    wspace=0.024
):
    """Radar plot of Raw peak-window (mean per cycle)."""
    if df_peak_window.empty or df_peak_window is None:
        print("No peak-window data for radar plot.")
        return
    ordered_cols = [c for c in ["s1","s2","s3","s4","s5","s6","s7","s8"] if c in sensor_cols]
    radar_df = (
        df_peak_window.groupby("cycle_id")[ordered_cols]
        .mean()
        .reset_index()
        .set_index("cycle_id")
        .head(max_radar)
    )
    if normalize:
        mins = radar_df.min(axis=0)
        maxs = radar_df.max(axis=0)
        denom = (maxs - mins).replace(0, np.nan)
        radar_df = ((radar_df - mins) / denom).fillna(0.0)
    angles = np.linspace(0, 2 * np.pi, len(ordered_cols), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, subplot_kw=dict(polar=True))
    axes = axes.flatten()
    for ax, (cid, row) in zip(axes, radar_df.iterrows()):
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        vals = row.to_numpy(dtype=float)
        vals = np.concatenate((vals, [vals[0]]))
        ax.plot(angles, vals, linewidth=2)
        ax.fill(angles, vals, alpha=0.18)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(ordered_cols, fontsize=10)
        ax.set_yticklabels([])
        ax.set_title(f"Cycle {int(cid)}", fontsize=12, pad=10)
    # ซ่อนอันที่เกิน
    for ax in axes[len(radar_df):]:
        ax.set_visible(False)
    fig.suptitle(
        f"RADAR — RAW PEAK WINDOW (mean per cycle, window [-3, +3])",
        fontsize=18,
        y=0.97,
    )
    fig.subplots_adjust(
        top=top,
        bottom=bottom,
        left=left,
        right=right,
        hspace=hspace,
        wspace=wspace,
    )
    return fig, ncols

# Function PCA consistency plot
def pca_consistency_plot(df_peak_window, sensor_cols, fingerprint="mean", outlier_std=0.5, figsize=(9, 7)):
    """PCA consistency plot for peak-window cycles."""
    if fingerprint == 'max':
        cycle_feat = df_peak_window.groupby("cycle_id")[sensor_cols].max()
    elif fingerprint == 'min':
        cycle_feat = df_peak_window.groupby("cycle_id")[sensor_cols].min()
    elif fingerprint == "mean":
        cycle_feat = df_peak_window.groupby("cycle_id")[sensor_cols].mean()
    elif fingerprint == "p95":
        cycle_feat = df_peak_window.groupby("cycle_id")[sensor_cols].quantile(0.95)
    else:
        raise ValueError("FINGERPRINT must be one of: 'max', 'mean', 'p95', 'min'")

    X = cycle_feat.to_numpy(dtype=float)
    scaler = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=0)
    pca_fit = pca.fit_transform(scaler)

    loading = pd.DataFrame(pca.components_.T, index=sensor_cols, columns=["PC1", "PC2"])

    center = pca_fit.mean(axis=0)
    dist = np.linalg.norm(pca_fit - center, axis=1)
    thr = dist.mean() + outlier_std * dist.std(ddof=1)
    labels = np.where(dist <= thr, "OK", "OUTLIER")

    fig, ax = plt.subplots(figsize=figsize)
    for lab, mk in [("OK", "o"), ("OUTLIER", "x")]:
        m = labels == lab
        ax.scatter(pca_fit[m, 0], pca_fit[m, 1], s=90, marker=mk, label=lab)
    
    # Calculate dynamic offset based on data range
    y_range = pca_fit[:, 1].max() - pca_fit[:, 1].min()
    offset = y_range * 0.05  # 5% of y-range
    
    for cid, (x1, x2), lab in zip(cycle_feat.index, pca_fit, labels):
        if lab == "OK":
            # Show numbers above OK circles (blue dots)
            ax.text(x1, x2 + offset, str(int(cid)), ha="center", va="bottom", fontsize=10, fontweight='bold', color='blue')
        else:
            # Show numbers above OUTLIER X marks
            ax.text(x1, x2 + offset, str(int(cid)), ha="center", va="bottom", fontsize=10, fontweight='bold', color='orange')

    circle = plt.Circle(center, thr, color="red", fill=False, linestyle="--", label="Outlier Threshold")
    ax.add_patch(circle)
    ax.scatter(center[0], center[1], marker="*", s=220, label="Centroid")
    # Add dummy point for cycle number legend
    ax.scatter([], [], alpha=0, label="Numbers = Cycle ID")
    ax.set_title(f"PCA consistency ({fingerprint.upper()} fingerprint)")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig, loading, labels

def box_plot(df_peak_window, sensor_cols, figsize=(10, 6)):
    """Box plot for sensor value distribution in peak-window data."""
    fig = plt.figure(figsize=figsize)
    sns.boxplot(
        data=df_peak_window[sensor_cols],
        orient="v",
        palette="Set2"
    )
    plt.title("Sensor Value Distribution (Boxplot) — peak-window raw")
    plt.xlabel("Sensor")
    plt.ylabel("Value")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    return fig

def create_report_package(chemical_name, figs):
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        fig0 = plt.figure(figsize=(8, 3))
        fig0.text(0.5, 0.5, f"Chemical Name: {chemical_name}", ha='center', va='center', fontsize=18)
        pdf.savefig(fig0)
        plt.close(fig0)
        for fig in figs:
            if fig is not None:
                pdf.savefig(fig)
    buf.seek(0)
    return buf








