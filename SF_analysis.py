from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "Times New Roman"],
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "axes.linewidth": 1.2,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight"
})

# =========================
# Config and result classes
# =========================

@dataclass
class SFAnalysisConfig:
    r_col: str = "R_pix"
    dphi_col: str = "Dphi"
    npairs_col: str = "Npairs"
    r_pc_col: str = "R_pc"

    min_points: int = 8
    min_npairs: int = 30

    tail_fraction: float = 0.25
    smooth_window: int = 5

    rel_tol: float = 0.08
    sigma_tol: float = 2.0

    require_consecutive_tail_points: int = 3
    start_offset_pix: int = 3

    # Физические и алгоритмические пороги (бывшие "магические числа")
    critical_slope: float = 0.015
    angle_tolerance_deg: float = 15.0
    noise_limit: float = 0.5


@dataclass
@dataclass
class SFAnalysisResult:
    object_name: Optional[str]
    mask_name: Optional[str]
    dataset: Optional[str]

    n_points_raw: int
    n_points_valid: int
    r_max_pix: Optional[float]

    plateau_detected: str
    plateau_level: Optional[float]
    plateau_scatter: Optional[float]
    saturation_scale_pix: Optional[float]
    saturation_scale_pc: Optional[float]

    tail_start_pix: Optional[float]
    
    # НОВЫЕ ПОЛЯ ДЛЯ СИНХРОНИЗАЦИИ С ТАБЛИЦЕЙ:
    ma_proxy: Optional[float] = None
    l_fit_px: Optional[float] = None
    
    notes: str = ""

def safe_filename(s: str) -> str:
    s = str(s)
    s = re.sub(r'[<>:"/\\|?*]', "_", s)
    s = s.replace(" ", "_")
    return s


# =========================
# Core functions
# =========================

def load_sf_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path)
    df["source_file"] = str(path)
    return df


def validate_sf_table(df: pd.DataFrame, config: SFAnalysisConfig) -> None:
    required = [config.r_col, config.dphi_col, config.npairs_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def prepare_valid_sf(df: pd.DataFrame, config: SFAnalysisConfig) -> pd.DataFrame:
    out = df.copy()

    out = out.dropna(subset=[config.r_col, config.dphi_col, config.npairs_col])
    out = out[out[config.r_col] > 0]
    out = out[out[config.npairs_col] > 0]
    out = out.sort_values(config.r_col).reset_index(drop=True)

    out = out[out[config.npairs_col] >= config.min_npairs].copy()

    if len(out) == 0:
        return out

    win = max(3, config.smooth_window)
    if win % 2 == 0:
        win += 1

    out["Dphi_smooth"] = (
        out[config.dphi_col]
        .rolling(window=win, center=True, min_periods=1)
        .median()
    )
    
    # Жесткая зачистка потенциальных NaN на краях, если min_periods не справился
    out = out.dropna(subset=["Dphi_smooth"]).reset_index(drop=True)

    return out


def estimate_plateau_from_tail(df: pd.DataFrame, config: SFAnalysisConfig) -> dict:
    n = len(df)
    n_tail = max(config.require_consecutive_tail_points, int(np.ceil(config.tail_fraction * n)))
    tail = df.iloc[-n_tail:].copy()

    plateau_level = float(np.median(tail[config.dphi_col]))
    mad = float(np.median(np.abs(tail[config.dphi_col] - plateau_level)))
    plateau_scatter = 1.4826 * mad

    tail_start_pix = float(tail[config.r_col].min())

    return {
        "plateau_level": plateau_level,
        "plateau_scatter": plateau_scatter,
        "tail_start_pix": tail_start_pix,
        "n_tail": n_tail,
    }


def detect_saturation_scale(
    df: pd.DataFrame,
    plateau_level: float,
    plateau_scatter: float,
    config: SFAnalysisConfig
) -> tuple[Optional[float], str, float]:
    vals = df[config.dphi_col].to_numpy()
    rvals = df[config.r_col].to_numpy()

    tol = max(config.rel_tol * plateau_level, config.sigma_tol * plateau_scatter)
    within = np.abs(vals - plateau_level) <= tol

    needed = config.require_consecutive_tail_points
    n = len(df)

    for i in range(n):
        if np.all(within[i:]):
            return float(rvals[i]), "detected", tol

    if np.sum(within[-needed:]) == needed:
        idx = np.where(within)[0]
        if len(idx) > 0:
            return float(rvals[idx[0]]), "uncertain", tol

    return None, "not_detected", tol


def estimate_r_break_piecewise_loglog(
    df: pd.DataFrame,
    config: SFAnalysisConfig,
    use_smooth: bool = True,
    min_segment_points: int = 3
) -> dict:
    y_col = "Dphi_smooth" if use_smooth and "Dphi_smooth" in df.columns else config.dphi_col

    work = df[[config.r_col, y_col]].copy()
    if config.r_pc_col in df.columns:
        work[config.r_pc_col] = df[config.r_pc_col]

    work = work.dropna(subset=[config.r_col, y_col]).copy()
    work = work[(work[config.r_col] > 0) & (work[y_col] > 0)].copy()
    work = work.sort_values(config.r_col).reset_index(drop=True)

    n = len(work)
    if n < min_segment_points * 2:
        return {"r_break_pix": None, "r_break_pc": None, "break_idx": None, "status": "not_enough_points"}

    r_vals = work[config.r_col].to_numpy()
    dphi_vals = work[y_col].to_numpy()

    slopes = np.diff(dphi_vals) / np.diff(r_vals)
    
    idx_break = 0
    
    # Сканируем справа налево
    start_scan_idx = int((n - 1) * 0.8)
    for i in range(start_scan_idx, 0, -1):
        if slopes[i-1] > config.critical_slope:
            if i > 1 and slopes[i-2] > config.critical_slope:
                idx_break = i
                break

    if idx_break == 0:
        idx_break = np.argmin(np.abs(slopes))

    r_break_current = float(r_vals[idx_break])

    # КРИТЕРИЙ 1: Физический масштаб
    if r_break_current <= 5.0:
        return {"r_break_pix": None, "status": "no_valid_plateau_indices", "notes": "Plateau starts too early (<= 5 px)"}

    tentative_level = np.median(dphi_vals[idx_break:])
    
    # КРИТЕРИЙ 1.5: Защита от белого шума (медиана не должна превышать noise_limit)
    if tentative_level > config.noise_limit:
        return {
            "r_break_pix": None, 
            "status": "noise_dominated", 
            "notes": f"Plateau level ({tentative_level:.3f}) exceeds white noise limit ({config.noise_limit})"
        }

    # УМНЫЙ ДЕТЕКТОР ХВОСТА
    angle_tolerance_rad_sq = (config.angle_tolerance_deg * np.pi / 180.0) ** 2
    max_tail_points = max(1, int(n * 0.25))
    tail_start_idx = n 
    
    for i in range(n - 1, max(idx_break, n - max_tail_points - 1), -1):
        val = dphi_vals[i]
        # Здесь также отсекаем точки хвоста, пробившие потолок шума
        if val > config.noise_limit or abs(val - tentative_level) > angle_tolerance_rad_sq:
            tail_start_idx = i 
        else:
            break

    # КРИТЕРИЙ 2: Достаточное количество точек
    points_on_plateau = tail_start_idx - idx_break
    if points_on_plateau <= 5:
        return {"r_break_pix": None, "status": "no_valid_plateau_indices", "notes": f"Too few points on plateau ({points_on_plateau} <= 5)"}

    x_log = np.log10(r_vals)
    y_log = np.log10(dphi_vals)

    x_left, y_left = x_log[:idx_break + 1], y_log[:idx_break + 1]
    x_right, y_right = x_log[idx_break:tail_start_idx], y_log[idx_break:tail_start_idx]

    slope_left, intercept_left = np.polyfit(x_left, y_left, 1)
    slope_right, intercept_right = np.polyfit(x_right, y_right, 1)

    return {
        "r_break_pix": r_break_current,
        "r_break_pc": float(work.loc[idx_break, config.r_pc_col]) if config.r_pc_col in work.columns else None,
        "break_idx": int(idx_break),
        "slope_left": float(slope_left),
        "slope_right": float(slope_right),
        "intercept_left": float(intercept_left),
        "intercept_right": float(intercept_right),
        "status": "ok",
    }

# =========================
# Plotting
# =========================

def plot_sf_diagnostic(
    valid: pd.DataFrame,
    result: SFAnalysisResult,
    extras: dict,
    config: SFAnalysisConfig,
    title: str = "SF diagnostic",
    tail_start_px=None,
    save_path=None
):
    fig, ax = plt.subplots(figsize=(8, 5))

    x = valid[config.r_col].to_numpy()
    y_raw = valid[config.dphi_col].to_numpy()
    y_smooth = valid["Dphi_smooth"].to_numpy()

    mask_raw = (x > 0) & (y_raw > 0)
    mask_smooth = (x > 0) & (y_smooth > 0)

    ax.loglog(x[mask_raw], y_raw[mask_raw], marker="o", color="black", lw=1.2, alpha=0.6, label="raw Dphi")
    ax.loglog(x[mask_smooth], y_smooth[mask_smooth], color="darkorange", lw=2, label="smoothed Dphi")

    ax.axhline(0.5, color="red", lw=1.5, ls="-", label="White Noise Limit (0.5)")

    if result.plateau_level is not None and result.plateau_level > 0:
        plateau = result.plateau_level
        tol = extras.get("tol", 0.0) # Безопасное извлечение
        y_low = plateau - tol
        y_high = plateau + tol

        ax.axhline(plateau, color="royalblue", ls="--", lw=1.8, label=f"plateau level = {plateau:.4f}")
        
        r_start = result.saturation_scale_pix or 1.0
        r_end = extras.get("r_end_plateau_pix") or float(x.max())
        
        plot_mask = (x >= r_start) & (x <= r_end)
        if np.any(plot_mask) and y_low > 0:
            ax.fill_between(
                x[plot_mask], y_low, y_high, 
                color="royalblue", alpha=0.18, label="plateau error band"
            )

    break_info = extras.get("break_info", {})
    r_break_pix = break_info.get("r_break_pix", None)

    if break_info.get("status") == "ok":
        y_col = "Dphi_smooth" if "Dphi_smooth" in valid.columns else config.dphi_col
        work = valid[[config.r_col, y_col]].dropna().copy()
        work = work[(work[config.r_col] > 0) & (work[y_col] > 0)].sort_values(config.r_col).reset_index(drop=True)

        if len(work) > 0:
            break_idx = break_info["break_idx"]
            left = work.iloc[:break_idx+1].copy()
            right = work.iloc[break_idx:].copy()

            if len(left) >= 2:
                x_l = left[config.r_col].to_numpy()
                y_l_fit = 10 ** (break_info["slope_left"] * np.log10(x_l) + break_info["intercept_left"])
                ax.loglog(x_l, y_l_fit, color="forestgreen", lw=2, ls=":", label=f"Left Fit (s={break_info['slope_left']:.2f})")

            if len(right) >= 2:
                x_r = right[config.r_col].to_numpy()
                y_r_fit = 10 ** (break_info["slope_right"] * np.log10(x_r) + break_info["intercept_right"])
                ax.loglog(x_r, y_r_fit, color="purple", lw=2, ls=":", label=f"Right Fit (s={break_info['slope_right']:.2f})")

    if r_break_pix is not None and r_break_pix > 0:
        ax.axvline(r_break_pix, color="darkgreen", ls="-.", lw=1.8, label=f"r_break (Plateau Start) = {r_break_pix:.2f} px")

    if tail_start_px is not None and tail_start_px  > 0:
        ax.axvline(tail_start_px , color="royalblue", ls=":", lw=1.5, label=f"Tail Start (Plateau End) = {tail_start_px :.2f} px")

    ax.set_xlabel(r"Separation $R$ [pixels]")
    ax.set_ylabel(r"Structure Function $D_{\psi}(R)$")
    ax.set_title(title)
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(alpha=0.2, which="both")
    
    y_min_limit = min(.01, np.nanmin(y_raw) * .75)
    y_max_limit = max(1.0, np.nanmax(y_raw) * 1.5)
    ax.set_ylim(y_min_limit, y_max_limit)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_npairs(valid: pd.DataFrame, config: SFAnalysisConfig, title: str = "Npairs diagnostic", save_path=None):
    fig, ax = plt.subplots(figsize=(8, 4))

    x = valid[config.r_col].to_numpy()
    y = valid[config.npairs_col].to_numpy()
    mask = (x > 0) & (y > 0)

    ax.loglog(x[mask], y[mask], marker="o", lw=1.5)
    ax.set_xlabel(config.r_col)
    ax.set_ylabel(config.npairs_col)
    ax.set_title(title)
    ax.grid(alpha=0.3, which="both")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close() 
    else:
        plt.show()

def detect_tail_oscillations(
    df: pd.DataFrame, 
    config: SFAnalysisConfig, 
    plateau_level: float, 
    plateau_scatter: float
) -> int:
    # Уменьшаем множитель с 2.0 (config.sigma_tol) до 1.2 или 1.0
    # Это сделает детектор более чувствительным к отклонениям
    strict_multiplier = 1.1 
    threshold = strict_multiplier * plateau_scatter
    
    n = len(df)
    tail_idx_start = int(n * (1 - config.tail_fraction))
    tail_df = df.iloc[tail_idx_start:].copy()
    
    for i in range(len(tail_df) - 1, 0, -1):
        val = tail_df.iloc[i][config.dphi_col]
        # Если отклонение превышает 1.2 * scatter, считаем это началом "вихляния"
        if abs(val - plateau_level) > threshold:
            return tail_idx_start + i
            
    return tail_idx_start
    
def analyze_sf_table(
    df: pd.DataFrame, 
    config: SFAnalysisConfig, 
    output_dir: Path, 
    l_pix_fallback: Optional[float] = None  # Передаем длину скелета сюда
) -> tuple[SFAnalysisResult, pd.DataFrame, dict]:
    
    df = df[df[config.r_col] >= config.start_offset_pix].copy()
    
    if len(df) < config.min_points:
        return SFAnalysisResult(
            object_name=df["object_name"].iloc[0] if "object_name" in df.columns else None,
            mask_name=df["mask_name"].iloc[0] if "mask_name" in df.columns else None,
            dataset=df["dataset"].iloc[0] if "dataset" in df.columns else None,
            n_points_raw=len(df), n_points_valid=0, r_max_pix=None,
            plateau_detected="too_few_points", plateau_level=None, plateau_scatter=None,
            saturation_scale_pix=None, saturation_scale_pc=None, tail_start_pix=None,
            ma_proxy=None, l_fit_px=l_pix_fallback,
            notes="Too few valid points"
        ), df, {}
        
    validate_sf_table(df, config)
    
    object_name = df["object_name"].iloc[0] if "object_name" in df.columns else None
    mask_name = df["mask_name"].iloc[0] if "mask_name" in df.columns else None
    dataset = df["dataset"].iloc[0] if "dataset" in df.columns else None

    r_break_pix = None
    plateau_status = "not_detected"
    chosen_plateau_level = None
    plateau_scatter = None
    tail_start_pix = None
    r_end_plateau_pix = None
    tol = 0.0
    ma_calculated = None # Будущее значение MA

    n_points_raw = len(df)
    valid = prepare_valid_sf(df, config)
    n_points_valid = len(valid)

    if n_points_valid < config.min_points:
        return SFAnalysisResult(
            object_name=object_name, mask_name=mask_name, dataset=dataset,
            n_points_raw=n_points_raw, n_points_valid=n_points_valid,
            r_max_pix=None if n_points_valid == 0 else float(valid[config.r_col].max()),
            plateau_detected="not_detected", plateau_level=None, plateau_scatter=None,
            saturation_scale_pix=None, saturation_scale_pc=None, tail_start_pix=None,
            ma_proxy=None, l_fit_px=l_pix_fallback,
            notes="Too few valid points after cleaning",
        ), valid, {}

    break_info = estimate_r_break_piecewise_loglog(valid, config=config, use_smooth=False, min_segment_points=3)

    if break_info.get("status") == "ok":
        r_break_pix = break_info["r_break_pix"]
        idx_break = break_info["break_idx"]
        plateau_status = "detected"
        
        full_plateau = valid.iloc[idx_break:].copy().reset_index(drop=True)
        temp_level = float(np.median(full_plateau[config.dphi_col]))
        mad = np.median(np.abs(full_plateau[config.dphi_col] - temp_level))
        scatter = 1.4826 * mad if mad > 0 else 0.02
        
        tail_osc_idx = detect_tail_oscillations(full_plateau, config, temp_level, scatter)
        clean_plateau = full_plateau.iloc[:tail_osc_idx].copy()
        
        chosen_plateau_level = float(np.median(clean_plateau[config.dphi_col]))
        plateau_scatter = scatter 
        tol = max(config.rel_tol * chosen_plateau_level, 2.0 * plateau_scatter)
        
        r_end_plateau_pix = float(clean_plateau[config.r_col].max()) 
        tail_start_pix = r_end_plateau_pix                           
        
        # ВЫЧИСЛЕНИЕ ЧИСЛА МАХА ПРЯМО ТУТ:
        if l_pix_fallback is not None and l_pix_fallback > 0 and r_break_pix > 0:
            mach_res = estimate_mach_from_break_and_length(r_break_pix=r_break_pix, L_pix=l_pix_fallback)
            if mach_res.status == "ok":
                ma_calculated = mach_res.MA_proxy
  
    else:
        r_break_pix = None
        plateau_status = "not_detected"
        chosen_plateau_level = None
        plateau_scatter = None
        tail_start_pix = None
        r_end_plateau_pix = None
        tol = 0.0

    r_sat_pc = None
    if r_break_pix is not None and config.r_pc_col in valid.columns:
        tmp = valid[[config.r_col, config.r_pc_col]].dropna()
        if len(tmp) > 1:
            r_sat_pc = float(np.interp(r_break_pix, tmp[config.r_col], tmp[config.r_pc_col]))

    # Сохраняем все вычисленные данные в результирующий класс
    result = SFAnalysisResult(
        object_name=object_name, mask_name=mask_name, dataset=dataset,
        n_points_raw=n_points_raw, n_points_valid=n_points_valid,
        r_max_pix=float(valid[config.r_col].max()),
        plateau_detected=plateau_status,
        plateau_level=chosen_plateau_level,
        plateau_scatter=plateau_scatter,
        saturation_scale_pix=r_break_pix, 
        saturation_scale_pc=r_sat_pc,
        tail_start_pix=tail_start_pix,  
        ma_proxy=ma_calculated,        # Пишется в MA_proxy таблицы
        l_fit_px=l_pix_fallback,       # Пишется в L_fit_px таблицы
        notes=break_info.get("notes", f"Fit status: {break_info.get('status')}"),
    )

    extras = {
        "tol": tol,
        "break_info": break_info,
        "r_end_plateau_pix": r_end_plateau_pix  
    }
    
    # [Блок отрисовки графиков остается без изменений]
    plots_dir = output_dir / "plots" / (object_name or "unknown_object")
    plots_dir.mkdir(parents=True, exist_ok=True)
    safe_object = safe_filename(object_name or "unknown_object")
    safe_mask = safe_filename(mask_name or "unknown_mask")
    
    plot_sf_diagnostic(
        valid, result, extras, config,
        tail_start_px=r_end_plateau_pix,
        save_path=output_dir / "plots" / (object_name or "unknown_object") / f"{safe_object}_{safe_mask}_sf_diagnostic.pdf"
    )

    plot_npairs(
        valid, config, 
        save_path=output_dir / "plots" / (object_name or "unknown_object") / f"{safe_object}_{safe_mask}_npairs.pdf"
    )

    return result, valid, extras

# =========================
# Mach estimate
# =========================

@dataclass
class MachEstimateResult:
    object_name: Optional[str]
    mask_name: Optional[str]

    lA_proxy_pix: Optional[float]
    L_proxy_pix: Optional[float]

    MA_proxy: Optional[float]
    status: str
    notes: str = ""


def estimate_mach_from_break_and_length(
    r_break_pix: Optional[float],
    L_pix: Optional[float],
    object_name: Optional[str] = None,
    mask_name: Optional[str] = None,
) -> MachEstimateResult:
    """
    Estimate Alfvén Mach number using:
        MA = (L / lA)^(1/3)

    with proxies:
        lA ~ r_break
        L  ~ skeleton spine length
    """
    if r_break_pix is None or L_pix is None:
        return MachEstimateResult(
            object_name=object_name,
            mask_name=mask_name,
            lA_proxy_pix=r_break_pix,
            L_proxy_pix=L_pix,
            MA_proxy=None,
            status="missing_input",
            notes="r_break_pix or L_pix is None",
        )

    if r_break_pix <= 0 or L_pix <= 0:
        return MachEstimateResult(
            object_name=object_name,
            mask_name=mask_name,
            lA_proxy_pix=r_break_pix,
            L_proxy_pix=L_pix,
            MA_proxy=None,
            status="invalid_input",
            notes="r_break_pix and L_pix must be > 0",
        )

    ma = (L_pix / r_break_pix) ** (1.0 / 3.0)

    notes = "Proxy estimate using lA ~ r_break and L ~ skeleton spine length"

    return MachEstimateResult(
        object_name=object_name,
        mask_name=mask_name,
        lA_proxy_pix=r_break_pix,
        L_proxy_pix=L_pix,
        MA_proxy=float(ma),
        status="ok",
        notes=notes,
    )