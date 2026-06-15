from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QVBoxLayout, QWidget

from ..parsers.base_parser import EISData
from ..fitting.fitting_engine import FittingResult


@dataclass
class PlotStyle:
    color: str = "#1f77b4"
    marker: str = "o"
    linestyle: str = "-"
    linewidth: float = 1.5
    markersize: float = 4
    alpha: float = 1.0
    label: str = ""


@dataclass
class PlotSample:
    sample_id: str
    data: EISData
    style: PlotStyle
    fitting_result: Optional[FittingResult] = None


class _NyquistCanvas(FigureCanvas):
    """奈奎斯特图画布"""

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(5, 4), dpi=100, tight_layout=True)
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111)
        self._setup_axes()

    def _setup_axes(self):
        self.ax.set_xlabel("Z' (Ω)", fontsize=10, fontfamily="SimHei")
        self.ax.set_ylabel("-Z\" (Ω)", fontsize=10, fontfamily="SimHei")
        self.ax.set_title("奈奎斯特图", fontsize=11, fontfamily="SimHei")
        self.ax.grid(True, linestyle="--", alpha=0.6)
        self.ax.set_aspect("equal", adjustable="datalim")

    def clear(self):
        self.ax.clear()
        self._setup_axes()

    def plot_sample(self, sample: PlotSample, show_fit: bool = True):
        if not sample.data.is_valid:
            return
        z_imag_pos = -sample.data.z_imag
        self.ax.plot(
            sample.data.z_real, z_imag_pos,
            color=sample.style.color,
            marker=sample.style.marker,
            linestyle=sample.style.linestyle,
            linewidth=sample.style.linewidth,
            markersize=sample.style.markersize,
            alpha=sample.style.alpha,
            label=sample.style.label or sample.sample_id,
        )
        if show_fit and sample.fitting_result and sample.fitting_result.success:
            fit_z_imag_pos = -sample.fitting_result.fitted_z_imag
            self.ax.plot(
                sample.fitting_result.fitted_z_real, fit_z_imag_pos,
                color=sample.style.color,
                linestyle="--",
                linewidth=sample.style.linewidth * 0.8,
                markersize=0,
                alpha=0.7,
            )
        self.ax.legend(loc="best", fontsize=8, prop={"family": "SimHei"})

    def set_xlim(self, xmin: Optional[float] = None, xmax: Optional[float] = None):
        if xmin is not None and xmax is not None:
            self.ax.set_xlim(xmin, xmax)
        elif xmin is not None:
            self.ax.set_xlim(left=xmin)
        elif xmax is not None:
            self.ax.set_xlim(right=xmax)

    def set_ylim(self, ymin: Optional[float] = None, ymax: Optional[float] = None):
        if ymin is not None and ymax is not None:
            self.ax.set_ylim(ymin, ymax)
        elif ymin is not None:
            self.ax.set_ylim(bottom=ymin)
        elif ymax is not None:
            self.ax.set_ylim(top=ymax)

    def auto_scale(self, samples: List[PlotSample]):
        all_real = np.concatenate([s.data.z_real for s in samples if s.data.is_valid])
        all_imag = np.concatenate([-s.data.z_imag for s in samples if s.data.is_valid])
        if len(all_real) > 0 and len(all_imag) > 0:
            margin = 0.1
            xmin, xmax = all_real.min(), all_real.max()
            ymin, ymax = all_imag.min(), all_imag.max()
            dx = xmax - xmin
            dy = ymax - ymin
            d = max(dx, dy) * margin
            self.ax.set_xlim(xmin - d, xmax + d)
            self.ax.set_ylim(ymin - d, ymax + d)
            self.ax.set_aspect("equal", adjustable="datalim")


class _BodeCanvas(FigureCanvas):
    """波特图画布"""

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(5, 4), dpi=100, tight_layout=True)
        super().__init__(self.fig)
        gs = GridSpec(2, 1, figure=self.fig, height_ratios=[1, 1], hspace=0.3)
        self.ax_mag = self.fig.add_subplot(gs[0])
        self.ax_phase = self.fig.add_subplot(gs[1], sharex=self.ax_mag)
        self._setup_axes()

    def _setup_axes(self):
        self.ax_mag.set_xscale("log")
        self.ax_mag.set_yscale("log")
        self.ax_mag.set_ylabel("|Z| (Ω)", fontsize=10, fontfamily="SimHei")
        self.ax_mag.set_title("波特图", fontsize=11, fontfamily="SimHei")
        self.ax_mag.grid(True, which="both", linestyle="--", alpha=0.6)

        if not self.ax_phase.get_shared_x_axes().joined(self.ax_phase, self.ax_mag):
            self.ax_phase.sharex(self.ax_mag)
        self.ax_phase.set_xscale("log")
        self.ax_phase.set_xlabel("频率 f (Hz)", fontsize=10, fontfamily="SimHei")
        self.ax_phase.set_ylabel("相位 φ (°)", fontsize=10, fontfamily="SimHei")
        self.ax_phase.grid(True, which="both", linestyle="--", alpha=0.6)

    def clear(self):
        self.ax_mag.clear()
        self.ax_phase.clear()
        self._setup_axes()

    def plot_sample(self, sample: PlotSample, show_fit: bool = True):
        if not sample.data.is_valid:
            return
        freqs = sample.data.frequencies
        mag = sample.data.z_magnitude
        phase = sample.data.z_phase

        self.ax_mag.plot(
            freqs, mag,
            color=sample.style.color,
            marker=sample.style.marker,
            linestyle=sample.style.linestyle,
            linewidth=sample.style.linewidth,
            markersize=sample.style.markersize,
            alpha=sample.style.alpha,
            label=sample.style.label or sample.sample_id,
        )
        self.ax_phase.plot(
            freqs, phase,
            color=sample.style.color,
            marker=sample.style.marker,
            linestyle=sample.style.linestyle,
            linewidth=sample.style.linewidth,
            markersize=sample.style.markersize,
            alpha=sample.style.alpha,
        )

        if show_fit and sample.fitting_result and sample.fitting_result.success:
            fit_freqs = sample.fitting_result.fitted_freq
            fit_z = sample.fitting_result.fitted_z_real + 1j * sample.fitting_result.fitted_z_imag
            fit_mag = np.abs(fit_z)
            fit_phase = np.degrees(np.arctan2(np.imag(fit_z), np.real(fit_z)))
            self.ax_mag.plot(
                fit_freqs, fit_mag,
                color=sample.style.color,
                linestyle="--",
                linewidth=sample.style.linewidth * 0.8,
                markersize=0,
                alpha=0.7,
            )
            self.ax_phase.plot(
                fit_freqs, fit_phase,
                color=sample.style.color,
                linestyle="--",
                linewidth=sample.style.linewidth * 0.8,
                markersize=0,
                alpha=0.7,
            )

        self.ax_mag.legend(loc="best", fontsize=8, prop={"family": "SimHei"})

    def set_xlim(self, xmin: Optional[float] = None, xmax: Optional[float] = None):
        if xmin is not None and xmax is not None:
            self.ax_mag.set_xlim(xmin, xmax)
            self.ax_phase.set_xlim(xmin, xmax)
        elif xmin is not None:
            self.ax_mag.set_xlim(left=xmin)
            self.ax_phase.set_xlim(left=xmin)
        elif xmax is not None:
            self.ax_mag.set_xlim(right=xmax)
            self.ax_phase.set_xlim(right=xmax)

    def set_ylim_mag(self, ymin: Optional[float] = None, ymax: Optional[float] = None):
        if ymin is not None and ymax is not None:
            self.ax_mag.set_ylim(ymin, ymax)
        elif ymin is not None:
            self.ax_mag.set_ylim(bottom=ymin)
        elif ymax is not None:
            self.ax_mag.set_ylim(top=ymax)

    def set_ylim_phase(self, ymin: Optional[float] = None, ymax: Optional[float] = None):
        if ymin is not None and ymax is not None:
            self.ax_phase.set_ylim(ymin, ymax)
        elif ymin is not None:
            self.ax_phase.set_ylim(bottom=ymin)
        elif ymax is not None:
            self.ax_phase.set_ylim(top=ymax)

    def auto_scale(self, samples: List[PlotSample]):
        valid = [s for s in samples if s.data.is_valid]
        if not valid:
            return

        all_freq = np.concatenate([s.data.frequencies for s in valid])
        all_mag = np.concatenate([s.data.z_magnitude for s in valid])
        all_phase = np.concatenate([s.data.z_phase for s in valid])

        log_margin = 0.15
        fmin, fmax = float(np.nanmin(all_freq)), float(np.nanmax(all_freq))
        if fmin > 0 and fmax > fmin:
            fmin_set = fmin / (1.0 + log_margin)
            fmax_set = fmax * (1.0 + log_margin)
            self.ax_mag.set_xlim(fmin_set, fmax_set)

        m_valid = all_mag[all_mag > 0]
        if len(m_valid) > 0:
            mmin, mmax = float(np.nanmin(m_valid)), float(np.nanmax(m_valid))
            if mmax > mmin > 0:
                mmin_set = mmin / (1.0 + log_margin)
                mmax_set = mmax * (1.0 + log_margin)
                self.ax_mag.set_ylim(mmin_set, mmax_set)

        p_valid = all_phase[~np.isnan(all_phase)]
        if len(p_valid) > 0:
            pmin, pmax = float(np.nanmin(p_valid)), float(np.nanmax(p_valid))
            dp = pmax - pmin
            if dp < 0.5:
                center = (pmin + pmax) * 0.5
                pmin_set = center - 5.0
                pmax_set = center + 5.0
            else:
                pad = max(dp * 0.15, 2.0)
                pmin_set = pmin - pad
                pmax_set = pmax + pad

            if pmin_set < -180.0:
                pmin_set = -180.0
            if pmax_set > 180.0:
                pmax_set = 180.0
            self.ax_phase.set_ylim(pmin_set, pmax_set)


class DualPlotCanvas(QWidget):
    """奈奎斯特 + 波特图双视图画布"""

    sample_clicked = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._samples: Dict[str, PlotSample] = {}
        self._color_cycle = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        ]
        self._color_idx = 0
        self._show_fit = True

        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        self.nyquist_canvas = _NyquistCanvas(self)
        self.bode_canvas = _BodeCanvas(self)

        self.nyquist_toolbar = NavigationToolbar(self.nyquist_canvas, self)
        self.bode_toolbar = NavigationToolbar(self.bode_canvas, self)

        layout.addWidget(self.nyquist_toolbar)
        layout.addWidget(self.nyquist_canvas, stretch=1)
        layout.addWidget(self.bode_toolbar)
        layout.addWidget(self.bode_canvas, stretch=1)

    def add_sample(self, data: EISData, fitting_result: Optional[FittingResult] = None) -> str:
        sample_id = data.sample_id
        if sample_id in self._samples:
            base_id = sample_id
            i = 1
            while f"{base_id}_{i}" in self._samples:
                i += 1
            sample_id = f"{base_id}_{i}"

        color = self._color_cycle[self._color_idx % len(self._color_cycle)]
        self._color_idx += 1

        style = PlotStyle(
            color=color,
            marker="o",
            linestyle="-",
            label=sample_id,
        )
        self._samples[sample_id] = PlotSample(
            sample_id=sample_id,
            data=data,
            style=style,
            fitting_result=fitting_result,
        )
        self._refresh()
        return sample_id

    def remove_sample(self, sample_id: str):
        if sample_id in self._samples:
            del self._samples[sample_id]
            self._refresh()

    def clear_samples(self):
        self._samples.clear()
        self._color_idx = 0
        self._refresh()

    def update_fitting(self, sample_id: str, fitting_result: FittingResult):
        if sample_id in self._samples:
            self._samples[sample_id].fitting_result = fitting_result
            self._refresh()

    def set_show_fit(self, show: bool):
        self._show_fit = show
        self._refresh()

    def set_axis_limits(self, nyquist_x: Optional[Tuple[float, float]] = None,
                        nyquist_y: Optional[Tuple[float, float]] = None,
                        bode_x: Optional[Tuple[float, float]] = None,
                        bode_mag_y: Optional[Tuple[float, float]] = None,
                        bode_phase_y: Optional[Tuple[float, float]] = None):
        if nyquist_x:
            self.nyquist_canvas.set_xlim(*nyquist_x)
        if nyquist_y:
            self.nyquist_canvas.set_ylim(*nyquist_y)
        if bode_x:
            self.bode_canvas.set_xlim(*bode_x)
        if bode_mag_y:
            self.bode_canvas.set_ylim_mag(*bode_mag_y)
        if bode_phase_y:
            self.bode_canvas.set_ylim_phase(*bode_phase_y)
        self.nyquist_canvas.draw()
        self.bode_canvas.draw()

    def auto_scale(self):
        samples = list(self._samples.values())
        self.nyquist_canvas.auto_scale(samples)
        self.bode_canvas.auto_scale(samples)
        self.nyquist_canvas.draw()
        self.bode_canvas.draw()

    def _refresh(self):
        self.nyquist_canvas.clear()
        self.bode_canvas.clear()
        for sample in self._samples.values():
            self.nyquist_canvas.plot_sample(sample, self._show_fit)
            self.bode_canvas.plot_sample(sample, self._show_fit)
        self.nyquist_canvas.draw()
        self.bode_canvas.draw()

    def get_sample_ids(self) -> List[str]:
        return list(self._samples.keys())

    def save_figure(self, file_path: str, dpi: int = 300):
        combined_fig = Figure(figsize=(12, 8), dpi=dpi, tight_layout=True)
        gs = GridSpec(2, 2, figure=combined_fig, width_ratios=[1, 1], height_ratios=[1, 1],
                      wspace=0.3, hspace=0.3)

        ax_nyq = combined_fig.add_subplot(gs[:, 0])
        ax_mag = combined_fig.add_subplot(gs[0, 1])
        ax_phase = combined_fig.add_subplot(gs[1, 1], sharex=ax_mag)

        ax_nyq.set_xlabel("Z' (Ω)", fontsize=12, fontfamily="SimHei")
        ax_nyq.set_ylabel("-Z\" (Ω)", fontsize=12, fontfamily="SimHei")
        ax_nyq.set_title("奈奎斯特图", fontsize=13, fontfamily="SimHei")
        ax_nyq.grid(True, linestyle="--", alpha=0.6)

        ax_mag.set_xscale("log")
        ax_mag.set_yscale("log")
        ax_mag.set_ylabel("|Z| (Ω)", fontsize=12, fontfamily="SimHei")
        ax_mag.set_title("波特图 - 幅频", fontsize=13, fontfamily="SimHei")
        ax_mag.grid(True, which="both", linestyle="--", alpha=0.6)

        ax_phase.set_xscale("log")
        ax_phase.set_xlabel("频率 f (Hz)", fontsize=12, fontfamily="SimHei")
        ax_phase.set_ylabel("相位 φ (°)", fontsize=12, fontfamily="SimHei")
        ax_phase.grid(True, which="both", linestyle="--", alpha=0.6)

        samples = list(self._samples.values())
        for s in samples:
            if not s.data.is_valid:
                continue
            ax_nyq.plot(s.data.z_real, -s.data.z_imag, color=s.style.color,
                        marker=s.style.marker, linestyle=s.style.linestyle,
                        linewidth=s.style.linewidth, markersize=s.style.markersize,
                        label=s.style.label)
            ax_mag.plot(s.data.frequencies, s.data.z_magnitude, color=s.style.color,
                        marker=s.style.marker, linestyle=s.style.linestyle,
                        linewidth=s.style.linewidth, markersize=s.style.markersize,
                        label=s.style.label)
            ax_phase.plot(s.data.frequencies, s.data.z_phase, color=s.style.color,
                          marker=s.style.marker, linestyle=s.style.linestyle,
                          linewidth=s.style.linewidth, markersize=s.style.markersize)

            if self._show_fit and s.fitting_result and s.fitting_result.success:
                fit_z = s.fitting_result.fitted_z_real + 1j * s.fitting_result.fitted_z_imag
                ax_nyq.plot(s.fitting_result.fitted_z_real, -s.fitting_result.fitted_z_imag,
                            color=s.style.color, linestyle="--", linewidth=s.style.linewidth * 0.8, alpha=0.7)
                ax_mag.plot(s.fitting_result.fitted_freq, np.abs(fit_z),
                            color=s.style.color, linestyle="--", linewidth=s.style.linewidth * 0.8, alpha=0.7)
                ax_phase.plot(s.fitting_result.fitted_freq, np.degrees(np.arctan2(np.imag(fit_z), np.real(fit_z))),
                              color=s.style.color, linestyle="--", linewidth=s.style.linewidth * 0.8, alpha=0.7)

        ax_nyq.legend(loc="best", fontsize=9, prop={"family": "SimHei"})
        ax_mag.legend(loc="best", fontsize=9, prop={"family": "SimHei"})

        if samples:
            all_real = np.concatenate([s.data.z_real for s in samples if s.data.is_valid])
            all_imag = np.concatenate([-s.data.z_imag for s in samples if s.data.is_valid])
            margin = 0.1
            if len(all_real) > 0 and len(all_imag) > 0:
                dx = all_real.max() - all_real.min()
                dy = all_imag.max() - all_imag.min()
                d = max(dx, dy) * margin
                ax_nyq.set_xlim(all_real.min() - d, all_real.max() + d)
                ax_nyq.set_ylim(all_imag.min() - d, all_imag.max() + d)
                ax_nyq.set_aspect("equal", adjustable="datalim")

            all_freq = np.concatenate([s.data.frequencies for s in samples if s.data.is_valid])
            all_mag = np.concatenate([s.data.z_magnitude for s in samples if s.data.is_valid])
            all_phase = np.concatenate([s.data.z_phase for s in samples if s.data.is_valid])

            log_margin = 0.15
            fmin, fmax = float(np.nanmin(all_freq)), float(np.nanmax(all_freq))
            if fmin > 0 and fmax > fmin:
                ax_mag.set_xlim(fmin / (1.0 + log_margin), fmax * (1.0 + log_margin))

            m_valid = all_mag[all_mag > 0]
            if len(m_valid) > 0:
                mmin, mmax = float(np.nanmin(m_valid)), float(np.nanmax(m_valid))
                if mmax > mmin > 0:
                    ax_mag.set_ylim(mmin / (1.0 + log_margin), mmax * (1.0 + log_margin))

            p_valid = all_phase[~np.isnan(all_phase)]
            if len(p_valid) > 0:
                pmin, pmax = float(np.nanmin(p_valid)), float(np.nanmax(p_valid))
                dp = pmax - pmin
                if dp < 0.5:
                    center = (pmin + pmax) * 0.5
                    pmin_set, pmax_set = center - 5.0, center + 5.0
                else:
                    pad = max(dp * 0.15, 2.0)
                    pmin_set, pmax_set = pmin - pad, pmax + pad
                pmin_set = max(pmin_set, -180.0)
                pmax_set = min(pmax_set, 180.0)
                ax_phase.set_ylim(pmin_set, pmax_set)

        combined_fig.savefig(file_path, dpi=dpi, bbox_inches="tight")
