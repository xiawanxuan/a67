from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

from ..parsers.base_parser import EISData
from ..parsers.eis_parser import EISParser
from ..fitting.circuit_template import CircuitTemplate, CircuitTemplateManager
from ..fitting.fitting_engine import FittingEngine, FittingResult
from ..export.data_exporter import DataExporter


@dataclass
class BatchProcessingOptions:
    template_key: str = "R_s-R_para-C_para"
    recursive: bool = True
    file_patterns: List[str] = field(
        default_factory=lambda: ["*.txt", "*.csv", "*.dta", "*.nox", "*.z", "*.zsimpwin"]
    )
    batch_id: str = ""
    auto_save_db: bool = True
    export_excel: bool = True
    export_figures: bool = True
    show_individual_plots: bool = False
    figure_dpi: int = 300
    fitting_method: str = "leastsq"
    max_retries: int = 2


@dataclass
class BatchFileResult:
    file_path: str
    file_name: str
    success: bool
    data: Optional[EISData] = None
    fitting_result: Optional[FittingResult] = None
    error_message: str = ""
    processing_time: float = 0.0


@dataclass
class BatchResult:
    total_files: int = 0
    success_count: int = 0
    failed_count: int = 0
    results: List[BatchFileResult] = field(default_factory=list)
    output_dir: str = ""
    excel_path: str = ""
    summary_figure_path: str = ""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    options: Optional[BatchProcessingOptions] = None

    @property
    def success_rate(self) -> float:
        if self.total_files == 0:
            return 0.0
        return self.success_count / self.total_files * 100

    @property
    def duration_seconds(self) -> float:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0


class BatchProcessor:
    """文件夹批量处理引擎 - 遍历目录内全部阻抗文件自动拟合、批量导出"""

    def __init__(self, template_manager: Optional[CircuitTemplateManager] = None,
                 db_manager=None):
        self.parser = EISParser()
        self.fitting_engine = FittingEngine()
        self.template_manager = template_manager or CircuitTemplateManager()
        self.db_manager = db_manager
        self.exporter = DataExporter()
        self._cancel_flag = False
        self._progress_callback: Optional[Callable[[int, int, str], None]] = None

    def set_progress_callback(self, callback: Callable[[int, int, str], None]):
        self._progress_callback = callback

    def cancel(self):
        self._cancel_flag = True

    def _notify_progress(self, current: int, total: int, message: str):
        if self._progress_callback:
            self._progress_callback(current, total, message)

    def scan_directory(self, directory: str, options: BatchProcessingOptions) -> List[str]:
        """扫描目录，获取所有匹配的 EIS 数据文件"""
        matched_files: List[str] = []
        if not os.path.isdir(directory):
            return matched_files

        regex_patterns = [self._glob_to_regex(p) for p in options.file_patterns]

        if options.recursive:
            for root, _, files in os.walk(directory):
                for fname in files:
                    if self._matches_patterns(fname, regex_patterns):
                        matched_files.append(os.path.join(root, fname))
        else:
            for fname in os.listdir(directory):
                fpath = os.path.join(directory, fname)
                if os.path.isfile(fpath) and self._matches_patterns(fname, regex_patterns):
                    matched_files.append(fpath)

        matched_files.sort()
        return matched_files

    @staticmethod
    def _glob_to_regex(glob_pattern: str) -> re.Pattern:
        regex = glob_pattern.replace(".", "\\.").replace("*", ".*").replace("?", ".")
        return re.compile(f"^{regex}$", re.IGNORECASE)

    @staticmethod
    def _matches_patterns(filename: str, patterns: List[re.Pattern]) -> bool:
        return any(p.match(filename) for p in patterns)

    def process_directory(self, directory: str,
                          options: Optional[BatchProcessingOptions] = None) -> BatchResult:
        """批量处理指定目录"""
        options = options or BatchProcessingOptions()
        result = BatchResult(
            start_time=datetime.now(),
            options=options,
        )

        files = self.scan_directory(directory, options)
        result.total_files = len(files)

        if result.total_files == 0:
            result.end_time = datetime.now()
            return result

        if options.batch_id:
            batch_id = options.batch_id
        else:
            batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result.output_dir = os.path.join(directory, f"eis_batch_output_{timestamp}")
        os.makedirs(result.output_dir, exist_ok=True)

        template = self.template_manager.get_template(options.template_key)
        if not template:
            template = self.template_manager.get_template(
                self.template_manager.default_key
            )

        samples_for_export: List[Tuple[EISData, Optional[FittingResult]]] = []
        individual_results: List[Tuple[EISData, Optional[FittingResult], str]] = []

        for idx, fpath in enumerate(files, 1):
            if self._cancel_flag:
                break

            file_name = os.path.basename(fpath)
            self._notify_progress(idx, result.total_files, f"处理中: {file_name}")

            file_result = BatchFileResult(file_path=fpath, file_name=file_name)
            start_t = datetime.now()

            try:
                data = self.parser.parse_file(fpath, batch_id=batch_id)
                if not data.is_valid:
                    file_result.success = False
                    file_result.error_message = "未解析到有效数据"
                else:
                    file_result.data = data
                    fit_res = self._try_fit(data, template, options)
                    file_result.fitting_result = fit_res
                    file_result.success = fit_res.success
                    if not fit_res.success:
                        file_result.error_message = fit_res.message

                    samples_for_export.append((data, fit_res))
                    individual_results.append((data, fit_res, file_name))

                    if options.auto_save_db and self.db_manager:
                        db_id = self.db_manager.add_sample(data)
                        if fit_res.success:
                            self.db_manager.add_fitting_result(db_id, fit_res)

                    if options.export_figures and options.show_individual_plots and data.is_valid:
                        self._export_individual_figure(
                            data, fit_res, file_name, result.output_dir, options, template
                        )

            except Exception as e:
                file_result.success = False
                file_result.error_message = f"异常: {str(e)}"

            file_result.processing_time = (datetime.now() - start_t).total_seconds()
            result.results.append(file_result)
            if file_result.success:
                result.success_count += 1
            else:
                result.failed_count += 1

        if not self._cancel_flag:
            if options.export_excel and samples_for_export:
                excel_name = f"fitting_summary_{timestamp}.xlsx"
                result.excel_path = os.path.join(result.output_dir, excel_name)
                self.exporter.export_excel(result.excel_path, samples_for_export)

                csv_name = f"fitting_params_{timestamp}.csv"
                csv_path = os.path.join(result.output_dir, csv_name)
                self.exporter.export_fitting_params_csv(csv_path, samples_for_export)

            if options.export_figures and individual_results:
                fig_name = f"comparison_{timestamp}.png"
                result.summary_figure_path = os.path.join(result.output_dir, fig_name)
                self._generate_comparison_figure(
                    individual_results, result.summary_figure_path,
                    options, template.name if template else ""
                )

            self._generate_summary_report(result, individual_results, template)

        result.end_time = datetime.now()
        self._notify_progress(result.total_files, result.total_files,
                              f"完成: {result.success_count}/{result.total_files} 成功")
        return result

    def _try_fit(self, data: EISData, template: CircuitTemplate,
                 options: BatchProcessingOptions) -> FittingResult:
        result = self.fitting_engine.fit(
            template, data, method=options.fitting_method
        )
        if not result.success and options.max_retries > 0:
            for attempt in range(options.max_retries):
                if self._cancel_flag:
                    break
                perturbed_guess = self._perturb_initial_guess(template, attempt + 1)
                result = self.fitting_engine.fit(
                    template, data, initial_guess=perturbed_guess,
                    method=options.fitting_method
                )
                if result.success:
                    break
        return result

    @staticmethod
    def _perturb_initial_guess(template: CircuitTemplate, seed: int) -> List[float]:
        rng = np.random.default_rng(seed=seed * 42)
        guess = []
        for i, val in enumerate(template.initial_guess):
            lb = template.lower_bounds[i]
            ub = template.upper_bounds[i]
            if val <= 0:
                factor = 1.0
            else:
                factor = rng.uniform(0.5, 2.0)
            new_val = val * factor
            new_val = max(lb, min(ub, new_val))
            guess.append(new_val)
        return guess

    def _export_individual_figure(self, data: EISData, fit_result: FittingResult,
                                  file_name: str, output_dir: str,
                                  options: BatchProcessingOptions,
                                  template: CircuitTemplate):
        from ..plotting.dual_canvas import DualPlotCanvas, PlotStyle, PlotSample

        fig = Figure(figsize=(14, 9), dpi=options.figure_dpi, tight_layout=True)
        gs = GridSpec(2, 2, figure=fig, width_ratios=[1, 1], height_ratios=[1, 1],
                      wspace=0.3, hspace=0.3)

        ax_nyq = fig.add_subplot(gs[:, 0])
        ax_mag = fig.add_subplot(gs[0, 1])
        ax_phase = fig.add_subplot(gs[1, 1], sharex=ax_mag)

        ax_nyq.set_xlabel("Z' (Ω)", fontsize=12, fontfamily="SimHei")
        ax_nyq.set_ylabel("-Z\" (Ω)", fontsize=12, fontfamily="SimHei")
        ax_nyq.set_title(f"奈奎斯特图 - {file_name}", fontsize=13, fontfamily="SimHei")
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

        z_imag_pos = -data.z_imag
        ax_nyq.plot(data.z_real, z_imag_pos, color="#1f77b4", marker="o",
                    linestyle="-", linewidth=1.5, markersize=4, label="实验数据")
        ax_mag.plot(data.frequencies, data.z_magnitude, color="#1f77b4", marker="o",
                    linestyle="-", linewidth=1.5, markersize=4, label="实验数据")
        ax_phase.plot(data.frequencies, data.z_phase, color="#1f77b4", marker="o",
                      linestyle="-", linewidth=1.5, markersize=4)

        if fit_result.success:
            fit_z_imag_pos = -fit_result.fitted_z_imag
            ax_nyq.plot(fit_result.fitted_z_real, fit_z_imag_pos,
                        color="#d62728", linestyle="--", linewidth=2, label="拟合曲线")
            fit_z = fit_result.fitted_z_real + 1j * fit_result.fitted_z_imag
            ax_mag.plot(fit_result.fitted_freq, np.abs(fit_z),
                        color="#d62728", linestyle="--", linewidth=2)
            ax_phase.plot(fit_result.fitted_freq, np.degrees(np.arctan2(np.imag(fit_z), np.real(fit_z))),
                          color="#d62728", linestyle="--", linewidth=2)

            fit_text = f"拟合成功\nχ² = {fit_result.chi_squared:.4e}\n"
            for name, val in fit_result.parameters.items():
                fit_text += f"{name} = {val:.4e}\n"
            ax_nyq.text(0.98, 0.02, fit_text.strip(),
                        transform=ax_nyq.transAxes, ha="right", va="bottom",
                        fontsize=10, fontfamily="SimHei",
                        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7))

        ax_nyq.legend(loc="best", fontsize=10, prop={"family": "SimHei"})
        ax_mag.legend(loc="best", fontsize=10, prop={"family": "SimHei"})

        self._auto_scale_bode_axes([(data, fit_result)], ax_mag, ax_phase)
        self._auto_scale_nyquist_axis([data], ax_nyq)

        base_name = os.path.splitext(file_name)[0]
        safe_name = re.sub(r'[<>:"/\\|?*]', "_", base_name)
        out_path = os.path.join(output_dir, f"{safe_name}_plot.png")
        fig.savefig(out_path, dpi=options.figure_dpi, bbox_inches="tight")

    def _generate_comparison_figure(self, results: List[Tuple[EISData, FittingResult, str]],
                                     output_path: str, options: BatchProcessingOptions,
                                     template_name: str):
        valid_results = [r for r in results if r[0].is_valid]
        if not valid_results:
            return

        n = len(valid_results)
        ncols = 3
        nrows = (n + ncols - 1) // ncols
        fig_w = min(18, 6 * ncols)
        fig_h = max(5, 4 * nrows)

        fig = Figure(figsize=(fig_w, fig_h), dpi=options.figure_dpi, tight_layout=True)
        gs = GridSpec(nrows, ncols, figure=fig, wspace=0.3, hspace=0.35)

        colors = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        ]

        for idx, (data, fit_result, fname) in enumerate(valid_results):
            row = idx // ncols
            col = idx % ncols
            ax = fig.add_subplot(gs[row, col])

            z_imag_pos = -data.z_imag
            color = colors[idx % len(colors)]
            ax.plot(data.z_real, z_imag_pos, color=color, marker="o",
                    linestyle="-", linewidth=1.2, markersize=3, alpha=0.8)

            if fit_result and fit_result.success:
                fit_z_imag_pos = -fit_result.fitted_z_imag
                ax.plot(fit_result.fitted_z_real, fit_z_imag_pos,
                        color=color, linestyle="--", linewidth=1.5, alpha=0.6)

            base_name = os.path.splitext(fname)[0]
            short_name = base_name[:20] + "..." if len(base_name) > 20 else base_name
            status = "✓" if (fit_result and fit_result.success) else "✗"
            ax.set_title(f"{status} {short_name}", fontsize=10, fontfamily="SimHei")
            ax.set_xlabel("Z' (Ω)", fontsize=8, fontfamily="SimHei")
            ax.set_ylabel("-Z\" (Ω)", fontsize=8, fontfamily="SimHei")
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.tick_params(axis="both", labelsize=7)
            ax.set_aspect("equal", adjustable="datalim")

        fig.suptitle(f"批量对比 - 奈奎斯特图 (共 {n} 个样品)\n等效电路: {template_name}",
                     fontsize=14, fontfamily="SimHei", y=1.005)
        fig.savefig(output_path, dpi=options.figure_dpi, bbox_inches="tight")

    def _generate_summary_report(self, batch_result: BatchResult,
                                  individual_results: List[Tuple[EISData, FittingResult, str]],
                                  template: CircuitTemplate):
        report_path = os.path.join(batch_result.output_dir, "processing_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("=" * 70 + "\n")
            f.write("EIS 阻抗谱批量处理报告\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"处理时间: {batch_result.start_time.strftime('%Y-%m-%d %H:%M:%S')} ~ "
                    f"{batch_result.end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"总耗时: {batch_result.duration_seconds:.2f} 秒\n")
            f.write(f"文件总数: {batch_result.total_files}\n")
            f.write(f"成功: {batch_result.success_count} ({batch_result.success_rate:.1f}%)\n")
            f.write(f"失败: {batch_result.failed_count}\n")
            f.write(f"等效电路: {template.name} ({template.key})\n")
            f.write(f"表达式: {template.circuit_expr}\n\n")

            f.write("-" * 70 + "\n")
            f.write("详细结果:\n")
            f.write("-" * 70 + "\n\n")

            for fr in batch_result.results:
                status = "✓ 成功" if fr.success else "✗ 失败"
                f.write(f"[{status}] {fr.file_name}\n")
                f.write(f"  耗时: {fr.processing_time:.3f}s\n")
                if fr.data and fr.data.is_valid:
                    f.write(f"  数据点数: {len(fr.data.frequencies)}\n")
                if fr.fitting_result and fr.fitting_result.success:
                    f.write(f"  χ² = {fr.fitting_result.chi_squared:.4e}\n")
                    for name, val in fr.fitting_result.parameters.items():
                        f.write(f"  {name} = {val:.6e}\n")
                if fr.error_message:
                    f.write(f"  错误: {fr.error_message}\n")
                f.write("\n")

            f.write("=" * 70 + "\n")
            f.write("报告结束\n")

    @staticmethod
    def _auto_scale_nyquist_axis(datas: List[EISData], ax):
        all_real = np.concatenate([d.z_real for d in datas if d.is_valid])
        all_imag = np.concatenate([-d.z_imag for d in datas if d.is_valid])
        if len(all_real) == 0:
            return
        margin = 0.1
        xmin, xmax = all_real.min(), all_real.max()
        ymin, ymax = all_imag.min(), all_imag.max()
        dx = xmax - xmin
        dy = ymax - ymin
        d = max(dx, dy) * margin
        ax.set_xlim(xmin - d, xmax + d)
        ax.set_ylim(ymin - d, ymax + d)
        ax.set_aspect("equal", adjustable="datalim")

    @staticmethod
    def _auto_scale_bode_axes(samples: List[Tuple[EISData, FittingResult]],
                               ax_mag, ax_phase):
        all_freq = np.concatenate([s[0].frequencies for s in samples if s[0].is_valid])
        all_mag = np.concatenate([s[0].z_magnitude for s in samples if s[0].is_valid])
        all_phase = np.concatenate([s[0].z_phase for s in samples if s[0].is_valid])

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
