from __future__ import annotations

import csv
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..parsers.base_parser import EISData
from ..fitting.fitting_engine import FittingResult
from ..plotting.dual_canvas import DualPlotCanvas


class DataExporter:
    """批量实验数据导出器"""

    @staticmethod
    def export_single_csv(file_path: str, data: EISData,
                          fitting_result: Optional[FittingResult] = None) -> bool:
        """导出单个样品的 EIS 数据到 CSV 文件"""
        try:
            with open(file_path, "w", encoding="utf-8-sig", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["样品ID:", data.sample_id])
                writer.writerow(["批次ID:", data.batch_id])
                writer.writerow(["设备类型:", data.device_type.value])
                writer.writerow(["测试温度:", f"{data.temperature} °C" if data.temperature else ""])
                writer.writerow(["测试日期:", data.test_date or ""])
                writer.writerow(["原始文件:", data.file_path])
                writer.writerow([])

                if fitting_result:
                    writer.writerow(["=== 拟合结果 ==="])
                    writer.writerow(["等效电路:", fitting_result.template_key])
                    writer.writerow(["拟合成功:", "是" if fitting_result.success else "否"])
                    writer.writerow(["χ²:", f"{fitting_result.chi_squared:.6e}"])
                    writer.writerow(["元件参数:"])
                    for name, value in fitting_result.parameters.items():
                        writer.writerow([f"  {name}", f"{value:.6e}"])
                    writer.writerow([])

                writer.writerow(["=== 原始数据 ==="])
                headers = ["频率 (Hz)", "Z' (Ω)", "-Z\" (Ω)", "|Z| (Ω)", "相位 (°)"]
                if fitting_result and fitting_result.success:
                    headers += ["拟合 Z' (Ω)", "拟合 -Z\" (Ω)"]
                writer.writerow(headers)

                z_imag_neg = -data.z_imag
                mag = data.z_magnitude
                phase = data.z_phase

                for i in range(len(data.frequencies)):
                    row = [
                        f"{data.frequencies[i]:.6e}",
                        f"{data.z_real[i]:.6e}",
                        f"{z_imag_neg[i]:.6e}",
                        f"{mag[i]:.6e}",
                        f"{phase[i]:.4f}",
                    ]
                    if fitting_result and fitting_result.success:
                        if i < len(fitting_result.fitted_z_real):
                            fit_z_imag_neg = -fitting_result.fitted_z_imag[i]
                            row += [
                                f"{fitting_result.fitted_z_real[i]:.6e}",
                                f"{fit_z_imag_neg:.6e}",
                            ]
                        else:
                            row += ["", ""]
                    writer.writerow(row)
            return True
        except Exception as e:
            print(f"导出CSV失败: {e}")
            return False

    @staticmethod
    def export_batch_csv(file_path: str,
                         samples: List[Tuple[EISData, Optional[FittingResult]]]) -> bool:
        """批量导出多个样品到单个 CSV 文件"""
        try:
            with open(file_path, "w", encoding="utf-8-sig", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["样品ID", "批次ID", "设备类型", "频率 (Hz)",
                                 "Z' (Ω)", "-Z\" (Ω)", "|Z| (Ω)", "相位 (°)",
                                 "拟合 Z' (Ω)", "拟合 -Z\" (Ω)", "等效电路",
                                 "拟合成功", "χ²"])

                for data, fit in samples:
                    z_imag_neg = -data.z_imag
                    mag = data.z_magnitude
                    phase = data.z_phase

                    template_key = fit.template_key if fit else ""
                    fit_success = "是" if (fit and fit.success) else "否"
                    chi_sq = f"{fit.chi_squared:.6e}" if fit else ""

                    for i in range(len(data.frequencies)):
                        fit_zr = ""
                        fit_zi = ""
                        if fit and fit.success and i < len(fit.fitted_z_real):
                            fit_zr = f"{fit.fitted_z_real[i]:.6e}"
                            fit_zi = f"{-fit.fitted_z_imag[i]:.6e}"

                        writer.writerow([
                            data.sample_id, data.batch_id, data.device_type.value,
                            f"{data.frequencies[i]:.6e}",
                            f"{data.z_real[i]:.6e}",
                            f"{z_imag_neg[i]:.6e}",
                            f"{mag[i]:.6e}",
                            f"{phase[i]:.4f}",
                            fit_zr, fit_zi,
                            template_key, fit_success, chi_sq,
                        ])
            return True
        except Exception as e:
            print(f"批量导出CSV失败: {e}")
            return False

    @staticmethod
    def export_fitting_params_csv(file_path: str,
                                  samples: List[Tuple[EISData, Optional[FittingResult]]]) -> bool:
        """导出所有样品的拟合参数汇总表"""
        try:
            all_params: List[str] = []
            for _, fit in samples:
                if fit and fit.success:
                    for p in fit.parameters.keys():
                        if p not in all_params:
                            all_params.append(p)

            with open(file_path, "w", encoding="utf-8-sig", newline="") as f:
                writer = csv.writer(f)
                headers = ["样品ID", "批次ID", "设备类型", "等效电路", "拟合成功", "χ²"] + all_params
                writer.writerow(headers)

                for data, fit in samples:
                    row = [
                        data.sample_id,
                        data.batch_id,
                        data.device_type.value,
                        fit.template_key if fit else "",
                        "是" if (fit and fit.success) else "否",
                        f"{fit.chi_squared:.6e}" if fit else "",
                    ]
                    if fit and fit.success:
                        for p in all_params:
                            val = fit.parameters.get(p, "")
                            row.append(f"{val:.6e}" if isinstance(val, (int, float)) else val)
                    else:
                        row.extend([""] * len(all_params))
                    writer.writerow(row)
            return True
        except Exception as e:
            print(f"导出拟合参数失败: {e}")
            return False

    @staticmethod
    def export_json(file_path: str, data: EISData,
                    fitting_result: Optional[FittingResult] = None) -> bool:
        """导出单个样品数据为 JSON 格式"""
        try:
            out = data.to_dict()
            if fitting_result:
                out["fitting"] = fitting_result.to_dict()
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"导出JSON失败: {e}")
            return False

    @staticmethod
    def export_excel(file_path: str,
                     samples: List[Tuple[EISData, Optional[FittingResult]]]) -> bool:
        """导出到 Excel，包含原始数据和拟合参数两个工作表"""
        try:
            raw_rows = []
            param_rows = []
            all_params: List[str] = []
            for _, fit in samples:
                if fit and fit.success:
                    for p in fit.parameters.keys():
                        if p not in all_params:
                            all_params.append(p)

            for data, fit in samples:
                z_imag_neg = -data.z_imag
                mag = data.z_magnitude
                phase = data.z_phase

                template_key = fit.template_key if fit else ""
                fit_success = "是" if (fit and fit.success) else "否"
                chi_sq = fit.chi_squared if fit else None

                for i in range(len(data.frequencies)):
                    fit_zr = None
                    fit_zi = None
                    if fit and fit.success and i < len(fit.fitted_z_real):
                        fit_zr = fit.fitted_z_real[i]
                        fit_zi = -fit.fitted_z_imag[i]
                    raw_rows.append({
                        "样品ID": data.sample_id,
                        "批次ID": data.batch_id,
                        "设备类型": data.device_type.value,
                        "频率 (Hz)": data.frequencies[i],
                        "Z' (Ω)": data.z_real[i],
                        "-Z\" (Ω)": z_imag_neg[i],
                        "|Z| (Ω)": mag[i],
                        "相位 (°)": phase[i],
                        "拟合 Z' (Ω)": fit_zr,
                        "拟合 -Z\" (Ω)": fit_zi,
                        "等效电路": template_key,
                        "拟合成功": fit_success,
                        "χ²": chi_sq,
                    })

                param_row = {
                    "样品ID": data.sample_id,
                    "批次ID": data.batch_id,
                    "设备类型": data.device_type.value,
                    "等效电路": template_key,
                    "拟合成功": fit_success,
                    "χ²": chi_sq,
                }
                if fit and fit.success:
                    for p in all_params:
                        param_row[p] = fit.parameters.get(p)
                param_rows.append(param_row)

            df_raw = pd.DataFrame(raw_rows)
            df_params = pd.DataFrame(param_rows)

            with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
                df_raw.to_excel(writer, sheet_name="原始数据", index=False)
                df_params.to_excel(writer, sheet_name="拟合参数", index=False)
            return True
        except ImportError:
            print("需要安装 openpyxl 才能导出 Excel")
            return False
        except Exception as e:
            print(f"导出Excel失败: {e}")
            return False

    @staticmethod
    def export_figure(canvas: DualPlotCanvas, file_path: str, dpi: int = 300) -> bool:
        """导出绘图为图片"""
        try:
            ext = os.path.splitext(file_path)[1].lower()
            if ext in (".png", ".jpg", ".jpeg", ".svg", ".pdf", ".eps"):
                canvas.save_figure(file_path, dpi=dpi)
                return True
            else:
                print(f"不支持的图片格式: {ext}")
                return False
        except Exception as e:
            print(f"导出图片失败: {e}")
            return False
