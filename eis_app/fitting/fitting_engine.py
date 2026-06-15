from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import least_squares, differential_evolution

from ..parsers.base_parser import EISData
from .circuit_template import CircuitTemplate


@dataclass
class FittingResult:
    success: bool
    template_key: str
    parameters: Dict[str, float] = field(default_factory=dict)
    chi_squared: float = 0.0
    residuals: np.ndarray = field(default_factory=lambda: np.array([]))
    fitted_freq: np.ndarray = field(default_factory=lambda: np.array([]))
    fitted_z_real: np.ndarray = field(default_factory=lambda: np.array([]))
    fitted_z_imag: np.ndarray = field(default_factory=lambda: np.array([]))
    message: str = ""

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "template_key": self.template_key,
            "parameters": self.parameters,
            "chi_squared": self.chi_squared,
            "message": self.message,
        }


class FittingEngine:
    """EIS 阻抗谱非线性拟合引擎"""

    def __init__(self):
        self._j = 1j
        self._pi2 = 2.0 * np.pi

    def compute_impedance(self, template: CircuitTemplate, params: List[float],
                          frequencies: np.ndarray) -> np.ndarray:
        w = self._pi2 * frequencies
        param_dict = dict(zip(template.parameters, params))
        return self._evaluate_circuit(template.circuit_expr, w, param_dict)

    def _evaluate_circuit(self, expr: str, w: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        local_vars = {
            "w": w,
            "j": self._j,
            "pi": np.pi,
            "sqrt": np.sqrt,
            "exp": np.exp,
            "log": np.log,
            "sin": np.sin,
            "cos": np.cos,
        }
        local_vars.update(params)
        try:
            result = eval(expr, {"__builtins__": {}}, local_vars)
            if isinstance(result, (int, float, complex)):
                result = np.full_like(w, result, dtype=complex)
            return result
        except Exception as e:
            raise ValueError(f"等效电路表达式计算错误: {e}")

    def _residuals(self, params: List[float], template: CircuitTemplate,
                   frequencies: np.ndarray, z_exp: np.ndarray,
                   weight: str = "modulus") -> np.ndarray:
        z_fit = self.compute_impedance(template, params, frequencies)
        if weight == "modulus":
            w = 1.0 / (np.abs(z_exp) + 1e-15)
        elif weight == "real":
            w = 1.0 / (np.abs(z_exp.real) + 1e-15)
        elif weight == "unit":
            w = np.ones_like(z_exp)
        else:
            w = 1.0 / (np.abs(z_exp) + 1e-15)

        diff = z_fit - z_exp
        res_real = np.real(diff) * w
        res_imag = np.imag(diff) * w
        return np.concatenate([res_real, res_imag])

    def fit(self, template: CircuitTemplate, data: EISData,
            initial_guess: Optional[List[float]] = None,
            method: str = "leastsq", max_nfev: int = 20000,
            weight: str = "modulus") -> FittingResult:
        if not data.is_valid:
            return FittingResult(success=False, template_key=template.key,
                                 message="无效的 EIS 数据")

        if initial_guess is None:
            initial_guess = list(template.initial_guess)
        if len(initial_guess) != template.param_count:
            return FittingResult(success=False, template_key=template.key,
                                 message=f"初始值数量错误，应为 {template.param_count}")

        z_exp = data.z_real + 1j * data.z_imag
        freqs = data.frequencies

        try:
            if method == "de":
                result = self._fit_differential_evolution(template, freqs, z_exp, weight, max_nfev)
            else:
                result = self._fit_leastsq(template, freqs, z_exp, initial_guess, weight, max_nfev)

            if result.success:
                fitted_params = list(result.x)
                param_dict = dict(zip(template.parameters, fitted_params))
                z_fit = self.compute_impedance(template, fitted_params, freqs)
                chi_sq = float(np.sum(np.abs(z_fit - z_exp) ** 2 / (np.abs(z_exp) ** 2 + 1e-30)) / len(z_exp))
                return FittingResult(
                    success=True,
                    template_key=template.key,
                    parameters=param_dict,
                    chi_squared=chi_sq,
                    residuals=result.fun if hasattr(result, 'fun') else np.array([]),
                    fitted_freq=freqs,
                    fitted_z_real=np.real(z_fit),
                    fitted_z_imag=np.imag(z_fit),
                    message="拟合成功",
                )
            else:
                return FittingResult(
                    success=False,
                    template_key=template.key,
                    message=f"拟合未收敛: {getattr(result, 'message', '未知错误')}",
                )
        except Exception as e:
            return FittingResult(
                success=False,
                template_key=template.key,
                message=f"拟合异常: {str(e)}",
            )

    def _fit_leastsq(self, template: CircuitTemplate, freqs: np.ndarray, z_exp: np.ndarray,
                     initial_guess: List[float], weight: str, max_nfev: int):
        return least_squares(
            self._residuals,
            x0=initial_guess,
            args=(template, freqs, z_exp, weight),
            bounds=template.bounds,
            method="trf",
            max_nfev=max_nfev,
            ftol=1e-12,
            xtol=1e-12,
        )

    def _fit_differential_evolution(self, template: CircuitTemplate, freqs: np.ndarray,
                                    z_exp: np.ndarray, weight: str, max_nfev: int):
        def objective(params):
            res = self._residuals(params, template, freqs, z_exp, weight)
            return float(np.sum(res ** 2))

        bounds = list(zip(template.lower_bounds, template.upper_bounds))
        return differential_evolution(
            objective,
            bounds=bounds,
            maxiter=max_nfev // 100,
            tol=1e-10,
            popsize=15,
        )

    @staticmethod
    def format_param_value(name: str, value: float) -> str:
        if name.startswith("R") or name == "W":
            if value >= 1e6:
                return f"{value / 1e6:.4f} MΩ"
            elif value >= 1e3:
                return f"{value / 1e3:.4f} kΩ"
            elif value >= 1:
                return f"{value:.4f} Ω"
            else:
                return f"{value * 1e3:.4f} mΩ"
        elif name.startswith("C") or name.startswith("Y"):
            if value >= 1:
                return f"{value:.4e} F"
            elif value >= 1e-3:
                return f"{value * 1e3:.4f} mF"
            elif value >= 1e-6:
                return f"{value * 1e6:.4f} μF"
            elif value >= 1e-9:
                return f"{value * 1e9:.4f} nF"
            else:
                return f"{value * 1e12:.4f} pF"
        else:
            return f"{value:.6g}"
