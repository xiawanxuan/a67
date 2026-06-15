from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class CircuitTemplate:
    key: str
    name: str
    description: str
    circuit_expr: str
    parameters: List[str]
    initial_guess: List[float]
    lower_bounds: List[float]
    upper_bounds: List[float]

    def __post_init__(self):
        if len(self.parameters) != len(self.initial_guess):
            raise ValueError("参数名与初始值数量不匹配")
        if len(self.parameters) != len(self.lower_bounds):
            raise ValueError("参数名与下界数量不匹配")
        if len(self.parameters) != len(self.upper_bounds):
            raise ValueError("参数名与上界数量不匹配")

    @property
    def param_count(self) -> int:
        return len(self.parameters)

    @property
    def bounds(self) -> Tuple[List[float], List[float]]:
        return (self.lower_bounds, self.upper_bounds)

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "name": self.name,
            "description": self.description,
            "circuit_expr": self.circuit_expr,
            "parameters": self.parameters,
            "initial_guess": self.initial_guess,
            "bounds": [self.lower_bounds, self.upper_bounds],
        }

    @classmethod
    def from_dict(cls, key: str, data: dict) -> "CircuitTemplate":
        bounds = data.get("bounds", [[], []])
        return cls(
            key=key,
            name=data.get("name", key),
            description=data.get("description", ""),
            circuit_expr=data.get("circuit_expr", ""),
            parameters=data.get("parameters", []),
            initial_guess=data.get("initial_guess", []),
            lower_bounds=bounds[0] if len(bounds) > 0 else [],
            upper_bounds=bounds[1] if len(bounds) > 1 else [],
        )


class CircuitTemplateManager:
    """等效电路模板管理器"""

    def __init__(self, config_path: Optional[str] = None):
        self._templates: Dict[str, CircuitTemplate] = {}
        self.default_key: str = ""
        if config_path and os.path.exists(config_path):
            self.load_from_file(config_path)
        else:
            self._load_defaults()

    def _load_defaults(self):
        defaults = {
            "R_s": {
                "name": "溶液电阻",
                "description": "单电阻模型",
                "circuit_expr": "R_s",
                "parameters": ["R_s"],
                "initial_guess": [10.0],
                "bounds": [[0.0], [1e6]],
            },
            "R_s-R_para-C_para": {
                "name": "RC并联模型",
                "description": "溶液电阻串联RC并联",
                "circuit_expr": "R_s + R_para/(1+j*w*C_para*R_para)",
                "parameters": ["R_s", "R_para", "C_para"],
                "initial_guess": [10.0, 100.0, 1e-5],
                "bounds": [[0.0, 0.0, 1e-12], [1e6, 1e8, 1.0]],
            },
            "R_s-R_para-CPE": {
                "name": "RQ并联模型",
                "description": "溶液电阻串联RQ并联",
                "circuit_expr": "R_s + R_para/(1+R_para*Y0*(j*w)^n)",
                "parameters": ["R_s", "R_para", "Y0", "n"],
                "initial_guess": [10.0, 100.0, 1e-5, 0.9],
                "bounds": [[0.0, 0.0, 1e-12, 0.5], [1e6, 1e8, 1.0, 1.0]],
            },
            "R_s-R_para-C_para-W": {
                "name": "Randles模型",
                "description": "含Warburg扩散阻抗",
                "circuit_expr": "R_s + R_para/(1+j*w*C_para*R_para) + W*sqrt(w)*(1-j)",
                "parameters": ["R_s", "R_para", "C_para", "W"],
                "initial_guess": [10.0, 100.0, 1e-5, 50.0],
                "bounds": [[0.0, 0.0, 1e-12, 0.0], [1e6, 1e8, 1.0, 1e4]],
            },
            "R_s-R1-C1-R2-C2": {
                "name": "双时间常数模型",
                "description": "两段RC并联串联",
                "circuit_expr": "R_s + R1/(1+j*w*C1*R1) + R2/(1+j*w*C2*R2)",
                "parameters": ["R_s", "R1", "C1", "R2", "C2"],
                "initial_guess": [10.0, 100.0, 1e-5, 200.0, 1e-6],
                "bounds": [[0.0, 0.0, 1e-12, 0.0, 1e-12], [1e6, 1e8, 1.0, 1e8, 1.0]],
            },
        }
        for key, data in defaults.items():
            self._templates[key] = CircuitTemplate.from_dict(key, data)
        self.default_key = "R_s-R_para-C_para"

    def load_from_file(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for key, tpl in data.get("templates", {}).items():
            self._templates[key] = CircuitTemplate.from_dict(key, tpl)
        self.default_key = data.get("default_template", next(iter(self._templates.keys())))

    def save_to_file(self, path: str):
        out = {"templates": {}, "default_template": self.default_key}
        for key, tpl in self._templates.items():
            out["templates"][key] = {
                "name": tpl.name,
                "description": tpl.description,
                "circuit_expr": tpl.circuit_expr,
                "parameters": tpl.parameters,
                "initial_guess": tpl.initial_guess,
                "bounds": [tpl.lower_bounds, tpl.upper_bounds],
            }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

    def get_template(self, key: str) -> Optional[CircuitTemplate]:
        return self._templates.get(key)

    def list_templates(self) -> List[CircuitTemplate]:
        return list(self._templates.values())

    def add_template(self, template: CircuitTemplate):
        self._templates[template.key] = template

    def remove_template(self, key: str):
        if key in self._templates:
            del self._templates[key]
            if self.default_key == key and self._templates:
                self.default_key = next(iter(self._templates.keys()))
