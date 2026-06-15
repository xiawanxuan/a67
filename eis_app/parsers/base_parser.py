from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional
import numpy as np


class DeviceType(Enum):
    AutoLab = "AutoLab"
    Gamry = "Gamry"
    Solartron = "Solartron"
    CHI = "CHI"
    Unknown = "Unknown"


@dataclass
class EISData:
    frequencies: np.ndarray = field(default_factory=lambda: np.array([]))
    z_real: np.ndarray = field(default_factory=lambda: np.array([]))
    z_imag: np.ndarray = field(default_factory=lambda: np.array([]))
    device_type: DeviceType = DeviceType.Unknown
    sample_id: str = ""
    batch_id: str = ""
    file_path: str = ""
    temperature: Optional[float] = None
    test_date: Optional[str] = None
    notes: str = ""

    @property
    def z_magnitude(self) -> np.ndarray:
        return np.sqrt(self.z_real ** 2 + self.z_imag ** 2)

    @property
    def z_phase(self) -> np.ndarray:
        return np.degrees(np.arctan2(self.z_imag, self.z_real))

    @property
    def is_valid(self) -> bool:
        return (len(self.frequencies) > 0 and
                len(self.frequencies) == len(self.z_real) == len(self.z_imag))

    def validate(self) -> bool:
        if not self.is_valid:
            return False
        if np.any(self.frequencies <= 0):
            return False
        return True

    def to_dict(self) -> dict:
        return {
            "frequencies": self.frequencies.tolist(),
            "z_real": self.z_real.tolist(),
            "z_imag": self.z_imag.tolist(),
            "device_type": self.device_type.value,
            "sample_id": self.sample_id,
            "batch_id": self.batch_id,
            "file_path": self.file_path,
            "temperature": self.temperature,
            "test_date": self.test_date,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EISData":
        return cls(
            frequencies=np.array(data.get("frequencies", [])),
            z_real=np.array(data.get("z_real", [])),
            z_imag=np.array(data.get("z_imag", [])),
            device_type=DeviceType(data.get("device_type", "Unknown")),
            sample_id=data.get("sample_id", ""),
            batch_id=data.get("batch_id", ""),
            file_path=data.get("file_path", ""),
            temperature=data.get("temperature"),
            test_date=data.get("test_date"),
            notes=data.get("notes", ""),
        )
