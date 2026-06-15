from __future__ import annotations

import os
import re
from typing import Optional, Tuple
import numpy as np

from .base_parser import EISData, DeviceType


class EISParser:
    """EIS 阻抗谱数据解析器，支持 AutoLab、Gamry、Solartron、CHI 四类工作站格式"""

    @classmethod
    def parse_file(cls, file_path: str, sample_id: str = "", batch_id: str = "") -> EISData:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        device = cls._detect_device(content, file_path)
        frequencies, z_real, z_imag = cls._parse_content(content, device)

        data = EISData(
            frequencies=frequencies,
            z_real=z_real,
            z_imag=z_imag,
            device_type=device,
            sample_id=sample_id or os.path.splitext(os.path.basename(file_path))[0],
            batch_id=batch_id,
            file_path=file_path,
        )
        cls._extract_metadata(content, data)
        return data

    @classmethod
    def _detect_device(cls, content: str, file_path: str) -> DeviceType:
        content_lower = content.lower()
        ext = os.path.splitext(file_path)[1].lower()

        if "autolab" in content_lower or "nova" in content_lower or ext == ".nox":
            return DeviceType.AutoLab
        if "gamry" in content_lower or ext == ".dta":
            return DeviceType.Gamry
        if "solartron" in content_lower or "zplot" in content_lower or ext in (".z", ".zsimpwin"):
            return DeviceType.Solartron
        if "chi" in content_lower or "chenhua" in content_lower or ext == ".txt":
            header_lines = content.splitlines()[:20]
            for line in header_lines:
                if "Frequency/Hz" in line or "Z'/Ohm" in line:
                    return DeviceType.CHI
        return DeviceType.CHI

    @classmethod
    def _parse_content(cls, content: str, device: DeviceType) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if device == DeviceType.AutoLab:
            return cls._parse_autolab(content)
        elif device == DeviceType.Gamry:
            return cls._parse_gamry(content)
        elif device == DeviceType.Solartron:
            return cls._parse_solartron(content)
        elif device == DeviceType.CHI:
            return cls._parse_chi(content)
        else:
            return cls._parse_generic(content)

    @classmethod
    def _parse_autolab(cls, content: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        lines = content.splitlines()
        data_start = False
        freqs, reals, imags = [], [], []

        freq_col, real_col, imag_col = 0, 1, 2
        for i, line in enumerate(lines[:50]):
            if re.search(r"(freq|frequency)", line, re.IGNORECASE):
                headers = re.split(r"[\t,;]+", line.strip())
                for j, h in enumerate(headers):
                    hl = h.lower()
                    if "freq" in hl:
                        freq_col = j
                    elif re.search(r"(z['\"]?\s*(real|re)|z')", hl) or "real" in hl:
                        real_col = j
                    elif re.search(r"(z['\"]?\s*(imag|im)|z\")|imag", hl) or "imag" in hl or "-z" in hl:
                        imag_col = j
                data_start = True
                continue
            if data_start or (i > 20 and not data_start):
                break

        for line in lines:
            line = line.strip()
            if not line or line.startswith(("#", "!", "?", "Freq")):
                continue
            parts = re.split(r"[\t,;\s]+", line)
            try:
                if len(parts) >= max(freq_col, real_col, imag_col) + 1:
                    f = float(parts[freq_col])
                    zr = float(parts[real_col])
                    zi = float(parts[imag_col])
                    freqs.append(f)
                    reals.append(zr)
                    imags.append(-abs(zi) if zi < 0 else -abs(zi))
            except (ValueError, IndexError):
                continue

        return np.array(freqs), np.array(reals), np.array(imags)

    @classmethod
    def _parse_gamry(cls, content: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        lines = content.splitlines()
        freqs, reals, imags = [], [], []
        in_data = False
        freq_col, real_col, imag_col = 0, 1, 2

        for i, line in enumerate(lines):
            stripped = line.strip()
            if "ZCURVE" in stripped or "ACIMP" in stripped:
                in_data = True
                continue
            if in_data and re.search(r"(freq|frequency)", stripped, re.IGNORECASE):
                headers = re.split(r"[\t,]+", stripped)
                for j, h in enumerate(headers):
                    hl = h.lower()
                    if "freq" in hl:
                        freq_col = j
                    elif "zreal" in hl or "zre" in hl:
                        real_col = j
                    elif "zimag" in hl or "zim" in hl:
                        imag_col = j
                continue
            if in_data and stripped and not stripped.startswith(("#", "PTS")):
                parts = re.split(r"[\t,]+", stripped)
                try:
                    if len(parts) >= 3:
                        f = float(parts[freq_col])
                        zr = float(parts[real_col])
                        zi = float(parts[imag_col])
                        freqs.append(f)
                        reals.append(zr)
                        imags.append(zi if zi < 0 else -abs(zi))
                except (ValueError, IndexError):
                    continue

        return np.array(freqs), np.array(reals), np.array(imags)

    @classmethod
    def _parse_solartron(cls, content: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        lines = content.splitlines()
        freqs, reals, imags = [], [], []
        freq_col, real_col, imag_col = 0, 1, 2

        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped or stripped.startswith(("!", "END", "freq")):
                if "freq" in stripped.lower() and i < 30:
                    headers = re.split(r"[\t,;\s]+", stripped)
                    for j, h in enumerate(headers):
                        hl = h.lower()
                        if "freq" in hl:
                            freq_col = j
                        elif "z'" in hl or "real" in hl or "re" == hl:
                            real_col = j
                        elif 'z"' in hl or "imag" in hl or "im" == hl:
                            imag_col = j
                continue
            parts = re.split(r"[\t,;\s]+", stripped)
            try:
                if len(parts) >= 3:
                    f = float(parts[freq_col])
                    zr = float(parts[real_col])
                    zi = float(parts[imag_col])
                    if f > 0:
                        freqs.append(f)
                        reals.append(zr)
                        imags.append(zi if zi < 0 else -abs(zi))
            except (ValueError, IndexError):
                continue

        return np.array(freqs), np.array(reals), np.array(imags)

    @classmethod
    def _parse_chi(cls, content: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        lines = content.splitlines()
        freqs, reals, imags = [], [], []
        data_started = False
        freq_col, real_col, imag_col = 0, 1, 2

        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped or stripped.startswith(("#", "//", ";")):
                continue
            if "Frequency/Hz" in stripped or "Freq/Hz" in stripped or "频率" in stripped:
                headers = re.split(r"[\t,;\s]+", stripped)
                for j, h in enumerate(headers):
                    hl = h.lower()
                    if "freq" in hl or "频率" in hl:
                        freq_col = j
                    elif "z'" in hl or "real" in hl or "实部" in hl or "zre" in hl:
                        real_col = j
                    elif 'z"' in hl or "imag" in hl or "虚部" in hl or "zim" in hl:
                        imag_col = j
                data_started = True
                continue
            if not data_started and i > 30:
                data_started = True
            if data_started:
                parts = re.split(r"[\t,;\s]+", stripped)
                try:
                    if len(parts) >= 3:
                        f = float(parts[freq_col])
                        zr = float(parts[real_col])
                        zi = float(parts[imag_col])
                        if f > 0:
                            freqs.append(f)
                            reals.append(zr)
                            imags.append(zi if zi < 0 else -abs(zi))
                except (ValueError, IndexError):
                    continue

        return np.array(freqs), np.array(reals), np.array(imags)

    @classmethod
    def _parse_generic(cls, content: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        lines = content.splitlines()
        freqs, reals, imags = [], [], []

        for line in lines:
            stripped = line.strip()
            if not stripped or stripped[0].isalpha():
                continue
            parts = re.split(r"[\t,;\s]+", stripped)
            try:
                if len(parts) >= 3:
                    nums = [float(p) for p in parts[:3]]
                    f, zr, zi = nums[0], nums[1], nums[2]
                    if f > 0:
                        freqs.append(f)
                        reals.append(zr)
                        imags.append(zi if zi < 0 else -abs(zi))
            except (ValueError, IndexError):
                continue

        return np.array(freqs), np.array(reals), np.array(imags)

    @classmethod
    def _extract_metadata(cls, content: str, data: EISData) -> None:
        for line in content.splitlines()[:50]:
            line_lower = line.lower()
            if "temp" in line_lower or "温度" in line_lower:
                match = re.search(r"(\d+\.?\d*)", line)
                if match:
                    try:
                        data.temperature = float(match.group(1))
                    except ValueError:
                        pass
            if "date" in line_lower or "日期" in line_lower:
                match = re.search(r"(\d{4}[-/]\d{1,2}[-/]\d{1,2})", line)
                if match:
                    data.test_date = match.group(1)
