from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

from ..parsers.base_parser import EISData, DeviceType
from ..fitting.fitting_engine import FittingResult


@dataclass
class SampleRecord:
    id: Optional[int]
    sample_id: str
    batch_id: str
    device_type: str
    file_path: str
    temperature: Optional[float]
    test_date: Optional[str]
    notes: str
    frequencies_json: str
    z_real_json: str
    z_imag_json: str
    created_at: str = ""

    def to_eis_data(self) -> EISData:
        return EISData(
            frequencies=np.array(json.loads(self.frequencies_json)),
            z_real=np.array(json.loads(self.z_real_json)),
            z_imag=np.array(json.loads(self.z_imag_json)),
            device_type=DeviceType(self.device_type) if self.device_type else DeviceType.Unknown,
            sample_id=self.sample_id,
            batch_id=self.batch_id,
            file_path=self.file_path,
            temperature=self.temperature,
            test_date=self.test_date,
            notes=self.notes,
        )

    @classmethod
    def from_eis_data(cls, data: EISData, record_id: Optional[int] = None) -> "SampleRecord":
        return cls(
            id=record_id,
            sample_id=data.sample_id,
            batch_id=data.batch_id,
            device_type=data.device_type.value,
            file_path=data.file_path,
            temperature=data.temperature,
            test_date=data.test_date,
            notes=data.notes,
            frequencies_json=json.dumps(data.frequencies.tolist()),
            z_real_json=json.dumps(data.z_real.tolist()),
            z_imag_json=json.dumps(data.z_imag.tolist()),
        )


@dataclass
class FittingRecord:
    id: Optional[int]
    sample_record_id: int
    template_key: str
    success: bool
    parameters_json: str
    chi_squared: float
    message: str
    created_at: str = ""

    def to_fitting_result(self) -> FittingResult:
        return FittingResult(
            success=bool(self.success),
            template_key=self.template_key,
            parameters=json.loads(self.parameters_json) if self.parameters_json else {},
            chi_squared=self.chi_squared,
            message=self.message,
        )

    @classmethod
    def from_fitting_result(cls, result: FittingResult, sample_record_id: int,
                            record_id: Optional[int] = None) -> "FittingRecord":
        return cls(
            id=record_id,
            sample_record_id=sample_record_id,
            template_key=result.template_key,
            success=result.success,
            parameters_json=json.dumps(result.parameters, ensure_ascii=False),
            chi_squared=result.chi_squared,
            message=result.message,
        )


class DatabaseManager:
    """本地 SQLite 数据库管理器，用于持久化电池样品和拟合参数"""

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            db_path = os.path.join(base_dir, "eis_data.db")
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._initialize()

    def _connect(self):
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    def _initialize(self):
        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sample_id TEXT NOT NULL,
                batch_id TEXT DEFAULT '',
                device_type TEXT DEFAULT 'Unknown',
                file_path TEXT DEFAULT '',
                temperature REAL,
                test_date TEXT,
                notes TEXT DEFAULT '',
                frequencies_json TEXT NOT NULL,
                z_real_json TEXT NOT NULL,
                z_imag_json TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS fitting_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sample_record_id INTEGER NOT NULL,
                template_key TEXT NOT NULL,
                success INTEGER DEFAULT 0,
                parameters_json TEXT DEFAULT '{}',
                chi_squared REAL DEFAULT 0.0,
                message TEXT DEFAULT '',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (sample_record_id) REFERENCES samples(id) ON DELETE CASCADE
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_samples_batch ON samples(batch_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_fitting_sample ON fitting_results(sample_record_id)")
        conn.commit()

    def add_sample(self, data: EISData) -> int:
        record = SampleRecord.from_eis_data(data)
        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO samples (sample_id, batch_id, device_type, file_path, temperature,
                                 test_date, notes, frequencies_json, z_real_json, z_imag_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record.sample_id, record.batch_id, record.device_type, record.file_path,
            record.temperature, record.test_date, record.notes,
            record.frequencies_json, record.z_real_json, record.z_imag_json,
        ))
        conn.commit()
        return cursor.lastrowid

    def update_sample(self, sample_id: int, data: EISData):
        record = SampleRecord.from_eis_data(data, sample_id)
        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE samples SET sample_id=?, batch_id=?, device_type=?, file_path=?,
                               temperature=?, test_date=?, notes=?,
                               frequencies_json=?, z_real_json=?, z_imag_json=?
            WHERE id=?
        """, (
            record.sample_id, record.batch_id, record.device_type, record.file_path,
            record.temperature, record.test_date, record.notes,
            record.frequencies_json, record.z_real_json, record.z_imag_json,
            sample_id,
        ))
        conn.commit()

    def delete_sample(self, sample_id: int):
        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM fitting_results WHERE sample_record_id=?", (sample_id,))
        cursor.execute("DELETE FROM samples WHERE id=?", (sample_id,))
        conn.commit()

    def get_sample(self, sample_id: int) -> Optional[SampleRecord]:
        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM samples WHERE id=?", (sample_id,))
        row = cursor.fetchone()
        if row:
            return SampleRecord(**dict(row))
        return None

    def get_all_samples(self) -> List[SampleRecord]:
        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM samples ORDER BY created_at DESC")
        rows = cursor.fetchall()
        return [SampleRecord(**dict(r)) for r in rows]

    def get_samples_by_batch(self, batch_id: str) -> List[SampleRecord]:
        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM samples WHERE batch_id=? ORDER BY created_at DESC", (batch_id,))
        rows = cursor.fetchall()
        return [SampleRecord(**dict(r)) for r in rows]

    def list_batches(self) -> List[str]:
        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT batch_id FROM samples WHERE batch_id != '' ORDER BY batch_id")
        rows = cursor.fetchall()
        return [r["batch_id"] for r in rows]

    def add_fitting_result(self, sample_record_id: int, result: FittingResult) -> int:
        record = FittingRecord.from_fitting_result(result, sample_record_id)
        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO fitting_results (sample_record_id, template_key, success,
                                         parameters_json, chi_squared, message)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            record.sample_record_id, record.template_key, int(record.success),
            record.parameters_json, record.chi_squared, record.message,
        ))
        conn.commit()
        return cursor.lastrowid

    def get_fitting_results(self, sample_record_id: int) -> List[FittingRecord]:
        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM fitting_results
            WHERE sample_record_id=?
            ORDER BY created_at DESC
        """, (sample_record_id,))
        rows = cursor.fetchall()
        return [FittingRecord(**dict(r)) for r in rows]

    def get_latest_fitting(self, sample_record_id: int) -> Optional[FittingRecord]:
        results = self.get_fitting_results(sample_record_id)
        return results[0] if results else None

    def delete_fitting_result(self, fitting_id: int):
        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM fitting_results WHERE id=?", (fitting_id,))
        conn.commit()

    def load_all_on_startup(self) -> Dict[int, tuple]:
        """启动时加载所有样品数据和最新拟合结果"""
        samples = self.get_all_samples()
        result: Dict[int, tuple] = {}
        for sample in samples:
            if sample.id is not None:
                latest_fit = self.get_latest_fitting(sample.id)
                fit_result = latest_fit.to_fitting_result() if latest_fit else None
                result[sample.id] = (sample.to_eis_data(), fit_result)
        return result
