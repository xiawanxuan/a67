from __future__ import annotations

import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
from PySide6.QtCore import Qt, QSize, QThread, Signal
from PySide6.QtGui import QFont, QIcon
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QListWidget,
    QListWidgetItem, QGroupBox, QFormLayout, QLabel, QLineEdit, QComboBox,
    QPushButton, QDoubleSpinBox, QTableWidget, QTableWidgetItem, QHeaderView,
    QStatusBar, QProgressBar, QFileDialog, QMessageBox, QDialog, QDialogButtonBox,
    QSpinBox, QCheckBox, QTabWidget, QTextEdit, QAbstractItemView, QFrame,
)

from ..parsers.base_parser import EISData, DeviceType
from ..parsers.eis_parser import EISParser
from ..fitting.circuit_template import CircuitTemplateManager, CircuitTemplate
from ..fitting.fitting_engine import FittingEngine, FittingResult
from ..plotting.dual_canvas import DualPlotCanvas
from ..database.db_manager import DatabaseManager, SampleRecord
from ..export.data_exporter import DataExporter
from ..batch.batch_dialog import BatchProcessDialog
from .menu_manager import MenuManager


class _FittingThread(QThread):
    finished_signal = Signal(str, object)
    progress_signal = Signal(int)

    def __init__(self, engine: FittingEngine, template: CircuitTemplate,
                 data: EISData, sample_key: str):
        super().__init__()
        self.engine = engine
        self.template = template
        self.data = data
        self.sample_key = sample_key

    def run(self):
        result = self.engine.fit(self.template, self.data)
        self.finished_signal.emit(self.sample_key, result)


class _SampleItemWidget(QWidget):
    def __init__(self, sample_id: str, device_type: str, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)

        self.lbl_name = QLabel(sample_id)
        font = QFont()
        font.setBold(True)
        self.lbl_name.setFont(font)

        self.lbl_device = QLabel(f"[{device_type}]")
        self.lbl_device.setStyleSheet("color: #666; font-size: 10px;")

        self.lbl_status = QLabel("")
        self.lbl_status.setStyleSheet("font-size: 10px;")

        layout.addWidget(self.lbl_name)
        layout.addWidget(self.lbl_device)
        layout.addStretch()
        layout.addWidget(self.lbl_status)

    def set_status(self, status: str, color: str = "#000"):
        self.lbl_status.setText(status)
        self.lbl_status.setStyleSheet(f"color: {color}; font-size: 10px;")


class _AxisLimitDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("设置坐标范围")
        self.resize(400, 400)
        layout = QVBoxLayout(self)

        tab_widget = QTabWidget()
        layout.addWidget(tab_widget)

        nyq_tab = QWidget()
        nyq_layout = QFormLayout(nyq_tab)
        self.nyq_xmin = QDoubleSpinBox()
        self.nyq_xmin.setRange(-1e9, 1e9)
        self.nyq_xmin.setDecimals(4)
        self.nyq_xmin.setMinimumWidth(120)
        self.nyq_xmax = QDoubleSpinBox()
        self.nyq_xmax.setRange(-1e9, 1e9)
        self.nyq_xmax.setDecimals(4)
        self.nyq_xmax.setMinimumWidth(120)
        self.nyq_ymin = QDoubleSpinBox()
        self.nyq_ymin.setRange(-1e9, 1e9)
        self.nyq_ymin.setDecimals(4)
        self.nyq_ymin.setMinimumWidth(120)
        self.nyq_ymax = QDoubleSpinBox()
        self.nyq_ymax.setRange(-1e9, 1e9)
        self.nyq_ymax.setDecimals(4)
        self.nyq_ymax.setMinimumWidth(120)
        self.chk_nyq_auto = QCheckBox("自动范围")
        self.chk_nyq_auto.setChecked(True)
        nyq_layout.addRow(QLabel("奈奎斯特图"))
        nyq_layout.addRow("Z' 最小 (Ω):", self.nyq_xmin)
        nyq_layout.addRow("Z' 最大 (Ω):", self.nyq_xmax)
        nyq_layout.addRow("-Z\" 最小 (Ω):", self.nyq_ymin)
        nyq_layout.addRow("-Z\" 最大 (Ω):", self.nyq_ymax)
        nyq_layout.addRow(self.chk_nyq_auto)
        tab_widget.addTab(nyq_tab, "奈奎斯特图")

        bode_tab = QWidget()
        bode_layout = QFormLayout(bode_tab)
        self.bode_fmin = QDoubleSpinBox()
        self.bode_fmin.setRange(1e-9, 1e9)
        self.bode_fmin.setDecimals(4)
        self.bode_fmin.setMinimumWidth(120)
        self.bode_fmax = QDoubleSpinBox()
        self.bode_fmax.setRange(1e-9, 1e9)
        self.bode_fmax.setDecimals(4)
        self.bode_fmax.setMinimumWidth(120)
        self.bode_mmin = QDoubleSpinBox()
        self.bode_mmin.setRange(1e-9, 1e9)
        self.bode_mmin.setDecimals(4)
        self.bode_mmin.setMinimumWidth(120)
        self.bode_mmax = QDoubleSpinBox()
        self.bode_mmax.setRange(1e-9, 1e9)
        self.bode_mmax.setDecimals(4)
        self.bode_mmax.setMinimumWidth(120)
        self.bode_pmin = QDoubleSpinBox()
        self.bode_pmin.setRange(-180, 0)
        self.bode_pmin.setMinimumWidth(120)
        self.bode_pmax = QDoubleSpinBox()
        self.bode_pmax.setRange(0, 180)
        self.bode_pmax.setMinimumWidth(120)
        self.chk_bode_auto = QCheckBox("自动范围")
        self.chk_bode_auto.setChecked(True)
        bode_layout.addRow(QLabel("波特图"))
        bode_layout.addRow("频率 最小 (Hz):", self.bode_fmin)
        bode_layout.addRow("频率 最大 (Hz):", self.bode_fmax)
        bode_layout.addRow("|Z| 最小 (Ω):", self.bode_mmin)
        bode_layout.addRow("|Z| 最大 (Ω):", self.bode_mmax)
        bode_layout.addRow("相位 最小 (°):", self.bode_pmin)
        bode_layout.addRow("相位 最大 (°):", self.bode_pmax)
        bode_layout.addRow(self.chk_bode_auto)
        tab_widget.addTab(bode_tab, "波特图")

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_limits(self):
        nyq = None
        if not self.chk_nyq_auto.isChecked():
            nyq = (
                (self.nyq_xmin.value(), self.nyq_xmax.value()),
                (self.nyq_ymin.value(), self.nyq_ymax.value()),
            )
        bode = None
        if not self.chk_bode_auto.isChecked():
            bode = (
                (self.bode_fmin.value(), self.bode_fmax.value()),
                (self.bode_mmin.value(), self.bode_mmax.value()),
                (self.bode_pmin.value(), self.bode_pmax.value()),
            )
        return nyq, bode


class MainWindow(QMainWindow):
    """EIS 阻抗谱分析系统主窗口"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("EIS 阻抗谱分析系统 - 新能源材料实验室")
        self.resize(1400, 900)

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.config_dir = os.path.join(os.path.dirname(base_dir), "configs")
        circuit_cfg = os.path.join(self.config_dir, "circuit_templates.json")
        self.calibration_cfg = os.path.join(self.config_dir, "calibration.json")

        self.parser = EISParser()
        self.template_manager = CircuitTemplateManager(circuit_cfg if os.path.exists(circuit_cfg) else None)
        self.fitting_engine = FittingEngine()
        self.db_manager = DatabaseManager()
        self.exporter = DataExporter()

        self._samples: Dict[str, Tuple[EISData, Optional[FittingResult], Optional[int]]] = {}
        self._current_sample_key: Optional[str] = None
        self._fitting_threads: Dict[str, _FittingThread] = {}

        self._init_ui()
        self._init_menu()
        self._load_startup_data()
        self._update_ui_state()

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(4)

        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        left_panel = self._build_left_panel()
        center_panel = self._build_center_panel()
        right_panel = self._build_right_panel()

        splitter.addWidget(left_panel)
        splitter.addWidget(center_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 4)
        splitter.setStretchFactor(2, 2)
        splitter.setSizes([240, 720, 440])

        self._init_status_bar()

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        grp = QGroupBox("样品列表")
        grp_layout = QVBoxLayout(grp)

        btn_row = QHBoxLayout()
        self.btn_add = QPushButton("导入文件")
        self.btn_add.clicked.connect(self._on_open_file)
        self.btn_batch = QPushButton("批量处理")
        self.btn_batch.clicked.connect(self._on_batch_process)
        self.btn_del = QPushButton("删除")
        self.btn_del.clicked.connect(self._on_delete_sample)
        self.btn_clear = QPushButton("清空")
        self.btn_clear.clicked.connect(self._on_clear_all)
        btn_row.addWidget(self.btn_add)
        btn_row.addWidget(self.btn_batch)
        btn_row.addWidget(self.btn_del)
        btn_row.addWidget(self.btn_clear)
        grp_layout.addLayout(btn_row)

        self.sample_list = QListWidget()
        self.sample_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.sample_list.itemSelectionChanged.connect(self._on_sample_selected)
        grp_layout.addWidget(self.sample_list, stretch=1)

        info_row = QFormLayout()
        self.lbl_count = QLabel("0")
        self.lbl_batch = QLineEdit("")
        self.lbl_batch.setPlaceholderText("批次号（可编辑）")
        info_row.addRow("样品数量:", self.lbl_count)
        info_row.addRow("当前批次:", self.lbl_batch)
        grp_layout.addLayout(info_row)

        layout.addWidget(grp)
        return panel

    def _build_center_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.plot_canvas = DualPlotCanvas()
        layout.addWidget(self.plot_canvas, stretch=1)

        ctrl_row = QHBoxLayout()
        ctrl_row.setContentsMargins(4, 0, 4, 4)
        self.chk_show_fit = QCheckBox("显示拟合曲线")
        self.chk_show_fit.setChecked(True)
        self.chk_show_fit.stateChanged.connect(self._on_toggle_show_fit)
        self.btn_autoscale = QPushButton("自动缩放")
        self.btn_autoscale.clicked.connect(self._on_auto_scale)
        self.btn_axis_limits = QPushButton("设置坐标范围")
        self.btn_axis_limits.clicked.connect(self._on_set_axis_limits)
        self.btn_save_fig = QPushButton("保存图像")
        self.btn_save_fig.clicked.connect(self._on_save_figure)
        ctrl_row.addWidget(self.chk_show_fit)
        ctrl_row.addStretch()
        ctrl_row.addWidget(self.btn_autoscale)
        ctrl_row.addWidget(self.btn_axis_limits)
        ctrl_row.addWidget(self.btn_save_fig)
        layout.addLayout(ctrl_row)

        return panel

    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        info_box = QGroupBox("样品信息")
        info_layout = QFormLayout(info_box)
        self.edt_sample_id = QLineEdit("")
        self.edt_sample_id.setReadOnly(False)
        self.edt_device = QLineEdit("")
        self.edt_device.setReadOnly(True)
        self.edt_file = QLineEdit("")
        self.edt_file.setReadOnly(True)
        self.edt_temp = QDoubleSpinBox()
        self.edt_temp.setRange(-50, 300)
        self.edt_temp.setSuffix(" °C")
        self.edt_temp.setSpecialValueText("未设置")
        self.edt_temp.setValue(-50)
        self.edt_date = QLineEdit("")
        self.edt_points = QLabel("0")
        info_layout.addRow("样品ID:", self.edt_sample_id)
        info_layout.addRow("设备类型:", self.edt_device)
        info_layout.addRow("数据点数:", self.edt_points)
        info_layout.addRow("测试温度:", self.edt_temp)
        info_layout.addRow("测试日期:", self.edt_date)
        info_layout.addRow("文件路径:", self.edt_file)
        layout.addWidget(info_box)

        fit_box = QGroupBox("拟合设置")
        fit_layout = QFormLayout(fit_box)
        self.cmb_circuit = QComboBox()
        self._populate_circuit_combo()
        self.cmb_circuit.currentIndexChanged.connect(self._on_circuit_changed)

        self.lbl_circuit_desc = QLabel("")
        self.lbl_circuit_desc.setWordWrap(True)
        self.lbl_circuit_desc.setStyleSheet("color: #555; font-size: 11px;")

        self.tbl_init_params = QTableWidget()
        self.tbl_init_params.setColumnCount(3)
        self.tbl_init_params.setHorizontalHeaderLabels(["参数", "初始值", "范围"])
        self.tbl_init_params.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl_init_params.verticalHeader().setVisible(False)
        self.tbl_init_params.setMaximumHeight(160)

        self.cmb_method = QComboBox()
        self.cmb_method.addItems(["LM-TRF (推荐)", "差分进化"])

        self.btn_fit = QPushButton("开始拟合")
        self.btn_fit.clicked.connect(self._on_fit_current)
        self.btn_fit_all = QPushButton("拟合全部")
        self.btn_fit_all.clicked.connect(self._on_fit_all)

        fit_layout.addRow("等效电路:", self.cmb_circuit)
        fit_layout.addRow(self.lbl_circuit_desc)
        fit_layout.addRow("初始参数:", self.tbl_init_params)
        fit_layout.addRow("拟合方法:", self.cmb_method)
        fit_row = QHBoxLayout()
        fit_row.addWidget(self.btn_fit)
        fit_row.addWidget(self.btn_fit_all)
        fit_layout.addRow(fit_row)
        layout.addWidget(fit_box)

        result_box = QGroupBox("拟合结果")
        result_layout = QVBoxLayout(result_box)
        self.lbl_fit_status = QLabel("未拟合")
        self.lbl_fit_status.setStyleSheet("font-weight: bold; color: #888;")
        self.lbl_chi_sq = QLabel("χ² = -")
        self.tbl_result = QTableWidget()
        self.tbl_result.setColumnCount(2)
        self.tbl_result.setHorizontalHeaderLabels(["参数", "值"])
        self.tbl_result.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl_result.verticalHeader().setVisible(False)

        result_layout.addWidget(self.lbl_fit_status)
        result_layout.addWidget(self.lbl_chi_sq)
        result_layout.addWidget(self.tbl_result, stretch=1)

        db_row = QHBoxLayout()
        self.btn_save_db = QPushButton("保存到数据库")
        self.btn_save_db.clicked.connect(self._on_save_to_db)
        self.btn_load_db = QPushButton("从数据库加载")
        self.btn_load_db.clicked.connect(self._on_load_from_db)
        db_row.addWidget(self.btn_save_db)
        db_row.addWidget(self.btn_load_db)
        result_layout.addLayout(db_row)

        export_row = QHBoxLayout()
        self.btn_export_csv = QPushButton("导出CSV")
        self.btn_export_csv.clicked.connect(self._on_export_single_csv)
        self.btn_export_params = QPushButton("导出参数汇总")
        self.btn_export_params.clicked.connect(self._on_export_params)
        export_row.addWidget(self.btn_export_csv)
        export_row.addWidget(self.btn_export_params)
        result_layout.addLayout(export_row)

        layout.addWidget(result_box, stretch=1)
        return panel

    def _init_menu(self):
        actions = {
            "open": self._on_open_file,
            "open_batch": self._on_open_batch,
            "batch_process": self._on_batch_process,
            "export_single_csv": self._on_export_single_csv,
            "export_batch_csv": self._on_export_batch_csv,
            "export_params": self._on_export_params,
            "export_excel": self._on_export_excel,
            "save_figure": self._on_save_figure,
            "exit": self.close,
            "delete_sample": self._on_delete_sample,
            "clear_all": self._on_clear_all,
            "auto_scale": self._on_auto_scale,
            "fit_current": self._on_fit_current,
            "fit_all": self._on_fit_all,
            "toggle_show_fit": lambda: self.chk_show_fit.toggle(),
            "circuit_template": self._on_manage_templates,
            "set_axis_limits": self._on_set_axis_limits,
            "save_to_db": self._on_save_to_db,
            "load_from_db": self._on_load_from_db,
            "manage_batches": self._on_manage_batches,
            "about": self._on_about,
        }
        self.menu_mgr = MenuManager(self.menuBar())
        self.menu_mgr.create_file_menu(actions)
        self.menu_mgr.create_edit_menu(actions)
        self.menu_mgr.create_fitting_menu(actions)
        self.menu_mgr.create_view_menu(actions)
        self.menu_mgr.create_database_menu(actions)
        self.menu_mgr.create_help_menu(actions)

    def _init_status_bar(self):
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)
        self.lbl_status_msg = QLabel("就绪")
        self.status_bar.addWidget(self.lbl_status_msg, stretch=1)

    def _populate_circuit_combo(self):
        self.cmb_circuit.clear()
        for tpl in self.template_manager.list_templates():
            self.cmb_circuit.addItem(f"{tpl.name} ({tpl.key})", tpl.key)
        if self.template_manager.default_key:
            idx = self.cmb_circuit.findData(self.template_manager.default_key)
            if idx >= 0:
                self.cmb_circuit.setCurrentIndex(idx)
        self._refresh_init_params_table()

    def _refresh_init_params_table(self):
        key = self.cmb_circuit.currentData()
        template = self.template_manager.get_template(key) if key else None
        if not template:
            self.tbl_init_params.setRowCount(0)
            self.lbl_circuit_desc.setText("")
            return
        self.lbl_circuit_desc.setText(f"{template.description}\n表达式: {template.circuit_expr}")
        self.tbl_init_params.setRowCount(template.param_count)
        for i, (name, guess, lb, ub) in enumerate(zip(
                template.parameters, template.initial_guess,
                template.lower_bounds, template.upper_bounds)):
            self.tbl_init_params.setItem(i, 0, QTableWidgetItem(name))
            spin = QDoubleSpinBox()
            spin.setDecimals(8)
            spin.setRange(max(-1e12, lb), min(1e12, ub))
            spin.setValue(guess)
            spin.setSingleStep(max(abs(guess) * 0.1, 1e-12))
            self.tbl_init_params.setCellWidget(i, 1, spin)
            self.tbl_init_params.setItem(i, 2, QTableWidgetItem(f"[{lb:.2e}, {ub:.2e}]"))

    def _on_circuit_changed(self, idx: int):
        self._refresh_init_params_table()

    def _set_status(self, msg: str):
        self.lbl_status_msg.setText(msg)

    def _load_startup_data(self):
        self._set_status("正在加载本地数据库...")
        try:
            loaded = self.db_manager.load_all_on_startup()
            count = 0
            for db_id, (data, fit_result) in loaded.items():
                key = f"db_{db_id}_{data.sample_id}"
                self._samples[key] = (data, fit_result, db_id)
                self._add_sample_to_list(key, data, fit_result)
                if data.is_valid:
                    self.plot_canvas.add_sample(data, fit_result)
                    count += 1
            if count > 0:
                self._set_status(f"已从数据库加载 {count} 个样品")
            else:
                self._set_status("就绪")
            self.lbl_count.setText(str(len(self._samples)))
        except Exception as e:
            self._set_status(f"加载数据库失败: {e}")

    def _add_sample_to_list(self, key: str, data: EISData,
                            fit_result: Optional[FittingResult] = None):
        item = QListWidgetItem()
        item.setData(Qt.UserRole, key)
        widget = _SampleItemWidget(data.sample_id, data.device_type.value)
        if fit_result and fit_result.success:
            widget.set_status("✓ 已拟合", "#2ca02c")
        item.setSizeHint(widget.sizeHint())
        self.sample_list.addItem(item)
        self.sample_list.setItemWidget(item, widget)

    def _update_sample_list_status(self, key: str, status: str, color: str):
        for i in range(self.sample_list.count()):
            item = self.sample_list.item(i)
            if item.data(Qt.UserRole) == key:
                widget = self.sample_list.itemWidget(item)
                if isinstance(widget, _SampleItemWidget):
                    widget.set_status(status, color)
                break

    def _on_open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择 EIS 数据文件", "",
            "所有支持的文件 (*.txt *.csv *.dta *.nox *.z *.zsimpwin);;文本文件 (*.txt *.csv);;所有文件 (*.*)"
        )
        if path:
            self._import_file(path)

    def _on_open_batch(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "选择多个 EIS 数据文件", "",
            "所有支持的文件 (*.txt *.csv *.dta *.nox *.z *.zsimpwin);;文本文件 (*.txt *.csv);;所有文件 (*.*)"
        )
        if paths:
            for p in paths:
                self._import_file(p)

    def _on_batch_process(self):
        dialog = BatchProcessDialog(self.template_manager, self.db_manager, self)
        if dialog.exec() == QDialog.Accepted:
            samples = dialog.get_importable_samples()
            if samples:
                count = 0
                for data, fit_result in samples:
                    key = f"batch_{len(self._samples)}_{data.sample_id}"
                    while key in self._samples:
                        key = key + "_"
                    db_id = None
                    if self.db_manager:
                        existing = self.db_manager.get_samples_by_batch(data.batch_id)
                        for rec in existing:
                            if rec.sample_id == data.sample_id:
                                db_id = rec.id
                                break
                    self._samples[key] = (data, fit_result, db_id)
                    self._add_sample_to_list(key, data, fit_result)
                    if data.is_valid:
                        self.plot_canvas.add_sample(data, fit_result)
                        count += 1
                self.lbl_count.setText(str(len(self._samples)))
                self._set_status(f"已从批量处理导入 {count} 个样品")
                self._update_ui_state()

    def _import_file(self, file_path: str):
        batch_id = self.lbl_batch.text().strip()
        try:
            data = self.parser.parse_file(file_path, batch_id=batch_id)
            if not data.is_valid:
                QMessageBox.warning(self, "导入失败", f"文件 {os.path.basename(file_path)} 未解析到有效数据")
                return

            key = f"file_{len(self._samples)}_{data.sample_id}"
            while key in self._samples:
                key = key + "_"
            self._samples[key] = (data, None, None)
            self._add_sample_to_list(key, data)

            plot_key = self.plot_canvas.add_sample(data)
            self.lbl_count.setText(str(len(self._samples)))
            self._set_status(f"已导入: {os.path.basename(file_path)} ({len(data.frequencies)} 点)")
            self._update_ui_state()
        except Exception as e:
            QMessageBox.critical(self, "导入错误", f"导入文件失败:\n{str(e)}")

    def _on_sample_selected(self):
        items = self.sample_list.selectedItems()
        if not items:
            self._current_sample_key = None
            self._clear_sample_info()
            return
        item = items[0]
        key = item.data(Qt.UserRole)
        self._current_sample_key = key
        if key in self._samples:
            data, fit_result, db_id = self._samples[key]
            self._show_sample_info(data, fit_result)

    def _clear_sample_info(self):
        self.edt_sample_id.setText("")
        self.edt_device.setText("")
        self.edt_file.setText("")
        self.edt_temp.setValue(-50)
        self.edt_date.setText("")
        self.edt_points.setText("0")
        self.lbl_fit_status.setText("未拟合")
        self.lbl_fit_status.setStyleSheet("font-weight: bold; color: #888;")
        self.lbl_chi_sq.setText("χ² = -")
        self.tbl_result.setRowCount(0)

    def _show_sample_info(self, data: EISData, fit_result: Optional[FittingResult]):
        self.edt_sample_id.setText(data.sample_id)
        self.edt_device.setText(data.device_type.value)
        self.edt_file.setText(data.file_path)
        if data.temperature is not None:
            self.edt_temp.setValue(data.temperature)
        else:
            self.edt_temp.setValue(-50)
        self.edt_date.setText(data.test_date or "")
        self.edt_points.setText(str(len(data.frequencies)))

        if fit_result:
            if fit_result.success:
                self.lbl_fit_status.setText("✓ 拟合成功")
                self.lbl_fit_status.setStyleSheet("font-weight: bold; color: #2ca02c;")
                self.lbl_chi_sq.setText(f"χ² = {fit_result.chi_squared:.4e}")
                self.tbl_result.setRowCount(len(fit_result.parameters))
                for i, (name, value) in enumerate(fit_result.parameters.items()):
                    self.tbl_result.setItem(i, 0, QTableWidgetItem(name))
                    formatted = self.fitting_engine.format_param_value(name, value)
                    self.tbl_result.setItem(i, 1, QTableWidgetItem(f"{formatted}  ({value:.6e})"))
            else:
                self.lbl_fit_status.setText(f"✗ 拟合失败: {fit_result.message}")
                self.lbl_fit_status.setStyleSheet("font-weight: bold; color: #d62728;")
                self.lbl_chi_sq.setText("χ² = -")
                self.tbl_result.setRowCount(0)
        else:
            self.lbl_fit_status.setText("未拟合")
            self.lbl_fit_status.setStyleSheet("font-weight: bold; color: #888;")
            self.lbl_chi_sq.setText("χ² = -")
            self.tbl_result.setRowCount(0)

    def _on_delete_sample(self):
        items = self.sample_list.selectedItems()
        if not items:
            return
        for item in items:
            key = item.data(Qt.UserRole)
            if key in self._samples:
                _, _, db_id = self._samples[key]
                if db_id is not None:
                    self.db_manager.delete_sample(db_id)
                plot_ids = self.plot_canvas.get_sample_ids()
                if key in plot_ids:
                    self.plot_canvas.remove_sample(key)
                del self._samples[key]
                self.sample_list.takeItem(self.sample_list.row(item))
        self.lbl_count.setText(str(len(self._samples)))
        self._current_sample_key = None
        self._clear_sample_info()
        self._set_status("已删除选中样品")
        self._update_ui_state()

    def _on_clear_all(self):
        if not self._samples:
            return
        reply = QMessageBox.question(self, "确认", "确定要清空所有样品吗？",
                                     QMessageBox.Yes | QMessageBox.No)
        if reply != QMessageBox.Yes:
            return
        for key in list(self._samples.keys()):
            _, _, db_id = self._samples[key]
            if db_id is not None:
                self.db_manager.delete_sample(db_id)
        self._samples.clear()
        self.sample_list.clear()
        self.plot_canvas.clear_samples()
        self.lbl_count.setText("0")
        self._current_sample_key = None
        self._clear_sample_info()
        self._set_status("已清空所有样品")
        self._update_ui_state()

    def _get_current_template(self) -> Optional[CircuitTemplate]:
        key = self.cmb_circuit.currentData()
        return self.template_manager.get_template(key) if key else None

    def _get_init_params_from_table(self, template: CircuitTemplate) -> List[float]:
        params = []
        for i in range(template.param_count):
            widget = self.tbl_init_params.cellWidget(i, 1)
            if isinstance(widget, QDoubleSpinBox):
                params.append(widget.value())
            else:
                params.append(template.initial_guess[i])
        return params

    def _on_fit_current(self):
        if not self._current_sample_key or self._current_sample_key not in self._samples:
            QMessageBox.information(self, "提示", "请先选择一个样品")
            return
        template = self._get_current_template()
        if not template:
            QMessageBox.warning(self, "提示", "请选择等效电路模型")
            return
        data, _, _ = self._samples[self._current_sample_key]
        self._run_fitting(self._current_sample_key, template, data)

    def _on_fit_all(self):
        if not self._samples:
            return
        template = self._get_current_template()
        if not template:
            QMessageBox.warning(self, "提示", "请选择等效电路模型")
            return
        for key, (data, _, _) in list(self._samples.items()):
            self._run_fitting(key, template, data)

    def _run_fitting(self, key: str, template: CircuitTemplate, data: EISData):
        if not data.is_valid:
            return
        self._update_sample_list_status(key, "拟合中...", "#ff7f0e")
        self.btn_fit.setEnabled(False)
        self.btn_fit_all.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self._set_status(f"正在拟合: {data.sample_id}")

        init_params = self._get_init_params_from_table(template)
        thread = _FittingThread(self.fitting_engine, template, data, key)
        thread.finished_signal.connect(self._on_fitting_finished)
        self._fitting_threads[key] = thread
        thread.start()

    def _on_fitting_finished(self, key: str, result: FittingResult):
        if key not in self._samples:
            return
        data, _, db_id = self._samples[key]
        self._samples[key] = (data, result, db_id)

        if result.success:
            self._update_sample_list_status(key, "✓ 已拟合", "#2ca02c")
            self.plot_canvas.update_fitting(key, result)
            self._set_status(f"拟合成功: {data.sample_id}, χ² = {result.chi_squared:.4e}")
        else:
            self._update_sample_list_status(key, "✗ 失败", "#d62728")
            self._set_status(f"拟合失败: {data.sample_id} - {result.message}")

        if self._current_sample_key == key:
            self._show_sample_info(data, result)

        if key in self._fitting_threads:
            del self._fitting_threads[key]
        if not self._fitting_threads:
            self.progress_bar.setVisible(False)
            self.btn_fit.setEnabled(True)
            self.btn_fit_all.setEnabled(True)

    def _on_toggle_show_fit(self, state: int):
        self.plot_canvas.set_show_fit(state == Qt.Checked)

    def _on_auto_scale(self):
        self.plot_canvas.auto_scale()

    def _on_set_axis_limits(self):
        dlg = _AxisLimitDialog(self)
        if dlg.exec() == QDialog.Accepted:
            nyq, bode = dlg.get_limits()
            nyq_x, nyq_y = (None, None)
            bode_x, bode_my, bode_py = (None, None, None)
            if nyq:
                nyq_x, nyq_y = nyq
            if bode:
                bode_x, bode_my, bode_py = bode
            self.plot_canvas.set_axis_limits(nyq_x, nyq_y, bode_x, bode_my, bode_py)
            if not nyq and not bode:
                self.plot_canvas.auto_scale()

    def _on_save_figure(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "保存图像", "eis_plot.png",
            "PNG 图片 (*.png);;JPEG 图片 (*.jpg);;SVG 矢量图 (*.svg);;PDF (*.pdf)"
        )
        if path:
            if self.exporter.export_figure(self.plot_canvas, path):
                self._set_status(f"图像已保存: {path}")
            else:
                QMessageBox.warning(self, "保存失败", "保存图像时发生错误")

    def _get_samples_for_export(self) -> List[Tuple[EISData, Optional[FittingResult]]]:
        items = self.sample_list.selectedItems()
        if items:
            result = []
            for item in items:
                key = item.data(Qt.UserRole)
                if key in self._samples:
                    data, fit, _ = self._samples[key]
                    result.append((data, fit))
            return result
        return [(d, f) for d, f, _ in self._samples.values()]

    def _on_export_single_csv(self):
        if not self._current_sample_key or self._current_sample_key not in self._samples:
            QMessageBox.information(self, "提示", "请先选择一个样品")
            return
        data, fit, _ = self._samples[self._current_sample_key]
        path, _ = QFileDialog.getSaveFileName(
            self, "导出 CSV", f"{data.sample_id}.csv", "CSV 文件 (*.csv)"
        )
        if path:
            if self.exporter.export_single_csv(path, data, fit):
                self._set_status(f"已导出: {path}")
            else:
                QMessageBox.warning(self, "导出失败", "导出 CSV 时发生错误")

    def _on_export_batch_csv(self):
        samples = self._get_samples_for_export()
        if not samples:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "批量导出 CSV", "eis_batch_data.csv", "CSV 文件 (*.csv)"
        )
        if path:
            if self.exporter.export_batch_csv(path, samples):
                self._set_status(f"已批量导出: {path}")
            else:
                QMessageBox.warning(self, "导出失败", "批量导出 CSV 时发生错误")

    def _on_export_params(self):
        samples = self._get_samples_for_export()
        if not samples:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "导出拟合参数", "fitting_params.csv", "CSV 文件 (*.csv)"
        )
        if path:
            if self.exporter.export_fitting_params_csv(path, samples):
                self._set_status(f"已导出参数: {path}")
            else:
                QMessageBox.warning(self, "导出失败", "导出拟合参数时发生错误")

    def _on_export_excel(self):
        samples = self._get_samples_for_export()
        if not samples:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "导出 Excel 报告", "eis_report.xlsx", "Excel 文件 (*.xlsx)"
        )
        if path:
            if self.exporter.export_excel(path, samples):
                self._set_status(f"已导出 Excel: {path}")
            else:
                QMessageBox.warning(self, "导出失败", "导出 Excel 时发生错误，请确保已安装 openpyxl")

    def _on_save_to_db(self):
        if not self._samples:
            return
        count = 0
        for key, (data, fit, db_id) in list(self._samples.items()):
            if db_id is None:
                new_id = self.db_manager.add_sample(data)
                self._samples[key] = (data, fit, new_id)
                if fit:
                    self.db_manager.add_fitting_result(new_id, fit)
                count += 1
            else:
                self.db_manager.update_sample(db_id, data)
                if fit:
                    self.db_manager.add_fitting_result(db_id, fit)
                count += 1
        self._set_status(f"已保存 {count} 个样品到数据库")

    def _on_load_from_db(self):
        loaded = self.db_manager.load_all_on_startup()
        count = 0
        for db_id, (data, fit_result) in loaded.items():
            key = f"db_{db_id}_{data.sample_id}"
            if key not in self._samples:
                self._samples[key] = (data, fit_result, db_id)
                self._add_sample_to_list(key, data, fit_result)
                if data.is_valid:
                    self.plot_canvas.add_sample(data, fit_result)
                    count += 1
        self.lbl_count.setText(str(len(self._samples)))
        self._set_status(f"已从数据库加载 {count} 个样品")
        self._update_ui_state()

    def _on_manage_templates(self):
        QMessageBox.information(self, "等效电路模板",
                                "等效电路模板管理功能\n可在 configs/circuit_templates.json 中编辑模板")

    def _on_manage_batches(self):
        batches = self.db_manager.list_batches()
        msg = "已有的批次:\n\n" + "\n".join(batches) if batches else "暂无批次数据"
        QMessageBox.information(self, "批次管理", msg)

    def _on_about(self):
        QMessageBox.about(self, "关于",
                          "EIS 阻抗谱分析系统 v1.0\n\n"
                          "新能源材料实验室\n"
                          "功能：EIS 数据解析、等效电路拟合、奈奎斯特/波特图绘制")

    def _update_ui_state(self):
        has_samples = bool(self._samples)
        has_selected = self._current_sample_key is not None
        self.menu_mgr.set_action_enabled("delete_sample", has_selected)
        self.menu_mgr.set_action_enabled("clear_all", has_samples)
        self.menu_mgr.set_action_enabled("auto_scale", has_samples)
        self.menu_mgr.set_action_enabled("fit_current", has_selected)
        self.menu_mgr.set_action_enabled("fit_all", has_samples)
        self.menu_mgr.set_action_enabled("save_figure", has_samples)
        self.menu_mgr.set_action_enabled("export_single_csv", has_selected)
        self.menu_mgr.set_action_enabled("export_batch_csv", has_samples)
        self.menu_mgr.set_action_enabled("export_params", has_samples)
        self.menu_mgr.set_action_enabled("export_excel", has_samples)
        self.menu_mgr.set_action_enabled("save_to_db", has_samples)

    def closeEvent(self, event):
        self.db_manager.close()
        super().closeEvent(event)
