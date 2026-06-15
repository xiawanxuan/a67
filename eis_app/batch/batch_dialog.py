from __future__ import annotations

import os
from typing import Optional

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, QLineEdit,
    QComboBox, QCheckBox, QPushButton, QProgressBar, QTextEdit, QFileDialog,
    QSpinBox, QGroupBox, QDialogButtonBox, QListWidget, QListWidgetItem,
    QMessageBox,
)

from ..fitting.circuit_template import CircuitTemplateManager
from ..database.db_manager import DatabaseManager
from .batch_processor import BatchProcessor, BatchProcessingOptions, BatchResult


class _BatchWorkerThread(QThread):
    progress_signal = Signal(int, int, str)
    finished_signal = Signal(object)

    def __init__(self, processor: BatchProcessor, directory: str,
                 options: BatchProcessingOptions):
        super().__init__()
        self.processor = processor
        self.directory = directory
        self.options = options

    def run(self):
        self.processor.set_progress_callback(self._on_progress)
        result = self.processor.process_directory(self.directory, self.options)
        self.finished_signal.emit(result)

    def _on_progress(self, current: int, total: int, message: str):
        self.progress_signal.emit(current, total, message)


class BatchProcessDialog(QDialog):
    """文件夹批量处理对话框"""

    def __init__(self, template_manager: CircuitTemplateManager,
                 db_manager: Optional[DatabaseManager] = None,
                 parent=None):
        super().__init__(parent)
        self.setWindowTitle("批量处理 EIS 阻抗谱数据")
        self.resize(720, 640)

        self.template_manager = template_manager
        self.db_manager = db_manager
        self.processor = BatchProcessor(template_manager, db_manager)
        self.worker_thread: Optional[_BatchWorkerThread] = None
        self.batch_result: Optional[BatchResult] = None

        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        dir_group = QGroupBox("目录设置")
        dir_layout = QHBoxLayout(dir_group)
        self.edt_dir = QLineEdit()
        self.edt_dir.setPlaceholderText("选择包含 EIS 数据文件的文件夹...")
        self.btn_browse = QPushButton("浏览...")
        self.btn_browse.clicked.connect(self._on_browse)
        dir_layout.addWidget(self.edt_dir, stretch=1)
        dir_layout.addWidget(self.btn_browse)
        layout.addWidget(dir_group)

        scan_group = QGroupBox("扫描选项")
        scan_layout = QFormLayout(scan_group)
        self.chk_recursive = QCheckBox("递归扫描子目录")
        self.chk_recursive.setChecked(True)

        self.edt_patterns = QLineEdit("*.txt, *.csv, *.dta, *.nox, *.z, *.zsimpwin")
        self.edt_patterns.setToolTip("多个模式用逗号分隔")

        self.edt_batch_id = QLineEdit()
        self.edt_batch_id.setPlaceholderText("可选：为这批样品指定批次号")

        self.cmb_circuit = QComboBox()
        for tpl in self.template_manager.list_templates():
            self.cmb_circuit.addItem(f"{tpl.name} ({tpl.key})", tpl.key)
        if self.template_manager.default_key:
            idx = self.cmb_circuit.findData(self.template_manager.default_key)
            if idx >= 0:
                self.cmb_circuit.setCurrentIndex(idx)

        scan_layout.addRow("扫描模式:", self.chk_recursive)
        scan_layout.addRow("文件匹配:", self.edt_patterns)
        scan_layout.addRow("批次号:", self.edt_batch_id)
        scan_layout.addRow("等效电路:", self.cmb_circuit)
        layout.addWidget(scan_group)

        opt_group = QGroupBox("处理选项")
        opt_layout = QFormLayout(opt_group)
        self.chk_save_db = QCheckBox("自动保存到本地数据库")
        self.chk_save_db.setChecked(True)
        self.chk_save_db.setEnabled(self.db_manager is not None)

        self.chk_export_excel = QCheckBox("导出参数汇总 Excel")
        self.chk_export_excel.setChecked(True)

        self.chk_export_figures = QCheckBox("生成对比图表")
        self.chk_export_figures.setChecked(True)

        self.chk_individual_plots = QCheckBox("导出每个样品的单独图")
        self.chk_individual_plots.setChecked(False)

        self.cmb_method = QComboBox()
        self.cmb_method.addItem("LM-TRF 非线性最小二乘 (推荐)", "leastsq")
        self.cmb_method.addItem("差分进化 (全局搜索)", "de")

        self.spin_retries = QSpinBox()
        self.spin_retries.setRange(0, 10)
        self.spin_retries.setValue(2)
        self.spin_retries.setSuffix(" 次")
        self.spin_retries.setToolTip("首次拟合失败时的重试次数")

        self.spin_dpi = QSpinBox()
        self.spin_dpi.setRange(72, 600)
        self.spin_dpi.setValue(300)
        self.spin_dpi.setSuffix(" DPI")

        opt_layout.addRow(self.chk_save_db)
        opt_layout.addRow(self.chk_export_excel)
        opt_layout.addRow(self.chk_export_figures)
        opt_layout.addRow(self.chk_individual_plots)
        opt_layout.addRow("拟合方法:", self.cmb_method)
        opt_layout.addRow("失败重试:", self.spin_retries)
        opt_layout.addRow("图像分辨率:", self.spin_dpi)
        layout.addWidget(opt_group)

        self.btn_scan = QPushButton("扫描目录")
        self.btn_scan.clicked.connect(self._on_scan)
        self.btn_start = QPushButton("开始批量处理")
        self.btn_start.clicked.connect(self._on_start)
        self.btn_start.setEnabled(False)
        self.btn_cancel = QPushButton("取消")
        self.btn_cancel.clicked.connect(self._on_cancel)

        btn_row = QHBoxLayout()
        btn_row.addWidget(self.btn_scan)
        btn_row.addStretch()
        btn_row.addWidget(self.btn_start)
        btn_row.addWidget(self.btn_cancel)
        layout.addLayout(btn_row)

        prog_group = QGroupBox("处理进度")
        prog_layout = QVBoxLayout(prog_group)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.lbl_progress = QLabel("就绪")
        self.lbl_progress.setStyleSheet("color: #555;")
        prog_layout.addWidget(self.progress_bar)
        prog_layout.addWidget(self.lbl_progress)
        layout.addWidget(prog_group)

        log_group = QGroupBox("处理日志")
        log_layout = QVBoxLayout(log_group)
        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setFont(QFont("Consolas", 9))
        log_layout.addWidget(self.txt_log)
        layout.addWidget(log_group, stretch=1)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Close)
        buttons.button(QDialogButtonBox.Ok).setText("导入到主界面")
        buttons.button(QDialogButtonBox.Ok).setEnabled(False)
        buttons.button(QDialogButtonBox.Close).setText("关闭")
        buttons.accepted.connect(self._on_import_and_close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.ok_button = buttons.button(QDialogButtonBox.Ok)

    def _on_browse(self):
        directory = QFileDialog.getExistingDirectory(
            self, "选择 EIS 数据目录", "",
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )
        if directory:
            self.edt_dir.setText(directory)

    def _on_scan(self):
        directory = self.edt_dir.text().strip()
        if not directory or not os.path.isdir(directory):
            QMessageBox.warning(self, "提示", "请先选择一个有效的目录")
            return

        options = self._collect_options()
        files = self.processor.scan_directory(directory, options)
        self.txt_log.append(f"扫描目录: {directory}")
        self.txt_log.append(f"找到 {len(files)} 个匹配文件:")
        for i, f in enumerate(files, 1):
            self.txt_log.append(f"  [{i}] {os.path.basename(f)}")
        self.txt_log.append("-" * 50)

        if len(files) > 0:
            self.btn_start.setEnabled(True)
            self.lbl_progress.setText(f"找到 {len(files)} 个文件，可以开始处理")
        else:
            self.btn_start.setEnabled(False)
            self.lbl_progress.setText("未找到匹配的文件，请检查文件匹配模式")

    def _collect_options(self) -> BatchProcessingOptions:
        patterns_raw = self.edt_patterns.text()
        patterns = [p.strip() for p in patterns_raw.split(",") if p.strip()]
        if not patterns:
            patterns = ["*.txt", "*.csv"]

        method = self.cmb_method.currentData() or "leastsq"

        return BatchProcessingOptions(
            template_key=self.cmb_circuit.currentData() or "R_s-R_para-C_para",
            recursive=self.chk_recursive.isChecked(),
            file_patterns=patterns,
            batch_id=self.edt_batch_id.text().strip(),
            auto_save_db=self.chk_save_db.isChecked(),
            export_excel=self.chk_export_excel.isChecked(),
            export_figures=self.chk_export_figures.isChecked(),
            show_individual_plots=self.chk_individual_plots.isChecked(),
            figure_dpi=self.spin_dpi.value(),
            fitting_method=method,
            max_retries=self.spin_retries.value(),
        )

    def _on_start(self):
        directory = self.edt_dir.text().strip()
        if not directory or not os.path.isdir(directory):
            QMessageBox.warning(self, "提示", "请先选择一个有效的目录")
            return

        options = self._collect_options()
        self._set_controls_enabled(False)
        self.txt_log.append("=" * 50)
        self.txt_log.append(f"开始批量处理 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.txt_log.append(f"等效电路: {self.cmb_circuit.currentText()}")
        self.txt_log.append(f"拟合方法: {self.cmb_method.currentText()}")
        self.txt_log.append("=" * 50)

        self.worker_thread = _BatchWorkerThread(self.processor, directory, options)
        self.worker_thread.progress_signal.connect(self._on_progress)
        self.worker_thread.finished_signal.connect(self._on_finished)
        self.worker_thread.start()

    def _on_progress(self, current: int, total: int, message: str):
        if total > 0:
            pct = int(current * 100 / total)
            self.progress_bar.setValue(pct)
        self.lbl_progress.setText(message)
        self.txt_log.append(f"[{current}/{total}] {message}")
        sb = self.txt_log.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _on_finished(self, result: BatchResult):
        self.batch_result = result
        self.progress_bar.setValue(100)
        self._set_controls_enabled(True)

        self.txt_log.append("=" * 50)
        self.txt_log.append(f"处理完成！")
        self.txt_log.append(f"总文件数: {result.total_files}")
        self.txt_log.append(f"成功: {result.success_count} ({result.success_rate:.1f}%)")
        self.txt_log.append(f"失败: {result.failed_count}")
        self.txt_log.append(f"总耗时: {result.duration_seconds:.2f} 秒")
        if result.output_dir:
            self.txt_log.append(f"输出目录: {result.output_dir}")
        if result.excel_path:
            self.txt_log.append(f"Excel 报告: {result.excel_path}")
        if result.summary_figure_path:
            self.txt_log.append(f"对比图表: {result.summary_figure_path}")

        if result.failed_count > 0:
            self.txt_log.append("\n失败文件列表:")
            for fr in result.results:
                if not fr.success:
                    self.txt_log.append(f"  - {fr.file_name}: {fr.error_message}")
        self.txt_log.append("=" * 50)

        sb = self.txt_log.verticalScrollBar()
        sb.setValue(sb.maximum())

        self.lbl_progress.setText(
            f"完成: {result.success_count}/{result.total_files} 成功, 耗时 {result.duration_seconds:.1f}s"
        )
        self.ok_button.setEnabled(result.success_count > 0)
        self.btn_cancel.setText("关闭")

    def _on_cancel(self):
        if self.worker_thread and self.worker_thread.isRunning():
            reply = QMessageBox.question(
                self, "确认", "正在处理中，确定要取消吗？",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.processor.cancel()
                self.txt_log.append("用户取消了处理")
        else:
            self.reject()

    def _set_controls_enabled(self, enabled: bool):
        self.edt_dir.setEnabled(enabled)
        self.btn_browse.setEnabled(enabled)
        self.chk_recursive.setEnabled(enabled)
        self.edt_patterns.setEnabled(enabled)
        self.edt_batch_id.setEnabled(enabled)
        self.cmb_circuit.setEnabled(enabled)
        self.chk_save_db.setEnabled(enabled and self.db_manager is not None)
        self.chk_export_excel.setEnabled(enabled)
        self.chk_export_figures.setEnabled(enabled)
        self.chk_individual_plots.setEnabled(enabled)
        self.cmb_method.setEnabled(enabled)
        self.spin_retries.setEnabled(enabled)
        self.spin_dpi.setEnabled(enabled)
        self.btn_scan.setEnabled(enabled)
        self.btn_start.setEnabled(enabled)
        if enabled:
            self.btn_cancel.setText("取消")

    def _on_import_and_close(self):
        self.accept()

    def get_importable_samples(self):
        """返回可导入主界面的样品列表"""
        if not self.batch_result:
            return []
        samples = []
        for fr in self.batch_result.results:
            if fr.success and fr.data and fr.data.is_valid:
                samples.append((fr.data, fr.fitting_result))
        return samples


from datetime import datetime
