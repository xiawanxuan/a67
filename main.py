#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EIS 阻抗谱分析系统 - 新能源材料实验室
主程序入口
"""

import sys
import os

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

from eis_app.ui import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("EIS 阻抗谱分析系统")
    app.setOrganizationName("新能源材料实验室")
    app.setOrganizationDomain("eis-analysis.local")

    try:
        font = QFont("Microsoft YaHei", 9)
        app.setFont(font)
    except Exception:
        pass

    app.setAttribute(Qt.AA_DontShowShortcutsInContextMenus, False)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
