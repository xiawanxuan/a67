from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import QMenu, QMenuBar


@dataclass
class MenuAction:
    name: str
    text: str
    callback: Optional[Callable]
    shortcut: str = ""
    status_tip: str = ""
    icon_text: str = ""


class MenuManager:
    """菜单栏布局管理器"""

    def __init__(self, menu_bar: QMenuBar):
        self.menu_bar = menu_bar
        self._menus: Dict[str, QMenu] = {}
        self._actions: Dict[str, QAction] = {}

    def create_file_menu(self, actions: Dict[str, Callable]):
        menu = self.menu_bar.addMenu("文件(&F)")
        self._menus["file"] = menu
        self._add_action(menu, "open", "打开文件...", actions.get("open"),
                         shortcut="Ctrl+O", status_tip="打开 EIS 阻抗谱数据文件")
        self._add_action(menu, "open_batch", "批量导入...", actions.get("open_batch"),
                         shortcut="Ctrl+Shift+O", status_tip="批量导入多个 EIS 数据文件")
        menu.addSeparator()
        self._add_action(menu, "export_single_csv", "导出当前样品CSV", actions.get("export_single_csv"),
                         status_tip="导出当前选中样品为CSV")
        self._add_action(menu, "export_batch_csv", "批量导出CSV", actions.get("export_batch_csv"),
                         status_tip="导出所有样品到CSV")
        self._add_action(menu, "export_params", "导出拟合参数", actions.get("export_params"),
                         status_tip="导出拟合参数汇总CSV")
        self._add_action(menu, "export_excel", "导出Excel报告", actions.get("export_excel"),
                         status_tip="导出包含原始数据和拟合参数的Excel")
        menu.addSeparator()
        self._add_action(menu, "save_figure", "保存图像...", actions.get("save_figure"),
                         shortcut="Ctrl+S", status_tip="保存当前绘图为图片")
        menu.addSeparator()
        self._add_action(menu, "exit", "退出", actions.get("exit"),
                         shortcut="Ctrl+Q", status_tip="退出程序")

    def create_edit_menu(self, actions: Dict[str, Callable]):
        menu = self.menu_bar.addMenu("编辑(&E)")
        self._menus["edit"] = menu
        self._add_action(menu, "delete_sample", "删除样品", actions.get("delete_sample"),
                         shortcut="Del", status_tip="删除选中样品")
        self._add_action(menu, "clear_all", "清空所有", actions.get("clear_all"),
                         status_tip="清空所有样品")
        menu.addSeparator()
        self._add_action(menu, "auto_scale", "自动缩放", actions.get("auto_scale"),
                         shortcut="Ctrl+R", status_tip="自动调整坐标轴范围")

    def create_fitting_menu(self, actions: Dict[str, Callable]):
        menu = self.menu_bar.addMenu("拟合(&I)")
        self._menus["fitting"] = menu
        self._add_action(menu, "fit_current", "拟合当前样品", actions.get("fit_current"),
                         shortcut="Ctrl+F", status_tip="使用当前等效电路拟合选中样品")
        self._add_action(menu, "fit_all", "拟合全部样品", actions.get("fit_all"),
                         shortcut="Ctrl+Shift+F", status_tip="拟合所有样品")
        menu.addSeparator()
        self._add_action(menu, "toggle_show_fit", "显示拟合曲线", actions.get("toggle_show_fit"),
                         status_tip="切换显示/隐藏拟合曲线")
        self._add_action(menu, "circuit_template", "等效电路模板...", actions.get("circuit_template"),
                         status_tip="管理等效电路模板")

    def create_view_menu(self, actions: Dict[str, Callable]):
        menu = self.menu_bar.addMenu("视图(&V)")
        self._menus["view"] = menu
        self._add_action(menu, "set_axis_limits", "设置坐标范围...", actions.get("set_axis_limits"),
                         status_tip="自定义横纵轴显示范围")

    def create_database_menu(self, actions: Dict[str, Callable]):
        menu = self.menu_bar.addMenu("数据库(&D)")
        self._menus["database"] = menu
        self._add_action(menu, "save_to_db", "保存到数据库", actions.get("save_to_db"),
                         shortcut="Ctrl+Shift+S", status_tip="保存样品到本地数据库")
        self._add_action(menu, "load_from_db", "从数据库加载", actions.get("load_from_db"),
                         shortcut="Ctrl+L", status_tip="从本地数据库加载样品")
        self._add_action(menu, "manage_batches", "批次管理...", actions.get("manage_batches"),
                         status_tip="管理样品批次")

    def create_help_menu(self, actions: Dict[str, Callable]):
        menu = self.menu_bar.addMenu("帮助(&H)")
        self._menus["help"] = menu
        self._add_action(menu, "about", "关于", actions.get("about"),
                         status_tip="关于本软件")

    def _add_action(self, menu: QMenu, name: str, text: str,
                     callback: Optional[Callable], shortcut: str = "",
                     status_tip: str = ""):
        action = QAction(text, menu)
        if shortcut:
            action.setShortcut(QKeySequence(shortcut))
        if status_tip:
            action.setStatusTip(status_tip)
        if callback:
            action.triggered.connect(callback)
        menu.addAction(action)
        self._actions[name] = action
        return action

    def set_action_enabled(self, name: str, enabled: bool):
        if name in self._actions:
            self._actions[name].setEnabled(enabled)

    def get_action(self, name: str) -> Optional[QAction]:
        return self._actions.get(name)
