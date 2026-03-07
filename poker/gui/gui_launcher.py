from PyQt6 import uic
from PyQt6.QtWidgets import QMainWindow, QDialog
from pathlib import Path


# Directory assoluta dove stanno i file .ui
_UI_DIR = Path(__file__).resolve().parent / "ui"


def load_ui(ui_filename: str, baseinstance):
    ui_path = _UI_DIR / ui_filename
    uic.loadUi(str(ui_path), baseinstance)


class AnalyserForm(QMainWindow):
    def __init__(self):
        super().__init__()
        load_ui("analyser_form.ui", self)
        self.show()


class TableSetupForm(QMainWindow):
    def __init__(self):
        super().__init__()
        load_ui("table_setup_form.ui", self)
        self.show()


class SetupForm(QMainWindow):
    def __init__(self):
        super().__init__()
        load_ui("setup_form.ui", self)
        self.show()


class StrategyEditorForm(QMainWindow):
    def __init__(self):
        super().__init__()
        load_ui("strategy_manager_form.ui", self)
        self.show()


class GeneticAlgo(QDialog):
    def __init__(self):
        super().__init__()
        load_ui("genetic_algo_form.ui", self)
        self.show()


class MainForm(QMainWindow):
    def __init__(self):
        super().__init__()
        load_ui("main_form.ui", self)
        self.show()


class UiPokerbot(QMainWindow):
    def __init__(self):
        super().__init__()
        load_ui("main_form.ui", self)
        self.show()
