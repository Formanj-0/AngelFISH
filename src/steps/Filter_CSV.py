import os
import time
import pandas as pd
import numpy as np
import sys
from magicgui import magicgui, widgets
from qtpy.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog
from magicgui.widgets import Container, ComboBox, CheckBox, FloatSlider, PushButton
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QCheckBox, QPushButton, QLineEdit, QScrollArea, QFileDialog, QComboBox
)
from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

from src import abstract_task, load_data


class filter_csv(abstract_task):
    @classmethod
    def task_name(cls):
        return 'filter_csv'
    
    @property
    def required_keys(self):
        return ['columns_to_filter_on', 'values_to_filter_on']
    
    def process(self, 
                new_params:dict = None, 
                p_range = None, 
                t_range = None,
                run_in_parallel:bool = False):

        start_time = time.time() 

        # loads data associated with receipt using data_loader
        self.data = load_data(self.receipt)

        # change parameters at run time
        if new_params:
            for k, v in new_params.items():
                self.receipt['steps'][self.step_name][k] = v

        # filter a csv by params
        csv_to_filter = self.receipt['steps'][self.step_name]['csv_to_filter']
        columns_to_filter_on = self.receipt['steps'][self.step_name]['columns_to_filter_on']
        values_to_filter_on = self.receipt['steps'][self.step_name]['values_to_filter_on']
        filter_on_std = self.receipt['steps'][self.step_name].get('filter_on_std', False)

        assert len(columns_to_filter_on) == len(values_to_filter_on), 'each column must have a range to filter on'

        df = self.data[csv_to_filter]

        for i, col_name in enumerate(columns_to_filter_on):
            range_to_filter_on = values_to_filter_on[i] # this will have form (min value, max value)
            min_value = range_to_filter_on[0]
            max_value = range_to_filter_on[1]

            if filter_on_std:
                col_mean = df[col_name].mean()
                col_std = df[col_name].std()
                z_scores = (df[col_name] - col_mean) / col_std
                df = df[(z_scores >= min_value) & (z_scores <= max_value)]
            else:
                df = df[(df[col_name] >= min_value) & (df[col_name] <= max_value)]

        output_csv_path = os.path.join(self.receipt['dirs']['results_dir'], f'{csv_to_filter}_filtered.csv')
        df.to_csv(output_csv_path, index=False)

        # records completion. This will mark completion for luigi
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        with open(self.output_path, "a") as f:
            f.write(f"{self.step_name} completed in {time.time() - start_time:.2f} seconds\n")

        return self.receipt

    def gui(self):
        self.data = load_data(self.receipt)

        dataframes = {k: v for k, v in self.data.items() if isinstance(v, pd.DataFrame)}
        if not dataframes:
            raise ValueError("No DataFrames found in self.data")

        app = QApplication.instance()

        self.filter_window = FilterWindow(dataframes)
        self.filter_window.hist_win.show()  # show histogram window explicitly

        original_close = self.filter_window.closeEvent

        def on_close(event):
            filters = self.filter_window.get_filters()
            df_key = self.filter_window.df_selector.currentText()
            use_z = self.filter_window.global_zscore.isChecked()

            self.receipt['steps'][self.step_name]['columns_to_filter_on'] = list(filters.keys())
            self.receipt['steps'][self.step_name]['values_to_filter_on'] = list(filters.values())
            self.receipt['steps'][self.step_name]['filter_on_std'] = use_z
            self.receipt['steps'][self.step_name]['csv_to_filter'] = df_key

            self.process()

            original_close(event)

        self.filter_window.closeEvent = on_close

        self.filter_window.show()

        app.exec_()

        return self.receipt



class HistogramWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Histograms")
        self.layout = QVBoxLayout()
        self.canvas = FigureCanvas(plt.figure(figsize=(8, 6)))
        self.layout.addWidget(self.canvas)
        self.setLayout(self.layout)

    def update_histograms(self, df, filters, use_zscore):
        fig = self.canvas.figure
        fig.clear()

        if not filters:
            self.canvas.draw()
            return

        row_mask = pd.Series(True, index=df.index)
        for col, (min_val, max_val) in filters.items():
            data = df[col]
            if use_zscore:
                data = (data - data.mean()) / data.std()
            row_mask &= (data >= min_val) & (data <= max_val)

        df_filtered = df[row_mask]
        n = len(filters)

        for i, (col, (min_val, max_val)) in enumerate(filters.items()):
            ax = fig.add_subplot(n, 1, i + 1)

            full_data = df[col].dropna()
            accepted_data = df_filtered[col].dropna()

            if use_zscore:
                mean = df[col].mean()
                std = df[col].std()
                full_data = (full_data - mean) / std
                accepted_data = (accepted_data - mean) / std

            bins = np.linspace(full_data.min(), full_data.max(), 100)
            is_accepted = full_data.index.isin(accepted_data.index)
            rejected_data = full_data[~is_accepted]

            ax.hist(rejected_data, bins=bins, color='red', alpha=0.6, label='Rejected')
            ax.hist(accepted_data, bins=bins, color='blue', alpha=0.6, label='Accepted')

            ax.set_title(f"{col}")
            ax.legend()

        self.canvas.draw()

from PyQt5.QtCore import QTimer

class FilterWindow(QWidget):
    def __init__(self, data: dict):
        super().__init__()
        self.setWindowTitle("Filter CSV Columns")
        self.data = data
        self.current_df_name = None
        self.current_df = None
        self.numeric_cols = []
        self.selected_columns = {}
        self.sliders = {}

        self.hist_win = HistogramWindow()
        # Don't show here!
        # self.hist_win.show()

        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)

        self.df_selector = QComboBox()
        self.df_selector.addItems([k for k, v in self.data.items() if isinstance(v, pd.DataFrame)])
        self.df_selector.currentTextChanged.connect(self.on_df_changed)
        self.main_layout.addWidget(QLabel("Select DataFrame:"))
        self.main_layout.addWidget(self.df_selector)

        self.global_zscore = QCheckBox("Use Z-Score for all filters")
        self.global_zscore.stateChanged.connect(self.schedule_update)
        self.main_layout.addWidget(self.global_zscore)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout()
        self.content_widget.setLayout(self.content_layout)
        self.scroll.setWidget(self.content_widget)
        self.main_layout.addWidget(self.scroll)

        self.run_button = QPushButton("Run Filter")
        self.run_button.clicked.connect(self.schedule_update)
        self.main_layout.addWidget(self.run_button)

        self._update_timer = QTimer()
        self._update_timer.setSingleShot(True)
        self._update_timer.timeout.connect(self.update_histograms)

        self.on_df_changed(self.df_selector.currentText())

    def on_df_changed(self, df_name):
        self.current_df_name = df_name
        self.current_df = self.data[df_name]
        self.numeric_cols = self.current_df.select_dtypes(include=np.number).columns.tolist()
        self.rebuild_filter_controls()

    def schedule_update(self):
        self._update_timer.start(100)  # 100ms delay

    def rebuild_filter_controls(self):
        # Clear layout
        while self.content_layout.count():
            item = self.content_layout.takeAt(0)
            if item.layout():
                while item.layout().count():
                    w = item.layout().takeAt(0).widget()
                    if w:
                        w.deleteLater()
                item.layout().deleteLater()
            elif item.widget():
                item.widget().deleteLater()

        self.selected_columns.clear()
        self.sliders.clear()

        for col in self.numeric_cols:
            col_layout = QHBoxLayout()

            checkbox = QCheckBox(col)
            checkbox.blockSignals(True)
            checkbox.setChecked(False)
            checkbox.blockSignals(False)
            checkbox.stateChanged.connect(self.schedule_update)
            self.selected_columns[col] = checkbox
            col_layout.addWidget(checkbox)

            min_val, max_val = self.get_col_range(col)

            min_input = QLineEdit(str(min_val))
            min_input.setValidator(QDoubleValidator())
            min_input.blockSignals(True)
            min_input.setText(str(min_val))
            min_input.blockSignals(False)
            min_input.editingFinished.connect(self.schedule_update)

            max_input = QLineEdit(str(max_val))
            max_input.setValidator(QDoubleValidator())
            max_input.blockSignals(True)
            max_input.setText(str(max_val))
            max_input.blockSignals(False)
            max_input.editingFinished.connect(self.schedule_update)

            self.sliders[col] = (min_input, max_input)
            col_layout.addWidget(QLabel("Min"))
            col_layout.addWidget(min_input)
            col_layout.addWidget(QLabel("Max"))
            col_layout.addWidget(max_input)

            self.content_layout.addLayout(col_layout)

        self.schedule_update()

    def get_col_range(self, col):
        data = self.current_df[col].dropna()
        if self.global_zscore.isChecked():
            zdata = (data - data.mean()) / data.std()
            return float(np.floor(zdata.min())), float(np.ceil(zdata.max()))
        else:
            return float(data.min()), float(data.max())

    def get_filters(self):
        filters = {}
        for col in self.numeric_cols:
            if self.selected_columns[col].isChecked():
                try:
                    min_val = float(self.sliders[col][0].text())
                    max_val = float(self.sliders[col][1].text())
                    filters[col] = (min_val, max_val)
                except ValueError:
                    continue
        return filters

    def update_histograms(self):
        if self.current_df is not None:
            filters = self.get_filters()
            if not filters:
                self.hist_win.canvas.figure.clear()
                self.hist_win.canvas.draw()
                return
            use_zscore = self.global_zscore.isChecked()
            self.hist_win.update_histograms(self.current_df, filters, use_zscore)

    def closeEvent(self, event):
        # if self.current_df is not None:
        #     filters = self.get_filters()
        #     use_zscore = self.global_zscore.isChecked()
        #     df_filtered = self.current_df.copy()

        #     for col, (min_val, max_val) in filters.items():
        #         data = df_filtered[col]
        #         if use_zscore:
        #             data = (data - data.mean()) / data.std()
        #         df_filtered = df_filtered[(data >= min_val) & (data <= max_val)]

        #     save_path, _ = QFileDialog.getSaveFileName(self, "Save Filtered CSV", "", "CSV Files (*.csv)")
        #     if save_path:
        #         df_filtered.to_csv(save_path, index=False)

        event.accept()