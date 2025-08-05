import os
import time

from src import abstract_task, load_data


class filter_csv(abstract_task):
    @classmethod
    def task_name(cls):
        return 'filter_csv'
    
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
        csv_to_filter = self.receipt['steps'][self.step_name]['columns_to_filter_on']
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





import pandas as pd
import os
import numpy as np
from magicgui import magicgui, widgets
from qtpy.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

class CSVFilterGUI(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.df = None
        self.original_df = None
        self.sliders = {}
        self.hist_canvases = {}

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Step 1: File selection
        self.file_button = QPushButton("Select CSV File")
        self.file_button.clicked.connect(self.select_file)
        self.layout.addWidget(self.file_button)

        self.column_dropdown = widgets.ComboBox(label="Select column")
        self.column_dropdown.changed.connect(self.update_histogram_and_slider)
        self.layout.addWidget(self.column_dropdown.native)

        # Z-score filter toggle
        self.zscore_checkbox = widgets.CheckBox(text="Filter on z-score")
        self.layout.addWidget(self.zscore_checkbox.native)

        # Output label
        self.result_label = QLabel("")
        self.layout.addWidget(self.result_label)

        # Filter button
        self.filter_button = QPushButton("Apply Filter")
        self.filter_button.clicked.connect(self.apply_filter)
        self.layout.addWidget(self.filter_button)

        # Save button
        self.save_button = QPushButton("Save Filtered CSV")
        self.save_button.clicked.connect(self.save_filtered_csv)
        self.layout.addWidget(self.save_button)

    def select_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open CSV", os.getcwd(), "CSV Files (*.csv)")
        if not path:
            return

        self.df = pd.read_csv(path)
        self.original_df = self.df.copy()
        self.result_label.setText(f"Loaded {len(self.df)} rows.")

        # Auto-select numeric columns
        numeric_cols = self.df.select_dtypes(include=np.number).columns.tolist()
        self.column_dropdown.choices = numeric_cols

    def update_histogram_and_slider(self, colname):
        if colname is None or colname == '':
            return

        values = self.df[colname].dropna()
        min_val, max_val = float(values.min()), float(values.max())

        # Histogram
        fig, ax = plt.subplots()
        ax.hist(values, bins=50, color='skyblue')
        ax.set_title(f"Histogram: {colname}")
        canvas = FigureCanvas(fig)

        # Remove previous histogram and sliders
        for widget in self.hist_canvases.values():
            self.layout.removeWidget(widget)
            widget.setParent(None)

        self.hist_canvases[colname] = canvas
        self.layout.addWidget(canvas)

        # Add sliders
        self.min_slider = widgets.FloatSlider(min=min_val, max=max_val, step=(max_val-min_val)/100, value=min_val, label="Min")
        self.max_slider = widgets.FloatSlider(min=min_val, max=max_val, step=(max_val-min_val)/100, value=max_val, label="Max")

        self.layout.addWidget(self.min_slider.native)
        self.layout.addWidget(self.max_slider.native)

        self.sliders[colname] = (self.min_slider, self.max_slider)

    def apply_filter(self):
        df = self.original_df.copy()
        for colname, (min_slider, max_slider) in self.sliders.items():
            min_val = min_slider.value
            max_val = max_slider.value

            if self.zscore_checkbox.value:
                z = (df[colname] - df[colname].mean()) / df[colname].std()
                df = df[(z >= min_val) & (z <= max_val)]
            else:
                df = df[(df[colname] >= min_val) & (df[colname] <= max_val)]

        self.df = df
        self.result_label.setText(f"Filtered to {len(df)} rows.")

    def save_filtered_csv(self):
        if self.df is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save CSV", os.getcwd(), "CSV Files (*.csv)")
        if path:
            self.df.to_csv(path, index=False)
            self.result_label.setText(f"Saved to {path}")































