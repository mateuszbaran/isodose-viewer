import diskcache
import nibabel
import napari
import pydicom

import diskcache as dc
import traceback

from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QPushButton, QFileDialog, QMessageBox, QGridLayout, \
    QLabel, QVBoxLayout, QHBoxLayout, QComboBox, QDoubleSpinBox

from PyQt5 import QtCore

import pyqtgraph as pg

import sys
import json
from common.dicom_tools import read_dicom_3d, read_tps_dose, draw_contours, display_structure_ids_ds, \
    rename_CTs_to_SOPInstanceUID, resample_doses_to_ct

from common.qa_utils import calculate_dvh, calculate_hot_cold_vols

import vispy.color

import numpy as np


def load_data(
        ct_fname='/home/mateusz/Desktop/tmp/isodose-data/CT.nii.gz',
        dose_measured_fname='/home/mateusz/Desktop/tmp/isodose-data/Dose_3DVH.nii.gz',
        dose_planned_fname='/home/mateusz/Desktop/tmp/isodose-data/Dose_TPS.nii.gz'
):
    ct = nibabel.load(ct_fname)
    dose_measured = nibabel.load(dose_measured_fname)
    dose_planned = nibabel.load(dose_planned_fname)
    return ct, dose_measured, dose_planned


class App(QWidget):
    def __init__(self):
        super().__init__()
        self.sel_isodose_line = None
        self.dd_plot_widget = None
        self.dvh_plot_widget = None
        self.isodose_spinbox = None
        self.roi_combobox = None
        self.planned_dose_ct = None
        self.measured_dose_ct = None
        self.status_label = None
        self.rois = None
        self.dicom_ct_dir = None
        self.contours = None
        self.rtstruct_label = None
        self.configs = None
        self.ct = None
        self.measured_dose = None
        self.planned_dose = None
        self.ct_label = None
        self.measured_dose_label = None
        self.planned_dose_label = None
        self.title = 'Isodose explorer'
        self.left = 10
        self.top = 10
        self.width = 1024
        self.height = 768
        self.cache = diskcache.Cache("tmp")
        self.cache.reset('cull_limit', 0)
        self.initUI()
        self.load_config()
        self.update_dvh_plot()

    def load_config(self):
        config_fname = 'config.json'
        with open(config_fname, 'r') as config_file:
            self.configs = json.load(config_file)

        self.update_ct(self.configs['CTDir'])
        self.update_measured_dose(self.configs['3DVHdoseFname'])
        self.update_planned_dose(self.configs['TPSdoseFname'])
        self.update_rtstruct(self.configs['RtStructFname'])

    def update_dvh_plot(self):
        selected_roi_id = self.roi_combobox.currentData()
        if selected_roi_id is None:
            return

        self.dvh_plot_widget.clear()
        self.dvh_plot_widget.addItem(self.sel_isodose_line)
        self.dvh_plot_widget.addLegend()
        dose_range = (float(self.configs['DVHRange'][0]), float(self.configs['DVHRange'][1]))
        nbins = self.configs['DVHBins']

        planned_doses_for_dvh = self.planned_dose_ct[self.contours[selected_roi_id] != 0]
        mod_dose_hist, mod_vol_hist = calculate_dvh(planned_doses_for_dvh, dose_range, nbins)
        self.dvh_plot_widget.plot(mod_dose_hist, mod_vol_hist, pen=(1, 3), name="Planned dose")

        measured_doses_for_dvh = self.measured_dose_ct[self.contours[selected_roi_id] != 0]
        mod_dose_hist, mod_vol_hist = calculate_dvh(measured_doses_for_dvh, dose_range, nbins)
        self.dvh_plot_widget.plot(mod_dose_hist, mod_vol_hist, pen=(2, 4), name="Measured dose")

        hot_vols, cold_vols = calculate_hot_cold_vols(planned_doses_for_dvh, measured_doses_for_dvh, mod_dose_hist)
        self.dvh_plot_widget.plot(mod_dose_hist, hot_vols, pen=pg.mkPen(color=(210, 10, 10)), name="Hot volume")
        self.dvh_plot_widget.plot(mod_dose_hist, cold_vols, pen=pg.mkPen(color=(10, 10, 210)), name="Cold volume")

        self.dvh_plot_widget.setXRange(dose_range[0], dose_range[1])

    def recalculate_plots_for_isodose(self):
        # DD plot
        isodose_Gy = self.isodose_spinbox.value()
        self.sel_isodose_line.setValue(isodose_Gy)

        selected_roi_id = self.roi_combobox.currentData()
        if selected_roi_id is None:
            return

        cold_region = np.logical_and(
            np.logical_and(self.planned_dose_ct >= isodose_Gy, self.measured_dose_ct <= isodose_Gy),
            self.contours[selected_roi_id])
        hot_region = np.logical_and(
            np.logical_and(self.planned_dose_ct <= isodose_Gy, self.measured_dose_ct >= isodose_Gy),
            self.contours[selected_roi_id])
        diffs_cold = (self.measured_dose_ct - self.planned_dose_ct)[cold_region != 0]
        diffs_hot = (self.measured_dose_ct - self.planned_dose_ct)[hot_region != 0]

        self.dd_plot_widget.clear()
        self.dd_plot_widget.addLegend()
        roi_voxel_count = np.count_nonzero(self.contours[selected_roi_id])
        cold_hist, cold_hist_bin_edges = np.histogram(diffs_cold, density=False)
        hot_hist, hot_hist_bin_edges = np.histogram(diffs_hot, density=False)
        bargraph_cold = pg.BarGraphItem(x0=cold_hist_bin_edges[:-1], x1=cold_hist_bin_edges[1:],
                                        height=cold_hist / roi_voxel_count, name="Cold region",
                                        brush=pg.mkBrush(color=(0, 0, 200)))
        bargraph_hot = pg.BarGraphItem(x0=hot_hist_bin_edges[:-1], x1=hot_hist_bin_edges[1:],
                                       height=hot_hist / roi_voxel_count, name="Hot region",
                                       brush=pg.mkBrush(color=(200, 0, 0)))
        self.dd_plot_widget.addItem(bargraph_cold)
        self.dd_plot_widget.addItem(bargraph_hot)
        self.dd_plot_widget.setXRange(cold_hist_bin_edges[0], hot_hist_bin_edges[-1])

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        self.status_label = QLabel("Ready", self)

        button_open_ct = QPushButton(self)
        button_open_ct.setText("Open CT directory")
        button_open_ct.clicked.connect(self.open_file_ct_clicked)
        self.ct_label = QLabel("no CT loaded", self)
        self.ct_label.setMaximumWidth(200)
        self.ct_label.setWordWrap(True)

        button_open_rtstruct = QPushButton(self)
        button_open_rtstruct.setText("Open RTStruct file")
        button_open_rtstruct.clicked.connect(self.open_file_rtstruct_clicked)
        self.rtstruct_label = QLabel("no RTStruct loaded", self)
        self.rtstruct_label.setMaximumWidth(200)
        self.rtstruct_label.setWordWrap(True)

        button_open_measured = QPushButton(self)
        button_open_measured.setText("Open measured dose file")
        button_open_measured.clicked.connect(self.open_file_measured_clicked)
        self.measured_dose_label = QLabel("no measured dose loaded", self)
        self.measured_dose_label.setMaximumWidth(200)
        self.measured_dose_label.setWordWrap(True)

        button_open_planned = QPushButton(self)
        button_open_planned.setText("Open planned dose file")
        button_open_planned.clicked.connect(self.open_file_planned_clicked)
        self.planned_dose_label = QLabel("no planned dose loaded", self)
        self.planned_dose_label.setMaximumWidth(200)
        self.planned_dose_label.setWordWrap(True)

        button_browse_slices = QPushButton(self)
        button_browse_slices.setText("Browse slices")
        button_browse_slices.clicked.connect(self.browse_slices)

        self.roi_combobox = QComboBox()
        self.roi_combobox.addItem("Select ROI")
        self.roi_combobox.currentIndexChanged.connect(self.update_roi_selection)

        self.isodose_spinbox = QDoubleSpinBox()
        self.isodose_spinbox.setRange(0.0, 100.0)
        self.isodose_spinbox.setPrefix("Isodose: ")
        self.isodose_spinbox.setSuffix(" Gy")
        self.isodose_spinbox.setValue(60.0)
        self.isodose_spinbox.valueChanged.connect(self.update_isodose_selection)

        self.dvh_plot_widget = pg.plot(title="DVH")
        self.dvh_plot_widget.setLabel('left', 'Volume (relative to volume of ROI)')
        self.dvh_plot_widget.setLabel('bottom', 'Dose [Gy]')
        self.sel_isodose_line = pg.InfiniteLine(self.isodose_spinbox.value(), pen=pg.mkPen(color=(200, 200, 255), style=QtCore.Qt.DotLine))
        self.dvh_plot_widget.addItem(self.sel_isodose_line)

        self.dd_plot_widget = pg.plot(title="Dose difference (measured - planned)")
        self.dd_plot_widget.setLabel('left', 'Fraction of region volume')
        self.dd_plot_widget.setLabel('bottom', 'Dose difference [Gy]')

        outer_layout = QVBoxLayout()
        grid_layout = QGridLayout()
        plot_layout = QHBoxLayout()
        # Add widgets to the layout
        grid_layout.addWidget(button_open_ct, 0, 0)
        grid_layout.addWidget(self.ct_label, 1, 0)
        grid_layout.addWidget(button_open_rtstruct, 0, 1)
        grid_layout.addWidget(self.rtstruct_label, 1, 1)
        grid_layout.addWidget(button_open_planned, 0, 2)
        grid_layout.addWidget(self.planned_dose_label, 1, 2)
        grid_layout.addWidget(button_open_measured, 0, 3)
        grid_layout.addWidget(self.measured_dose_label, 1, 3)
        grid_layout.addWidget(button_browse_slices, 0, 4)

        plot_layout.addWidget(self.dvh_plot_widget)
        plot_layout.addWidget(self.dd_plot_widget)
        # Set the layout on the application's window
        outer_layout.addLayout(grid_layout)
        outer_layout.addWidget(self.roi_combobox)
        outer_layout.addWidget(self.isodose_spinbox)
        outer_layout.addLayout(plot_layout)
        outer_layout.addWidget(self.status_label)
        self.setLayout(outer_layout)
        self.show()

    def close(self):
        self.cache.close()

    def display_warning(self, msg):
        msgBox = QMessageBox()
        msgBox.setIcon(QMessageBox.Warning)
        msgBox.setText(msg)
        msgBox.setStandardButtons(QMessageBox.Ok)
        msgBox.exec()

    def update_ct(self, fname):
        try:
            rename_CTs_to_SOPInstanceUID(fname)
            cache_key = f"CT file {fname}"
            self.ct = self.cache.get(cache_key, None)
            self.dicom_ct_dir = fname
            if self.ct is None:
                self.ct = read_dicom_3d(fname)
                self.cache.add(cache_key, self.ct)

            self.ct_label.setText("Loaded CT:\n" + fname)
        except Exception as e:
            print(e)
            print(traceback.format_exc())
            self.display_warning(f"Could not load {fname}:\n{e}")

    def update_rtstruct(self, fname):
        try:
            cache_key_struct_dcm = f"self.struct_dcm {fname}"
            # load data
            self.struct_dcm, self.rois, self.contours = self.cache.get(cache_key_struct_dcm, (None, None, None))
            if self.struct_dcm is None:
                self.struct_dcm = pydicom.dcmread(fname)
                self.rois = display_structure_ids_ds(self.struct_dcm)
                self.contours = {}
                for roi in self.rois:
                    self.contours[roi[0]] = draw_contours(self.ct[1].shape, roi[0], self.dicom_ct_dir, self.struct_dcm)
                self.cache.add(cache_key_struct_dcm, (self.struct_dcm, self.rois, self.contours))

            # update UI
            self.roi_combobox.clear()
            self.roi_combobox.addItem("Select ROI")
            for roi in self.rois:
                self.roi_combobox.addItem(roi[1], userData=roi[0])
                self.status_label.setText(f"Reading ROI {roi}")
            self.rtstruct_label.setText("Loaded RTStruct:\n" + fname)
        except Exception as e:
            print(e)
            print(traceback.format_exc())
            self.display_warning(f"Could not load {fname}:\n{e}")
        finally:
            self.status_label.setText("Ready")

    def update_planned_dose(self, fname):
        try:
            cache_key = f"dose file {fname}"
            self.planned_dose, self.planned_dose_ct = self.cache.get(cache_key, (None, None))
            if self.planned_dose is None:
                self.planned_dose = read_tps_dose(fname)
                self.planned_dose_ct = resample_doses_to_ct(self.planned_dose, self.ct)
                self.cache.add(cache_key, (self.planned_dose, self.planned_dose_ct))
            self.planned_dose_label.setText("Loaded planned dose\n" + fname)
        except Exception as e:
            print(e)
            print(traceback.format_exc())
            self.display_warning(f"Could not load {fname}:\n{e}")

    def update_measured_dose(self, fname):
        try:
            cache_key = f"dose file {fname}"
            self.measured_dose, self.measured_dose_ct = self.cache.get(cache_key, (None, None))
            if self.measured_dose is None:
                self.measured_dose = read_tps_dose(fname)
                self.measured_dose_ct = resample_doses_to_ct(self.measured_dose, self.ct)
                self.cache.add(cache_key, (self.measured_dose, self.measured_dose_ct))
            self.measured_dose_label.setText("Loaded measured dose:\n" + fname)
        except Exception as e:
            print(e)
            print(traceback.format_exc())
            self.display_warning(f"Could not load {fname}:\n{e}")

    def update_roi_selection(self):
        self.update_dvh_plot()
        self.recalculate_plots_for_isodose()

    def update_isodose_selection(self):
        self.recalculate_plots_for_isodose()

    def browse_slices(self):
        self.view_napari(self.ct[1], dose_measured=self.measured_dose_ct, dose_planned=self.planned_dose_ct)

    def open_file_ct_clicked(self):
        fname = QFileDialog.getExistingDirectory(self, 'Open CT folder')
        self.update_ct(fname[0])

    def open_file_rtstruct_clicked(self):
        fname = QFileDialog.getOpenFileName(self, 'Open RTStruct file', "RT files (*.dcm)")
        self.update_rtstruct(fname[0])

    def open_file_measured_clicked(self):
        fname = QFileDialog.getOpenFileName(self, 'Open measured dose file', "RT files (*.dcm)")
        self.update_measured_dose(fname[0])

    def open_file_planned_clicked(self):
        fname = QFileDialog.getOpenFileName(self, 'Open planned dose file', "RT files (*.dcm)")
        self.update_planned_dose(fname[0])

    def view_napari(self, ct, dose_measured=None, dose_planned=None):
        ct_scale = tuple((self.ct[0][i][2] - self.ct[0][i][1] for i in (0, 1, 2)))
        viewer = napari.view_image(ct, name="CT", scale=ct_scale)

        green = vispy.color.Colormap([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        red = vispy.color.Colormap([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

        if dose_measured is not None:
            measured_layer = viewer.add_image(dose_measured, name="Measured dose", opacity=0.4, scale=ct_scale)
            measured_layer.colormap = 'red', red
        if dose_planned is not None:
            planned_layer = viewer.add_image(dose_planned, name="Planned dose", opacity=0.4, scale=ct_scale)
            planned_layer.colormap = 'green', green

        viewer.scale_bar.visible = True
        viewer.scale_bar.unit = "mm"
        return viewer


if __name__ == '__main__':
    app = QApplication(sys.argv)
    global ex
    ex = App()
    sys.exit(app.exec_())
    # ct, dose_measured, dose_planned = load_data()
