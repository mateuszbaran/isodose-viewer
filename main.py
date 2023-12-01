import diskcache
import nibabel
import napari
import pydicom

import diskcache as dc
import traceback

from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QPushButton, QFileDialog, QMessageBox, QGridLayout, \
    QLabel, QVBoxLayout, QHBoxLayout, QComboBox, QDoubleSpinBox, QTableWidget, QTableWidgetItem, QAbstractItemView

from PyQt5 import QtCore, QtGui
from PyQt5.QtGui import QColor, QPainter

import pyqtgraph as pg

import sys
import json
from common.dicom_tools import read_dicom_3d, read_tps_dose, draw_contours, display_structure_ids_ds, \
    rename_CTs_to_SOPInstanceUID, resample_doses_to_ct

from common.qa_utils import calculate_dvh, calculate_hot_cold_vols, prepare_hot_cold_image, prepare_cold_val, \
    prepare_hot_val, calculate_Dx, calculate_Vx, calculate_gpr_for_roi, prepare_confusion_matrix

import vispy.color

import numpy as np

import pymedphys


class QAStatWindow(QMainWindow):
    def __init__(self, parent):
        self.parent = parent
        super(QAStatWindow, self).__init__(parent)

        main_layout = QHBoxLayout()
        self.setFixedSize(450, 550)

        self.stat_table = QTableWidget(self)
        self.stat_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        stat_column_headers = ["ROI name", "parameter", "value", "tolerance"]
        self.stat_table.setColumnCount(len(stat_column_headers))
        self.stat_table.setHorizontalHeaderLabels(stat_column_headers)
        self.stat_table.setMinimumHeight(400)
        self.stat_table.setMinimumWidth(400)
        self.stat_table.setMaximumHeight(700)
        self.setWindowTitle("QA Statistics")

        self.specification = self.parent.configs["qa_specification"]
        self.stat_table.setRowCount(len(self.specification))

        for (i, (roi_name, calculation_type, threshold, tolerance_text)) in enumerate(self.specification):
            threshold_unit = ""
            if calculation_type == "Vrel":
                threshold_unit = "%"
            elif calculation_type == "Vabs":
                threshold_unit = "cm³"
            elif calculation_type == "Dmax":
                threshold_unit = "Gy"

            self.stat_table.setItem(i, 0, QTableWidgetItem(roi_name))
            if calculation_type == "Dmax":
                self.stat_table.setItem(i, 1, QTableWidgetItem(f"{calculation_type}"))
            else:
                self.stat_table.setItem(i, 1, QTableWidgetItem(f"{calculation_type} {threshold}{threshold_unit}"))
            self.stat_table.setItem(i, 3, QTableWidgetItem(tolerance_text))

        self.fill_table()

        self.stat_table.resizeColumnsToContents()
        self.stat_table.resizeRowsToContents()

        main_layout.addWidget(self.stat_table)
        self.setLayout(main_layout)

    def fill_table(self):

        for (i, (roi_name, calculation_type, threshold, tolerance_text)) in enumerate(self.specification):
            roi_num = -1
            for roi in self.parent.rois:
                if roi[1] == roi_name:
                    roi_num = roi[0]

            if roi_num != -1:
                mask = self.parent.contours[roi_num]
                if calculation_type in ["Vrel", "Vabs"]:
                    roi_v_perc, roi_v_cm3 = calculate_Vx(self.parent.get_voxel_volume_cm3(), mask,
                                                         self.parent.measured_dose_ct, threshold)
                    if calculation_type == "Vrel":
                        txt = f"{roi_v_perc:.2f}%"
                    else:
                        txt = f"{roi_v_cm3:.2f} cm³"
                    iw = QTableWidgetItem(txt)
                elif calculation_type == "Dmax":
                    Dmax = np.max(self.parent.measured_dose_ct[mask != 0])
                    iw = QTableWidgetItem(f"{Dmax:.2f}Gy")

                try:
                    if calculation_type == "Vrel" and tolerance_text[:2] == "< " and tolerance_text[-1:] == "%":
                        if roi_v_perc < float(tolerance_text[2:-1]):
                            iw.setBackground(QColor(100, 255, 100))
                        else:
                            iw.setBackground(QColor(255, 100, 100))
                    elif calculation_type == "Vabs" and tolerance_text[:2] == "< " and tolerance_text[-3:] == "cm³":
                        if roi_v_cm3 < float(tolerance_text[2:-3]):
                            iw.setBackground(QColor(100, 255, 100))
                        else:
                            iw.setBackground(QColor(255, 100, 100))
                    elif calculation_type == "Dmax" and tolerance_text[:2] == "< " and tolerance_text[-2:] == "Gy":
                        if Dmax < float(tolerance_text[2:-2]):
                            iw.setBackground(QColor(100, 255, 100))
                        else:
                            iw.setBackground(QColor(255, 100, 100))
                except Exception as e:
                    print(e)
                self.stat_table.setItem(i, 2, iw)


def load_data(
        ct_fname='/home/mateusz/Desktop/tmp/isodose-data/CT.nii.gz',
        dose_measured_fname='/home/mateusz/Desktop/tmp/isodose-data/Dose_3DVH.nii.gz',
        dose_planned_fname='/home/mateusz/Desktop/tmp/isodose-data/Dose_TPS.nii.gz'
):
    ct = nibabel.load(ct_fname)
    dose_measured = nibabel.load(dose_measured_fname)
    dose_planned = nibabel.load(dose_planned_fname)
    return ct, dose_measured, dose_planned


class VerticalLabel(QLabel):

    def __init__(self, *args):
        QLabel.__init__(self, *args)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.translate(0, self.height())
        painter.rotate(-90)
        # calculate the size of the font
        fm = QtGui.QFontMetrics(painter.font())
        xoffset = int(fm.boundingRect(self.text()).width() / 2)
        yoffset = int(fm.boundingRect(self.text()).height() / 2)
        x = int(self.width() / 2) + yoffset
        y = int(self.height() / 2) - xoffset
        # because we rotated the label, x affects the vertical placement, and y affects the horizontal
        painter.drawText(y, x, self.text())
        painter.end()

    def minimumSizeHint(self):
        size = QLabel.minimumSizeHint(self)
        return QtCore.QSize(size.height(), size.width())

    def sizeHint(self):
        size = QLabel.sizeHint(self)
        return QtCore.QSize(size.height(), size.width())


class App(QWidget):
    def __init__(self):
        super().__init__()
        self.struct_dcm = None
        self.roi_stat_table = None
        self.confusion_matrix_table = None
        self.gamma_full = None
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
        self.ct_label = None
        self.measured_dose_label = None
        self.planned_dose_label = None
        self.title = 'Isodose explorer'
        self.left = 10
        self.top = 10
        self.load_config()
        self.width = self.configs["window_width"]
        self.height = self.configs["window_height"]
        self.cache = None
        self.initUI()
        self.load_cache()
        self.update_dvh_plot()

    def load_config(self):
        config_fname = 'config.json'
        with open(config_fname, 'r') as config_file:
            self.configs = json.load(config_file)

    def load_cache(self):
        self.cache = diskcache.Cache("tmp", size_limit=int(int(self.configs["cacheLimitGB"]) * 1e9))
        self.cache.reset('cull_limit', 0)

        self.update_ct(self.configs['CTDir'])
        self.update_measured_dose(self.configs['3DVHdoseFname'])
        self.update_planned_dose(self.configs['TPSdoseFname'])
        self.update_rtstruct(self.configs['RtStructFname'])

    def calc_dvh_for_roi(self, doses, roi_id):
        dose_range = (float(self.configs['DVHRange'][0]), float(self.configs['DVHRange'][1]))
        nbins = self.configs['DVHBins']

        doses_for_dvh = doses[self.contours[roi_id] != 0]
        mod_dose_hist, mod_vol_hist = calculate_dvh(doses_for_dvh, dose_range, nbins)
        return mod_dose_hist, mod_vol_hist

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

    def update_gamma(self):
        self.gamma_label.setText("Updating gamma...")
        self.gamma_full = pymedphys.gamma(self.ct[0], self.planned_dose_ct, self.ct[0],
                                          self.measured_dose_ct,
                                          dose_percent_threshold=float(self.configs["gamma"]["doseTol_percent"]),
                                          distance_mm_threshold=float(self.configs["gamma"]["distanceTol_mm"]),
                                          max_gamma=float(self.configs["gamma"]["maxGamma"]),
                                          lower_percent_dose_cutoff=float(
                                              self.configs["gamma"]["lowerDoseCutoffPercent"]),
                                          interp_fraction=int(self.configs["gamma"]["interpFraction"]))
        self.gamma_label.setText("Gamma updated")
        self.update_roi_stat_table()

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

        button_update_gamma = QPushButton(self)
        button_update_gamma.setText("Update gamma analysis")
        button_update_gamma.clicked.connect(self.update_gamma)
        self.gamma_label = QLabel("Gamma not updated", self)
        self.gamma_label.setMaximumWidth(200)
        self.gamma_label.setWordWrap(True)

        button_close_patient = QPushButton(self)
        button_close_patient.setText("Close patient")
        button_close_patient.clicked.connect(self.close_patient)

        button_browse_slices = QPushButton(self)
        button_browse_slices.setText("Browse slices")
        button_browse_slices.clicked.connect(self.browse_slices)

        button_qa_stats = QPushButton(self)
        button_qa_stats.setText("QA statistics")
        button_qa_stats.clicked.connect(self.show_qa_statistics)

        self.roi_combobox = QComboBox(self)
        self.roi_combobox.addItem("Select ROI")
        self.roi_combobox.currentIndexChanged.connect(self.update_roi_selection)

        self.isodose_spinbox = QDoubleSpinBox(self)
        self.isodose_spinbox.setRange(0.0, 100.0)
        self.isodose_spinbox.setPrefix("Isodose: ")
        self.isodose_spinbox.setSuffix(" Gy")
        self.isodose_spinbox.setValue(60.0)
        self.isodose_spinbox.valueChanged.connect(self.update_isodose_selection)

        self.dvh_plot_widget = pg.plot(title="DVH")
        self.dvh_plot_widget.setLabel('left', 'Volume (relative to volume of ROI)')
        self.dvh_plot_widget.setLabel('bottom', 'Dose [Gy]')
        self.sel_isodose_line = pg.InfiniteLine(self.isodose_spinbox.value(),
                                                pen=pg.mkPen(color=(200, 200, 255), style=QtCore.Qt.DotLine))
        self.dvh_plot_widget.addItem(self.sel_isodose_line)

        self.dd_plot_widget = pg.plot(title="Dose difference (measured - planned)")
        self.dd_plot_widget.setLabel('left', 'Fraction of region volume')
        self.dd_plot_widget.setLabel('bottom', 'Dose difference [Gy]')

        self.roi_stat_table = QTableWidget(self)
        self.roi_stat_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        roi_stat_column_headers = ["ROI", "GPR [%]", "ΔD 95%", "ΔD 50%", "ΔD 5%", "mean D ref", "std D ref",
                                   "min D ref", "max D ref", "mean D cmp", "std D cmp", "min D cmp", "max D cmp"]
        self.roi_stat_table.setColumnCount(len(roi_stat_column_headers))
        self.roi_stat_table.setHorizontalHeaderLabels(roi_stat_column_headers)
        self.roi_stat_table.setMinimumHeight(220)
        self.roi_stat_table.setMinimumWidth(700)
        self.roi_stat_table.setMaximumHeight(370)

        self.confusion_matrix_table = QTableWidget(self)
        self.confusion_matrix_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.confusion_matrix_table.setColumnCount(self.configs['confusion_matrix_N'])
        self.confusion_matrix_table.setRowCount(self.configs['confusion_matrix_N'])
        self.confusion_matrix_table.setMinimumHeight(220)
        self.confusion_matrix_table.setMaximumHeight(370)

        self.update_roi_stat_table()

        outer_layout = QVBoxLayout()
        grid_layout = QGridLayout()
        plot_layout = QHBoxLayout()
        mid_layout = QHBoxLayout()
        bottom_layout = QHBoxLayout()
        # Add widgets to the layout
        grid_layout.addWidget(button_open_ct, 0, 0)
        grid_layout.addWidget(self.ct_label, 1, 0)
        grid_layout.addWidget(button_open_rtstruct, 0, 1)
        grid_layout.addWidget(self.rtstruct_label, 1, 1)
        grid_layout.addWidget(button_open_planned, 0, 2)
        grid_layout.addWidget(self.planned_dose_label, 1, 2)
        grid_layout.addWidget(button_open_measured, 0, 3)
        grid_layout.addWidget(self.measured_dose_label, 1, 3)
        grid_layout.addWidget(button_update_gamma, 0, 4)
        grid_layout.addWidget(button_close_patient, 0, 5)
        grid_layout.addWidget(self.gamma_label, 1, 4)
        grid_layout.addWidget(button_browse_slices, 0, 6)
        grid_layout.addWidget(button_qa_stats, 0, 7)

        mid_layout.addWidget(self.roi_combobox)
        mid_layout.addWidget(self.isodose_spinbox)

        plot_layout.addWidget(self.dvh_plot_widget)
        plot_layout.addWidget(self.dd_plot_widget)

        bottom_layout.addWidget(self.roi_stat_table, 55)
        bottom_layout.addWidget(VerticalLabel("Measured dose in rows, expected in columns [Gy]"), 1)
        bottom_layout.addWidget(VerticalLabel("Upper: confusion matrix (global normalization"), 1)
        bottom_layout.addWidget(VerticalLabel("Lower: cumulative dose error [Gy*cm^3]"), 1)
        bottom_layout.addWidget(self.confusion_matrix_table, 40)
        # Set the layout on the application's window
        outer_layout.addLayout(grid_layout)
        outer_layout.addLayout(mid_layout)
        outer_layout.addLayout(plot_layout)
        outer_layout.addLayout(bottom_layout)
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
            rename_CTs_to_SOPInstanceUID(fname + "/")
            cache_key = f"CT file {fname}"
            self.ct = self.cache.get(cache_key, None)
            self.dicom_ct_dir = fname
            if self.ct is None:
                self.ct = read_dicom_3d(fname + "/")
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

            self.update_roi_stat_table()
        except Exception as e:
            print(e)
            print(traceback.format_exc())
            self.display_warning(f"Could not load {fname}:\n{e}")
        finally:
            self.status_label.setText("Ready")

    def update_planned_dose(self, fname):
        try:
            cache_key = f"dose file {fname}"
            self.planned_dose_ct = self.cache.get(cache_key, None)
            if self.planned_dose_ct is None:
                planned_dose = read_tps_dose(fname)
                self.planned_dose_ct = resample_doses_to_ct(planned_dose, self.ct)
                self.cache.add(cache_key, self.planned_dose_ct)
            self.gamma_full = None
            self.update_roi_stat_table()
            self.gamma_label.setText("Gamma not updated")
            self.planned_dose_label.setText("Loaded planned dose\n" + fname)
        except Exception as e:
            print(e)
            print(traceback.format_exc())
            self.display_warning(f"Could not load {fname}:\n{e}")

    def update_measured_dose(self, fname):
        try:
            cache_key = f"dose file {fname}"
            self.measured_dose_ct = self.cache.get(cache_key, None)
            if self.measured_dose_ct is None:
                measured_dose = read_tps_dose(fname)
                self.measured_dose_ct = resample_doses_to_ct(measured_dose, self.ct)
                self.cache.add(cache_key, self.measured_dose_ct)
            self.gamma_full = None
            self.update_roi_stat_table()
            self.gamma_label.setText("Gamma not updated")
            self.measured_dose_label.setText("Loaded measured dose:\n" + fname)
        except Exception as e:
            print(e)
            print(traceback.format_exc())
            self.display_warning(f"Could not load {fname}:\n{e}")

    def get_voxel_volume_cm3(self):
        lens = tuple((self.ct[0][i][2] - self.ct[0][i][1] for i in (0, 1, 2)))
        vol = abs(lens[0] * lens[1] * lens[2] / 1000)
        return vol

    def update_confusion_matrix(self):
        selected_roi_id = self.roi_combobox.currentData()

        N = int(self.configs['confusion_matrix_N'])
        if selected_roi_id is None:
            for cm_row in range(N):
                for cm_col in range(N):
                    self.confusion_matrix_table.setItem(cm_row, cm_col, QTableWidgetItem(""))
                    self.confusion_matrix_table.item(cm_row, cm_col).setBackground(QColor(255, 255, 255))
        else:
            mask = self.contours[selected_roi_id]
            lower_q = np.quantile(self.planned_dose_ct[mask != 0], 1 / N)
            upper_q = np.quantile(self.planned_dose_ct[mask != 0], 1 - 1 / N)
            dose_levels = np.linspace(lower_q, upper_q, N - 1)
            self.confusion_matrix_table.setColumnCount(N)
            self.confusion_matrix_table.setRowCount(N)
            cm_vals, dd_vals = prepare_confusion_matrix(self.planned_dose_ct, self.measured_dose_ct, mask, dose_levels)
            labels = [f"(0, {dose_levels[0]:.1f})"]
            labels.extend([f"[{dose_levels[i]:.1f}, {dose_levels[i + 1]:.1f})" for i in range(0, N - 2)])
            labels.append(f"[{dose_levels[-1]:.1f}, ∞)")
            self.confusion_matrix_table.setHorizontalHeaderLabels(labels)
            self.confusion_matrix_table.setVerticalHeaderLabels(labels)
            vol = self.get_voxel_volume_cm3()
            max_dd_color = 10
            for cm_row in range(N):
                for cm_col in range(N):
                    cm_val = cm_vals[cm_row, cm_col]
                    dd_val = dd_vals[cm_row, cm_col] * vol
                    self.confusion_matrix_table.setItem(cm_row, cm_col, QTableWidgetItem(f"{cm_val:.4f}\n{dd_val:.2f}"))
                    if dd_val >= 0:
                        val = 255 - min(255, int(255 * (dd_val / max_dd_color)))
                        color = QColor(255, val, val)
                    else:
                        val = 255 - min(255, int(255 * (-dd_val / max_dd_color)))
                        color = QColor(val, val, 255)

                    self.confusion_matrix_table.item(cm_row, cm_col).setBackground(color)

        self.confusion_matrix_table.setWordWrap(True)
        self.confusion_matrix_table.resizeColumnsToContents()
        self.confusion_matrix_table.resizeRowsToContents()

    def update_roi_selection(self):
        self.update_dvh_plot()
        self.recalculate_plots_for_isodose()
        self.update_confusion_matrix()

    def update_isodose_selection(self):
        self.recalculate_plots_for_isodose()

    def update_roi_stat_table(self):
        self.roi_stat_table.clearSpans()
        if self.rois is not None:
            self.roi_stat_table.setRowCount(len(self.rois))
            for roi_row, roi in enumerate(self.rois):
                self.roi_stat_table.setItem(roi_row, 0, QTableWidgetItem(roi[1]))

                if self.planned_dose_ct is not None and self.contours is not None:
                    doses_ref = self.planned_dose_ct[self.contours[roi[0]] != 0]
                    mean_dose = np.mean(doses_ref)
                    self.roi_stat_table.setItem(roi_row, 5, QTableWidgetItem(f"{mean_dose:.2f}"))
                    std_dose = np.std(doses_ref)
                    self.roi_stat_table.setItem(roi_row, 6, QTableWidgetItem(f"{std_dose:.2f}"))
                    min_dose = np.min(doses_ref)
                    self.roi_stat_table.setItem(roi_row, 7, QTableWidgetItem(f"{min_dose:.2f}"))
                    max_dose = np.max(doses_ref)
                    self.roi_stat_table.setItem(roi_row, 8, QTableWidgetItem(f"{max_dose:.2f}"))

                if self.measured_dose_ct is not None and self.contours is not None:
                    doses_measured = self.measured_dose_ct[self.contours[roi[0]] != 0]
                    mean_dose = np.mean(doses_measured)
                    self.roi_stat_table.setItem(roi_row, 9, QTableWidgetItem(f"{mean_dose:.2f}"))
                    std_dose = np.std(doses_measured)
                    self.roi_stat_table.setItem(roi_row, 10, QTableWidgetItem(f"{std_dose:.2f}"))
                    min_dose = np.min(doses_measured)
                    self.roi_stat_table.setItem(roi_row, 11, QTableWidgetItem(f"{min_dose:.2f}"))
                    max_dose = np.max(doses_measured)
                    self.roi_stat_table.setItem(roi_row, 12, QTableWidgetItem(f"{max_dose:.2f}"))

                if self.planned_dose_ct is not None and self.measured_dose_ct is not None and self.contours is not None:
                    planned_mod_dose_hist, planned_mod_vol_hist = self.calc_dvh_for_roi(self.planned_dose_ct, roi[0])
                    measured_mod_dose_hist, measured_mod_vol_hist = self.calc_dvh_for_roi(self.measured_dose_ct, roi[0])
                    for qid, quant in enumerate([95, 50, 5]):
                        pq = calculate_Dx(planned_mod_dose_hist, planned_mod_vol_hist, quant)
                        mq = calculate_Dx(measured_mod_dose_hist, measured_mod_vol_hist, quant)
                        self.roi_stat_table.setItem(roi_row, qid + 2, QTableWidgetItem(f"{(mq - pq):.2f}"))
                if self.gamma_full is not None and self.contours is not None:
                    gpr = 100 * calculate_gpr_for_roi(self.gamma_full, self.contours[roi[0]])
                    self.roi_stat_table.setItem(roi_row, 1, QTableWidgetItem(f"{gpr:.1f}"))

        self.roi_stat_table.resizeColumnsToContents()
        self.roi_stat_table.resizeRowsToContents()

    def browse_slices(self):
        self.view_napari(self.ct[1], dose_measured=self.measured_dose_ct, dose_planned=self.planned_dose_ct)

    def show_qa_statistics(self):
        qa_stat_window = QAStatWindow(self)
        qa_stat_window.show()

    def open_file_ct_clicked(self):
        fname = QFileDialog.getExistingDirectory(self, 'Open CT folder')
        if fname != "":
            self.update_ct(fname)

    def open_file_rtstruct_clicked(self):
        fname = QFileDialog.getOpenFileName(self, 'Open RTStruct file', filter="RT files (*.dcm);;All files (*.*)")
        if fname[0] != "":
            self.update_rtstruct(fname[0])

    def open_file_measured_clicked(self):
        fname = QFileDialog.getOpenFileName(self, 'Open measured dose file', filter="RT files (*.dcm);;All files (*.*)")
        if fname[0] != "":
            self.update_measured_dose(fname[0])

    def open_file_planned_clicked(self):
        fname = QFileDialog.getOpenFileName(self, 'Open planned dose file', filter="RT files (*.dcm);;All files (*.*)")
        if fname[0] != "":
            self.update_planned_dose(fname[0])

    def view_napari(self, ct, dose_measured=None, dose_planned=None):
        ct_scale = tuple((self.ct[0][i][2] - self.ct[0][i][1] for i in (0, 1, 2)))
        viewer = napari.view_image(ct, name="CT", scale=ct_scale)

        green = vispy.color.Colormap([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        red = vispy.color.Colormap([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        blue = vispy.color.Colormap([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        neg_blue = vispy.color.Colormap([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
        divergent = vispy.color.Colormap([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], [0.0, 0.5, 1.0])

        if dose_measured is not None:
            measured_layer = viewer.add_image(dose_measured, name="Measured dose", opacity=0.4, scale=ct_scale,
                                              visible=False)
            measured_layer.colormap = 'red', red
        if dose_planned is not None:
            planned_layer = viewer.add_image(dose_planned, name="Planned dose", opacity=0.4, scale=ct_scale,
                                             visible=False)
            planned_layer.colormap = 'blue', blue

        selected_roi_id = self.roi_combobox.currentData()
        if selected_roi_id is not None:
            isodose_Gy = self.isodose_spinbox.value()
            roi_mask_layer = viewer.add_image(self.contours[selected_roi_id].astype(np.float32),
                                              name=f"ROI {self.roi_combobox.currentText()} Gy",
                                              opacity=0.8, scale=ct_scale, blending='additive', visible=False)
            roi_mask_layer.colormap = 'green', green
            if self.planned_dose_ct is not None:
                img_hot_cold = prepare_hot_cold_image(self.planned_dose_ct, self.measured_dose_ct,
                                                      self.contours[selected_roi_id], isodose_Gy)

                hot_cold_layer = viewer.add_image(img_hot_cold, name=f"Hot and cold areas for isodose {isodose_Gy} Gy",
                                                  opacity=0.9, rgb=True, scale=ct_scale, blending='additive')

                img_hot_val = prepare_hot_val(self.planned_dose_ct, self.measured_dose_ct,
                                              self.contours[selected_roi_id], isodose_Gy)
                img_cold_val = prepare_cold_val(self.planned_dose_ct, self.measured_dose_ct,
                                                self.contours[selected_roi_id], isodose_Gy)
                cold_val_layer = viewer.add_image(img_cold_val,
                                                  name=f"Cold areas for isodose {isodose_Gy} Gy",
                                                  opacity=0.8, scale=ct_scale, blending='additive')
                hot_val_layer = viewer.add_image(img_hot_val,
                                                 name=f"Hot areas for isodose {isodose_Gy} Gy",
                                                 opacity=0.8, scale=ct_scale, blending='additive')
                hot_val_layer.colormap = 'red'
                cold_val_layer.colormap = 'neg_blue', neg_blue

        if self.gamma_full is not None:
            gamma_layer = viewer.add_image((self.gamma_full > 1.0).astype(np.float32),
                                           name=f"gamma index failures (gamma > 1)",
                                           opacity=0.8, scale=ct_scale, blending='additive', visible=False)
            gamma_layer.colormap = 'red'

        viewer.scale_bar.visible = True
        viewer.scale_bar.unit = "mm"
        viewer.dims.order = (2, 0, 1)
        return viewer

    def close_patient(self):
        self.ct = None
        self.measured_dose_ct = None
        self.planned_dose_ct = None
        self.struct_dcm = None
        self.rois = None
        self.contours = None
        self.gamma_full = None
        self.ct_label.setText("Loaded CT: None")
        self.rtstruct_label.setText("Loaded RTStruct: None")
        self.measured_dose_label.setText("Loaded measured dose: None")
        self.planned_dose_label.setText("Loaded planned dose: None")
        self.gamma_label.setText("Gamma not updated")
        self.roi_combobox.clear()
        self.roi_stat_table.clearContents()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    global ex
    ex = App()
    sys.exit(app.exec_())
    # ct, dose_measured, dose_planned = load_data()
