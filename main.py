import nibabel
import napari

from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QFileDialog, QMessageBox, QGridLayout, QLabel

import sys


def load_data(
        ct_fname='/home/mateusz/Desktop/tmp/isodose-data/CT.nii.gz',
        dose_measured_fname='/home/mateusz/Desktop/tmp/isodose-data/Dose_3DVH.nii.gz',
        dose_planned_fname='/home/mateusz/Desktop/tmp/isodose-data/Dose_TPS.nii.gz'
):
    ct = nibabel.load(ct_fname)
    dose_measured = nibabel.load(dose_measured_fname)
    dose_planned = nibabel.load(dose_planned_fname)
    return ct, dose_measured, dose_planned


def view_napari(ct, dose_measured=None, dose_planned=None):
    viewer = napari.view_image(ct.get_fdata(), name="CT")
    if dose_measured is not None:
        napari.add_image(dose_measured.get_fdata(), name="Measured dose")
    if dose_planned is not None:
        viewer.add_image(dose_planned.get_fdata(), name="Planned dose")

    return viewer


class App(QWidget):
    def __init__(self):
        super().__init__()
        self.ct = None
        self.measured_dose = None
        self.planned_dose = None
        self.ct_label = None
        self.measured_dose_label = None
        self.planned_dose_label = None
        self.title = 'Isodose explorer'
        self.left = 10
        self.top = 10
        self.width = 640
        self.height = 480
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        button_open_ct = QPushButton(self)
        button_open_ct.setText("Open CT file")
        button_open_ct.clicked.connect(self.open_file_ct_clicked)
        self.ct_label = QLabel("no CT loaded", self)

        button_open_measured = QPushButton(self)
        button_open_measured.setText("Open measured dose file")
        button_open_measured.clicked.connect(self.open_file_measured_clicked)
        self.measured_dose_label = QLabel("no measured dose loaded", self)

        button_open_planned = QPushButton(self)
        button_open_planned.setText("Open planned dose file")
        button_open_planned.clicked.connect(self.open_file_planned_clicked)
        self.planned_dose_label = QLabel("no planned dose loaded", self)

        button_browse_slices = QPushButton(self)
        button_browse_slices.setText("Browse slices")
        button_browse_slices.clicked.connect(self.browse_slices)

        layout = QGridLayout()
        # Add widgets to the layout
        layout.addWidget(button_open_ct, 0, 0)
        layout.addWidget(self.ct_label, 1, 0)
        layout.addWidget(button_open_planned, 0, 1)
        layout.addWidget(self.planned_dose_label, 1, 1)
        layout.addWidget(button_open_measured, 0, 2)
        layout.addWidget(self.measured_dose_label, 1, 2)
        layout.addWidget(button_browse_slices, 0, 3)
        # Set the layout on the application's window
        self.setLayout(layout)
        self.show()

    def display_warning(self, msg):
        msgBox = QMessageBox()
        msgBox.setIcon(QMessageBox.Warning)
        msgBox.setText(msg)
        msgBox.setStandardButtons(QMessageBox.Ok)

    def update_ct(self, fname):
        self.ct = nibabel.load(fname)

    def update_planned_dose(self, fname):
        self.planned_dose = nibabel.load(fname)

    def update_measured_dose(self, fname):
        self.measured_dose = nibabel.load(fname)

    def browse_slices(self):
        view_napari(self.ct, self.measured_dose, self.planned_dose)

    def open_file_ct_clicked(self):
        fname = QFileDialog.getOpenFileName(self, 'Open CT file', "RT files (*.nii.gz)")
        self.update_ct(fname[0])

    def open_file_measured_clicked(self):
        fname = QFileDialog.getOpenFileName(self, 'Open measured dose file', "RT files (*.nii.gz)")
        self.update_measured_dose(fname[0])

    def open_file_planned_clicked(self):
        fname = QFileDialog.getOpenFileName(self, 'Open planned dose file', "RT files (*.nii.gz)")
        self.update_planned_dose(fname[0])


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
    # ct, dose_measured, dose_planned = load_data()
