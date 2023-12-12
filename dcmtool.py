import nibabel
import napari
import pydicom

import traceback

import sys
import json
from common.dicom_tools import read_dicom_3d, read_tps_dose, draw_contours, display_structure_ids_ds, \
    rename_CTs_to_SOPInstanceUID, resample_doses_to_ct, ct_transform_matrix

import numpy as np
import os


def get_seqs(dicom_rs, roid_id, path_to_ct):
    roi_numbers = [dum.ROINumber for dum in dicom_rs.StructureSetROISequence]
    struct_id = roi_numbers.index(roid_id)

    # print(ds.StructureSetROISequence[struct_id].ROINumber,ds.StructureSetROISequence[struct_id].ROIName)

    # assert ds.SOPClassUID == StructureDICOM_SOPClassUID, "This is not a Dicom Structure file"

    # ROINumber and ROIName are defined in ds.StructureSetROISequence
    # ROINumber is referenced to in ds.ROIContourSequence
    # Each ROI in ds.StructureSetROISequence correpsponds to a contour sequence in ds.ROIContourSequence
    # They must be matched based on ROINumber, referenced to in ds.ROIContourSequence and defined in ds.StructureSetROISequence
    # There is a single contour sequence corresponding to a ROI specified by struct_id - I extract this contour sequence

    ROI = [dicom_rs.ROIContourSequence[u].ContourSequence for u in range(len(dicom_rs.ROIContourSequence)) if
           dicom_rs.ROIContourSequence[u].ReferencedROINumber
           == dicom_rs.StructureSetROISequence[struct_id].ROINumber][0]

    point_seqs = []
    for seq in ROI:
        dic_image = pydicom.dcmread(path_to_ct + '/CT.' + seq.ContourImageSequence[0].ReferencedSOPInstanceUID + '.dcm')

        M = ct_transform_matrix(dic_image)
        points = np.swapaxes(np.reshape(seq.ContourData, (-1, 3)), 0, 1)
        points[2, :].fill(1)
        points = np.dot(M, points)
        zpos = -dic_image[0x0020, 0x0032].value[2]
        points[2, :] = zpos
        point_seqs.append(points)

    return np.concatenate(point_seqs, axis=1)


def export_shapes():
    roi_data = {}
    for p in [1, 2]:
        for f in range(26):
            print(f"Processing patient {p}, fraction {f}")
            dir_ct = f"/home/mateusz/Desktop/tmp/sco/231121_sco_anonymized/Pacjent_{p:02d}_anonymized/frakcja_{f:02d}/CT/"
            dir = f"/home/mateusz/Desktop/tmp/sco/231121_sco_anonymized/Pacjent_{p:02d}_anonymized/frakcja_{f:02d}/RTStruct/"
            try:
                rs_fname = os.listdir(dir)[0]
                struct_dcm = pydicom.dcmread(dir + rs_fname)
                rois = display_structure_ids_ds(struct_dcm)
                for roi in rois:
                    if roi[1] not in roi_data.keys():
                        roi_data[roi[1]] = {}
                    if p not in roi_data[roi[1]].keys():
                        roi_data[roi[1]][p] = {}

                    print(f"Adding data for {roi}, {p}, {f}")
                    roi_data[roi[1]][p][f] = get_seqs(struct_dcm, roi[0], dir_ct)

            except Exception as e:
                print(e)
    return roi_data


roi_data = export_shapes()


def save_shapes(roi_data):
    for roi_name, patients in roi_data.items():
        dir = f"/home/mateusz/Desktop/tmp/sco/export/{roi_name}/"
        if not os.path.isdir(dir):
            os.mkdir(dir)
        for pi, pdata in patients.items():
            for fi, fdata in pdata.items():
                fname = f"{dir}p_{pi:02d}_f{fi:02d}.csv"
                try:
                    np.savetxt(fname, fdata, delimiter=";")
                except Exception as e:
                    print(e)


save_shapes(roi_data)
