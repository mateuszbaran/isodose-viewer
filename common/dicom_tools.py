import pydicom
import numpy as np
import os
import cv2
import pandas as pd
import scipy


def read_tps_dose(filename):
    ds = pydicom.dcmread(filename)

    assert ds.Modality == 'RTDOSE', 'This is not a Dicom DOSE file!'

    dose = ds.pixel_array
    scaling = float(ds.DoseGridScaling)

    dose = np.swapaxes(dose, 0, 1)
    dose = np.swapaxes(dose, 1, 2)
    dose = np.asarray(dose, dtype=np.float64) * scaling

    print(dose.shape)

    doseOrigin = ds.ImagePositionPatient
    dosePixelSize = ds.PixelSpacing
    doseSlicePositions = ds[0x3004, 0x000c].value

    grid = (np.arange(doseOrigin[1], doseOrigin[1] + dosePixelSize[1] * dose.shape[0], dosePixelSize[1]),
            np.arange(doseOrigin[0], doseOrigin[0] + dosePixelSize[0] * dose.shape[1], dosePixelSize[0]),
            np.asarray([doseOrigin[2] + x for x in doseSlicePositions], dtype=np.float64))

    return grid, dose


def read_tps_doses_dir(dicom_dir):
    dicom_files = [file for file in os.listdir(dicom_dir) if file.endswith('dcm')]
    dicoms = []
    for file in dicom_files:
        ds = pydicom.dcmread(dicom_dir + file)
        if ds.Modality != "RTDOSE":
            continue
        dicoms.append(dicom_dir + file)

    grids_doses = [read_tps_dose(dcm) for dcm in dicoms]
    # TODO check grids?
    summed_dose = sum([gd[1] for gd in grids_doses])
    return grids_doses[0][0], summed_dose


def read_dicom_3d(dicom_dir):
    dicom_files = [file for file in os.listdir(dicom_dir) if file.endswith('dcm')]
    dicoms = []
    for file in dicom_files:
        ds = pydicom.dcmread(dicom_dir + file)
        if ds.Modality != "CT":
            continue
        dicoms.append((dicom_dir + file, int(ds.InstanceNumber)))
    dicoms = sorted(dicoms, key=lambda x: x[1])

    im3D = []
    slice_positions = []
    for d in dicoms:
        ds = pydicom.dcmread(d[0])
        slice_positions.append(ds.ImagePositionPatient[2])
        # print("Rescale type: ", ds.RescaleType)
        # print(ds.RescaleSlope, ds.RescaleIntercept)
        im3D.append(ds.RescaleSlope * ds.pixel_array + ds.RescaleIntercept)

    im3D = np.asarray(im3D, dtype=np.int16)
    im3D = np.swapaxes(im3D, 0, 1)
    im3D = np.swapaxes(im3D, 1, 2)

    ds = pydicom.dcmread(dicoms[0][0])
    # https://dicom.innolitics.com/ciods/ct-image/image-plane/00200032
    # ImagePositionPatient specifies coordinates of the upper left hand corner of the image: it is the center of the first voxel transmitted
    CTOrigin = ds.ImagePositionPatient
    CTPixelSize = ds.PixelSpacing
    CTSliceThickness = ds.SliceThickness

    grid = (np.arange(CTOrigin[1], CTOrigin[1] + CTPixelSize[1] * im3D.shape[0], CTPixelSize[1]),
            np.arange(CTOrigin[0], CTOrigin[0] + CTPixelSize[0] * im3D.shape[1], CTPixelSize[0]),
            np.asarray(slice_positions, dtype=np.float64))

    return grid, im3D


def display_structure_ids(structures_file):
    ds = pydicom.dcmread(structures_file)
    return display_structure_ids_ds(ds)


def display_structure_ids_ds(ds):
    roi_numbers = []
    for nstruct, struct in enumerate(ds.StructureSetROISequence):
        roi_numbers.append((struct.ROINumber, struct.ROIName))
    return roi_numbers


def ct_transform_matrix(dic_image):
    M = np.zeros((3, 3), dtype=np.float32)
    iop = dic_image[0x0020, 0x0037]
    ipp = dic_image[0x0020, 0x0032]
    pxs = dic_image[0x0028, 0x0030]
    M[0, 0] = iop.value[1] * pxs.value[0]
    M[1, 0] = iop.value[0] * pxs.value[0]
    M[0, 1] = iop.value[4] * pxs.value[1]
    M[1, 1] = iop.value[3] * pxs.value[1]
    M[0, 2] = ipp.value[0]
    M[1, 2] = ipp.value[1]
    M[2, 2] = 1.0
    M = np.linalg.inv(M)
    return M

# Creates volume corresponding to a structure with struct_id
# Returns 3D image with black pixels correcsponding to background and pixels labeled with ds.StructureSetROISequence[struct_id].ROINumber
# corresponding to the structure of interest
def draw_contours(im_size, roid_id, path_to_ct, dicom_rs):
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

    dum = np.zeros(im_size, dtype=np.uint8)

    for seq in ROI:
        dic_image = pydicom.dcmread(path_to_ct + '/CT.' + seq.ContourImageSequence[0].ReferencedSOPInstanceUID + '.dcm')

        M = ct_transform_matrix(dic_image)
        points = np.swapaxes(np.reshape(seq.ContourData, (-1, 3)), 0, 1)
        points[2, :].fill(1)
        points = np.dot(M, points)[:2, :]

        big = int(dicom_rs.StructureSetROISequence[struct_id].ROINumber)  # 255
        CTSlice = int(dic_image[0x0020, 0x0013].value) - 1  # numery sliców w Dicom startują od 1
        dum2D = np.zeros(im_size[0:2], dtype=np.uint8)
        for id in range(points.shape[1] - 1):
            cv2.line(dum2D, (int(points[1, id]), int(points[0, id])), (int(points[1, id + 1]), int(points[0, id + 1])),
                     big, 1)
        cv2.line(dum2D, (int(points[1, points.shape[1] - 1]), int(points[0, points.shape[1] - 1])),
                 (int(points[1, 0]), int(points[0, 0])), big, 1)

        im_flood_fill = dum2D.copy()
        h, w = dum2D.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)
        im_flood_fill = im_flood_fill.astype("uint8")
        cv2.floodFill(im_flood_fill, mask, (0, 0), 128)
        dum2D[im_flood_fill != 128] = big
        np.copyto(dum[:, :, CTSlice], dum2D)

    return dum


def rename_CTs_to_SOPInstanceUID(dicom_dir):
    # draw_contours expects CT files to be named according to SOPInstanceUID. Sometimes that's not the case
    # and then the easiest solution is to rename the files
    dicom_files = [file for file in os.listdir(dicom_dir) if file.endswith('dcm')]
    for file in dicom_files:
        ds = pydicom.dcmread(dicom_dir + file)
        if ds.Modality != "CT":
            continue
        print("Rename", file, " to ", ds.SOPInstanceUID)
        os.rename(dicom_dir + file, dicom_dir + "CT." + ds.SOPInstanceUID + ".dcm")


"""
    Data based on http://sbcrowe.net/ct-density-tables/
"""
electron_density_table = pd.DataFrame(
    data={"Material": ["Air",
                       "Lung 300",
                       "Lung 450",
                       "Adipose",
                       "Breast",
                       "Solid Water",
                       "Water",
                       "Brain",
                       "Liver",
                       "Inner Bone",
                       "B-200",
                       "CB2 30%",
                       "CB2 50%",
                       "Cortical Bone",
                       ], "Mass density": [0,
                                           0.29,
                                           0.45,
                                           0.943,
                                           0.985,
                                           1.016,
                                           1,
                                           1.052,
                                           1.089,
                                           1.145,
                                           1.159,
                                           1.335,
                                           1.56,
                                           1.823,
                                           ], "Relative electron density": [0,
                                                                            0.278,
                                                                            0.443,
                                                                            0.926,
                                                                            0.962,
                                                                            0.987,
                                                                            1,
                                                                            1.048,
                                                                            1.058,
                                                                            1.098,
                                                                            1.111,
                                                                            1.28,
                                                                            1.47,
                                                                            1.695,
                                                                            ], "GE LightSpeed": [-991.5,
                                                                                                 -729.2,
                                                                                                 -541.8,
                                                                                                 -92.8,
                                                                                                 -33,
                                                                                                 3.7,
                                                                                                 -2.9,
                                                                                                 28.7,
                                                                                                 64.9,
                                                                                                 212.7,
                                                                                                 227.4,
                                                                                                 442,
                                                                                                 791.2,
                                                                                                 1191.8,
                                                                                                 ], "Siemens": [-969.8,
                                                                                                                -712.9,
                                                                                                                -536.5,
                                                                                                                -95.6,
                                                                                                                -45.6,
                                                                                                                -1.9,
                                                                                                                -5.6,
                                                                                                                25.7,
                                                                                                                65.6,
                                                                                                                207.5,
                                                                                                                220.7,
                                                                                                                429.9,
                                                                                                                775.3,
                                                                                                                1173.7,
                                                                                                                ],
          "Toshiba": [-970.3,
                      -720.8,
                      -543.3,
                      -67.2,
                      -36.4,
                      -5.5,
                      -5.1,
                      16.8,
                      65.8,
                      211,
                      229.6,
                      464.4,
                      831.1,
                      1256.2,
                      ]})


def hu_to_electron_density_converter(ct_machine="Siemens"):
    """
    Return a function that converts HU to electron density relative to water.
    `ct_machine` can be either "GE LightSpeed", "Siemens" or "Toshiba"

    """
    eds = electron_density_table["Relative electron density"]
    HUs = electron_density_table[ct_machine]
    return scipy.interpolate.interp1d(np.array(HUs), np.array(eds), fill_value=(0.0, np.max(HUs)), bounds_error=False)


def hu_to_mass_density_converter(ct_machine="Siemens"):
    """
    Return a function that converts HU to mass density relative to water.
    `ct_machine` can be either "GE LightSpeed", "Siemens" or "Toshiba"

    """
    eds = electron_density_table["Mass density"]
    densities = electron_density_table[ct_machine]
    return scipy.interpolate.interp1d(np.array(densities), np.array(eds), fill_value=(0.0, np.max(densities)),
                                      bounds_error=False)


def resample_doses_to_ct(doses, ct):
    grid_ct = ct[0]
    grid_doses = doses[0]
    start = []
    end = []
    start.append(np.where(grid_ct[0] > np.min(grid_doses[0]))[0][0])
    end.append(np.where(grid_ct[0] < np.max(grid_doses[0]))[0][-1])
    # print(start[0], end[0])
    start.append(np.where(grid_ct[1] > np.min(grid_doses[1]))[0][0])
    end.append(np.where(grid_ct[1] < np.max(grid_doses[1]))[0][-1])
    # print(start[1], end[1])

    if (grid_doses[2][1] - grid_doses[2][0]) * (grid_ct[2][1] - grid_ct[2][0]) > 0:
        start.append(np.where(grid_ct[2] > np.min(grid_doses[2]))[0][0])
        end.append(np.where(grid_ct[2] < np.max(grid_doses[2]))[0][-1])
        # print(start[2], end[2])
    else:
        end.append(np.where(grid_ct[2] > np.min(grid_doses[2]))[0][-1])
        start.append(np.where(grid_ct[2] < np.max(grid_doses[2]))[0][0])
        # print(start[2], end[2])

    computation_points = np.asarray(
        [[grid_ct[0][i], grid_ct[1][j], grid_ct[2][k]] for i in range(start[0], end[0]) for j in range(start[1], end[1])
         for k in range(start[2], end[2])])

    results = scipy.interpolate.interpn(grid_doses, doses[1], computation_points)
    results = np.array(results, dtype=np.float32)
    results = results.reshape(end[0] - start[0], end[1] - start[1], end[2] - start[2])

    results2 = np.zeros(ct[1].shape, dtype=np.float32)
    results2[start[0]:end[0], start[1]:end[1], start[2]:end[2]] = np.copy(results)
    return results2
