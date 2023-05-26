import numpy as np


def calculate_dvh(dose, _xrange, nbins):
    _hist, x = np.histogram(dose, bins=nbins, range=_xrange, density=True)
    _fhist = _hist[::-1]  # reverse histogram, so first element is for highest dose
    _fhist = np.cumsum(_fhist)
    _hist = _fhist[::-1]  # flip back again to normal representation

    y = 100.0 * _hist / _hist[0]  # volume histograms always plot the right edge of bin, since V(D < x_pos).
    y = np.insert(y, 0, 100.0, axis=0)  # but the leading bin edge is always at V = 100.0%
    return x, y


def make_grid_increasing_in_z(input_grid):
    dif = input_grid[2][0] - input_grid[2][1]
    if dif > 0:
        z_grid = np.arange(0, input_grid[2].shape[0] * dif, dif)
    else:
        z_grid = input_grid[2]

    return (input_grid[0], input_grid[1], z_grid)


def calculate_roi_volume(roi_mask, grid_cm=None):
    # TODO: convert to cm^3?
    number_of_voxels = len(np.where(roi_mask > 0)[0])
    if grid_cm is None:
        return number_of_voxels
    else:
        vol_voxel_mm3 = np.abs((grid_cm[0][1] - grid_cm[0][0])*(grid_cm[1][1] - grid_cm[1][0])*(grid_cm[2][1] - grid_cm[2][0]))
        return number_of_voxels * vol_voxel_mm3 / 1000


def calculate_gpr_for_roi(gamma_img, roi_mask):
    vol = len(np.where(roi_mask > 0)[0])

    gg = gamma_img.copy()
    # NaN means not calculated, for example because the point is below the threshold
    # global_normalization and lower_percent_dose_cutoff determine that
    # note that PRIMO sometimes gives very high doses for some points outside the body which is then taken as
    # the global normalization level and as a result most points don't even have gamma calculated even though
    # they should have
    gg[np.isnan(gamma_img)] = 0
    gg[roi_mask == 0] = 0
    big_gamma = len(np.where(gg > 1)[0])
    return 1 - big_gamma / vol


def read_phsp_txt(fname, **load_params):
    """
    Open fname and read it as a TXT phase space file (six columns, in order: X [cm], Y [cm], dX, dY, dZ, Ekin [MeV]
    :param fname:
    :return:
    """
    return np.loadtxt(fname, dtype=np.float64, **load_params)
    # with open(fname) as file:
    #     lines = file.readlines()
    #     proc_lines = [[float(n) for n in line.split()] for line in lines]
    #     return np.array(proc_lines)


def filter_phsp(input_phsp, limit_cm=5.5, check_y=True, symmetrize=True):
    within_circle = (np.abs(input_phsp[:, 0])) ** 2 + (np.abs(input_phsp[:, 1])) ** 2 < limit_cm ** 2

    if check_y:
        selected = input_phsp[within_circle & (input_phsp[:, 1] < 0), :]
    else:
        selected = input_phsp[within_circle, :]

    if symmetrize:
        signs = np.random.randint(0, 2, selected.shape[0]) * 2 - 1
        selected[:, 0] = selected[:, 0] * signs
        selected[:, 1] = selected[:, 1] * signs
        selected[:, 2] = selected[:, 2] * signs
        selected[:, 3] = selected[:, 3] * signs

    return selected
    

def calculate_Dx(dose_hist, vol_hist, dx):
    """
    Calculate the lowest dose to the most irradiated dx percent of ROI based on DVH data.
    :param dose_hist:
    :param vol_hist:
    :param dx:
    :return:
    """
    return np.interp(dx, np.flip(vol_hist), np.flip(dose_hist))
