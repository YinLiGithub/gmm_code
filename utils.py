"""
Various helper functions.

Note:
    The functions are in alphabetical order.
"""
import numpy as np


def cov2cov(matrices: np.ndarray):
    """
    Convert the real representations of complex covariance matrices back to
    complex representations.
    """
    if matrices.ndim == 2:
        # the case of diagonal matrices
        n_mats, n_diag = matrices.shape
        mats = np.zeros([n_mats, n_diag, n_diag])
        for i in range(n_mats):
            mats[i, :, :] = np.diag(matrices[i, :])
    else:
        mats = matrices

    n_mats, rows, columns = mats.shape
    row_half = rows // 2
    column_half = columns // 2
    covs = np.zeros((n_mats, row_half, column_half), dtype=np.complex)
    for c in range(n_mats):
        upper_left_block = mats[c, :row_half, :column_half]
        upper_right_block = mats[c, :row_half, column_half:]
        lower_left_block = mats[c, row_half:, :column_half]
        lower_right_block = mats[c, row_half:, column_half:]
        covs[c, :, :] = upper_left_block + lower_right_block + 1j * (lower_left_block - upper_right_block)
    return covs


def cplx2real(vec: np.ndarray, axis=0):
    """
    Concatenate real and imaginary parts of vec along axis=axis.
    """
    return np.concatenate([vec.real, vec.imag], axis=axis)


def crandn(*arg, rng=np.random.default_rng()):
    return np.sqrt(0.5) * (rng.standard_normal(arg) + 1j * rng.standard_normal(arg))


def mat2bsc(mat: np.ndarray):
    """
    Arrange the real and imaginary parts of a complex matrix mat in block-
    skew-circulant form.

    Source:
        See https://ieeexplore.ieee.org/document/7018089.
    """
    upper_half = np.concatenate((mat.real, -mat.imag), axis=-1)
    lower_half = np.concatenate((mat.imag, mat.real), axis=-1)
    return np.concatenate((upper_half, lower_half), axis=-2)


def real2real(mats):
    re = np.real(mats)
    im = np.imag(mats)
    rows = mats.shape[1]
    cols = mats.shape[2]
    out = np.zeros([mats.shape[0], 2*rows, 2*cols])
    out[:, :rows, :cols] = 0.5*re
    out[:, rows:, cols:] = 0.5*re
    return out


def imag2imag(mats):
    im = np.real(mats)
    rows = mats.shape[1]
    cols = mats.shape[2]
    out = np.zeros([mats.shape[0], 2*rows, 2*cols])
    #out[:, :rows, :cols] = 0.5*re
    for i in range(mats.shape[0]):
        out[i, :rows, cols:] = 0.5*im[i,:,:].T
        out[i, rows:, :cols] = 0.5*im[i,:,:]
    #out[:, rows:, cols:] = 0.5*re
    return out


def nmse(actual: np.ndarray, desired: np.ndarray):
    """
    Mean squared error between actual and desired divided by the total number
    of elements.
    """
    return np.sum(np.abs(actual - desired) ** 2) / desired.size


def real2cplx(vec: np.ndarray, axis=0):
    """
    Assume vec consists of concatenated real and imaginary parts. Return the
    corresponding complex vector. Split along axis=axis.
    """
    re, im = np.split(vec, 2, axis=axis)
    return re + 1j * im


def sec2hours(seconds: float):
    """"
    Convert a number of seconds to a string h:mm:ss.
    """
    # hours
    h = seconds // 3600
    # remaining seconds
    r = seconds % 3600
    return '{:.0f}:{:02.0f}:{:02.0f}'.format(h, r // 60, r % 60)


def kron_real(mats1: np.ndarray, mats2: np.ndarray):
    """
    Assuming mats1 and mats2 are real representations of complex covariance
    matrices, compute the real representation of the Kronecker product of the
    complex covariance matrices.
    """
    if mats1.ndim != mats2.ndim:
        raise ValueError(
            'The two arrays need to have the same number of dimensions, '
            f'but we have mats1.ndim = {mats1.ndim} and mats2.ndim = {mats2.ndim}.'
        )
    if mats1.ndim == 2:
        mats1 = np.expand_dims(mats1, 0)
        mats2 = np.expand_dims(mats2, 0)

    n = mats1.shape[0]
    rows1, cols1 = mats1.shape[-2:]
    rows2, cols2 = mats2.shape[-2:]
    row_half1 = rows1 // 2
    column_half1 = cols1 // 2
    row_half2 = rows2 // 2
    column_half2 = cols2 // 2
    rows3 = 2 * row_half1 * row_half2
    cols3 = 2 * column_half1 * column_half2

    out_kron_prod = np.zeros((n, rows3, cols3))
    for i in range(n):
        A1 = mats1[i, :row_half1, :column_half1]
        B1 = mats1[i, :row_half1, column_half1:]
        C1 = mats1[i, row_half1:, :column_half1]
        D1 = mats1[i, row_half1:, column_half1:]

        A2 = mats2[i, :row_half2, :column_half2]
        B2 = mats2[i, :row_half2, column_half2:]
        C2 = mats2[i, row_half2:, :column_half2]
        D2 = mats2[i, row_half2:, column_half2:]

        A = np.kron(A1 + D1, A2 + D2)
        D = -np.kron(C1 - B1, C2 - B2)
        B = -np.kron(A1 + D1, C2 - B2)
        C = np.kron(C1 - B1, A2 + D2)

        A = 0.5 * (A + D)
        D = A
        B = 0.5 * (B - C)
        C = -B

        out_kron_prod[i, :, :] = np.concatenate(
            (np.concatenate((A, B), axis=1), np.concatenate((C, D), axis=1)),
            axis=0
        )
    return np.squeeze(out_kron_prod)
