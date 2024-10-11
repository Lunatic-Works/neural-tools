import numpy as np
from numba import njit, prange
from skimage.color import rgb2xyz
from skimage.filters import gaussian


def cross_sum(a, out):
    a = np.pad(a, ((1, 1), (1, 1), (0, 0)))
    out[:] = a[2:, 1:-1] + a[:-2, 1:-1] + a[1:-1, 2:] + a[1:-1, :-2]


@njit(nogil=True, parallel=True)
def cross_sum_unroll(a, out, mask):
    for i in prange(1, a.shape[0] - 1):
        for j in range(1, a.shape[1] - 1):
            if mask[i, j]:
                continue
            out[i, j] = 4 * a[i, j]
            out[i, j] += a[i - 1, j]
            out[i, j] += a[i + 1, j]
            out[i, j] += a[i, j - 1]
            out[i, j] += a[i, j + 1]

    for i in range(1, a.shape[0] - 1):
        j = 0
        if not mask[i, j]:
            out[i, j] = 4 * a[i, j]
            out[i, j] += a[i - 1, j]
            out[i, j] += a[i + 1, j]
            out[i, j] += a[i, j + 1]

        j = a.shape[1] - 1
        if not [i, j]:
            out[i, j] = 4 * a[i, j]
            out[i, j] += a[i - 1, j]
            out[i, j] += a[i + 1, j]
            out[i, j] += a[i, j - 1]

    for j in range(1, a.shape[1] - 1):
        i = 0
        if not mask[i, j]:
            out[i, j] = 4 * a[i, j]
            out[i, j] += a[i + 1, j]
            out[i, j] += a[i, j - 1]
            out[i, j] += a[i, j + 1]

        i = a.shape[0] - 1
        if not mask[i, j]:
            out[i, j] = 4 * a[i, j]
            out[i, j] += a[i - 1, j]
            out[i, j] += a[i, j - 1]
            out[i, j] += a[i, j + 1]

    i = 0
    j = 0
    if not mask[i, j]:
        out[i, j] = 4 * a[i, j]
        out[i, j] += a[i + 1, j]
        out[i, j] += a[i, j + 1]

    i = 0
    j = a.shape[1] - 1
    if not mask[i, j]:
        out[i, j] = 4 * a[i, j]
        out[i, j] += a[i + 1, j]
        out[i, j] += a[i, j - 1]

    i = a.shape[0] - 1
    j = 0
    if not mask[i, j]:
        out[i, j] = 4 * a[i, j]
        out[i, j] += a[i - 1, j]
        out[i, j] += a[i, j + 1]

    i = a.shape[0] - 1
    j = a.shape[1] - 1
    if not mask[i, j]:
        out[i, j] = 4 * a[i, j]
        out[i, j] += a[i - 1, j]
        out[i, j] += a[i, j - 1]


def tv_smooth(img, mask, max_iter=10**3, tol=1e-3):
    assert mask.ndim == 2
    mask_0 = mask[:, :, None]
    mask = mask_0.astype(img.dtype)

    img_new = np.zeros_like(img)
    mask_new = np.zeros_like(mask)
    for _ in range(max_iter):
        cross_sum(mask * img, img_new)
        cross_sum(mask, mask_new)
        mask = mask_new
        img_new /= np.maximum(mask, 1e-7).astype(mask.dtype)
        img_new = np.where(mask_0, img, img_new)

        norm = ((img_new - img) ** 2).max()
        print(f"{_} {norm:.3g}")
        if norm < tol:
            break

        img, img_new = img_new, img
        mask = (mask > 0).astype(mask.dtype)

    return img_new


# uwx: (H - 1, W)
# uwy: (H, W - 1)
def compute_texture_weights(img, sigma, sharpness=0.02, sharpness_lf=1e-3):
    img = rgb2xyz(img)
    gx = img[1:, :] - img[:-1, :]
    gy = img[:, 1:] - img[:, :-1]
    wx = 1 / (np.sqrt((gx**2).sum(axis=2)) + sharpness)
    wy = 1 / (np.sqrt((gy**2).sum(axis=2)) + sharpness)
    # print('wx', wx.min(), wx.max(), wx.mean(), wx.std())
    # print('wy', wy.min(), wy.max(), wy.mean(), wy.std())

    gx_lf = gaussian(gx, sigma, channel_axis=2)
    gy_lf = gaussian(gy, sigma, channel_axis=2)
    ux = 1 / (np.sqrt((gx_lf**2).sum(axis=2)) + sharpness_lf)
    uy = 1 / (np.sqrt((gy_lf**2).sum(axis=2)) + sharpness_lf)
    ux = gaussian(ux, sigma)
    uy = gaussian(uy, sigma)
    # print('ux', ux.min(), ux.max(), ux.mean(), ux.std())
    # print('uy', uy.min(), uy.max(), uy.mean(), uy.std())

    uwx = ux * wx
    uwy = uy * wy
    # print('uwx', uwx.min(), uwx.max(), uwx.mean(), uwx.std())
    # print('uwy', uwy.min(), uwy.max(), uwy.mean(), uwy.std())
    return uwx, uwy


def solve_img(img, uwx, uwy, lam=0.01):
    import cupy as cp
    from cupyx.scipy.sparse import csr_matrix as cp_csr_array
    from cupyx.scipy.sparse.linalg import cg as cp_cg
    from scipy.sparse import dia_array

    H, W, C = img.shape
    size = H * W

    print("Begin building A")
    e = np.pad(uwx, ((0, 1), (0, 0)))
    e = -lam * e.flatten()
    w = np.pad(e[:-W], ((W, 0),))
    s = np.pad(uwy, ((0, 0), (0, 1)))
    s = -lam * s.flatten()
    n = np.pad(s[:-1], ((1, 0),))
    d = 1 - (e + w + s + n)
    A = dia_array(([d, s, n, e, w], [0, -1, 1, -W, W]), shape=(size, size))
    A = A.tocsr()

    A = cp_csr_array(A)
    print("End building A")

    out = np.empty_like(img)
    for i in range(C):
        b = img[:, :, i].flatten()

        print("Begin solve")
        b = cp.asarray(b)
        x, info = cp_cg(A, b, tol=1e-3, atol=1e-3, maxiter=10**3)
        if info != 0:
            print("info", info)
        x = cp.asnumpy(x)
        print("End solve")

        out[:, :, i] = x.reshape((H, W))

    return out


def rtv_smooth(img, sigma=3, max_iter=10, tol=1e-4):
    for i in range(max_iter):
        print("rtv_smooth", i)

        print("Begin compute_texture_weights")
        uwx, uwy = compute_texture_weights(img, sigma)
        print("End compute_texture_weights")

        img_new = solve_img(img, uwx, uwy)

        norm = ((img_new - img) ** 2).mean()
        if norm < tol:
            break

        img = img_new

    return img_new
