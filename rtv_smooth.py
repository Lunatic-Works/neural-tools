import numpy as np
from scipy.sparse import dia_array
from scipy.sparse.linalg import bicgstab
from skimage.color import rgb2xyz
from skimage.filters import gaussian


def cross_sum(a):
    a_pad = np.pad(a, ((1, 1), (1, 1), (0, 0)))
    out = (a_pad[2:, 1:-1] + a_pad[:-2, 1:-1] + a_pad[1:-1, 2:] +
           a_pad[1:-1, :-2])
    return out


def tv_smooth(img, mask, max_iter=10**3, tol=1e-6, eps=1e-15):
    assert mask.ndim == 2
    mask = mask[:, :, None]

    img_old = img
    mask_inv = 1 - mask
    for _ in range(max_iter):
        img_new = mask_inv * img_old
        img_new = cross_sum(img_new)
        mask_inv = cross_sum(mask_inv)
        img_new /= mask_inv + eps
        img_new = mask * img_new + (1 - mask) * img

        norm = ((img_new - img_old)**2).mean()
        if norm < tol:
            break

        img_old = img_new
        mask_inv[mask_inv > 0] = 1

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
    H, W, C = img.shape
    size = H * W

    e = np.concatenate([uwx, np.zeros((1, W))], axis=0)
    e = -lam * e.flatten()
    w = np.concatenate([np.zeros(W), e[:-W]])
    s = np.concatenate([uwy, np.zeros((H, 1))], axis=1)
    s = -lam * s.flatten()
    n = np.concatenate([np.zeros(1), s[:-1]])
    d = 1 - (e + w + s + n)
    A = dia_array(([d, s, n, e, w], [0, -1, 1, -W, W]), shape=(size, size))
    A = A.tocsr()

    out = np.empty_like(img)
    for i in range(C):
        b = img[:, :, i].flatten()
        x, info = bicgstab(A, b, tol=1e-2, atol=1e-2, maxiter=10**3)
        if info != 0:
            print('info', info)
        out[:, :, i] = x.reshape((H, W))

    return out


def rtv_smooth(img, sigma=3, max_iter=10, tol=1e-4):
    img_old = img
    for i in range(max_iter):
        print('rtv_smooth', i)

        uwx, uwy = compute_texture_weights(img_old, sigma)
        img_new = solve_img(img, uwx, uwy)

        norm = ((img_new - img_old)**2).mean()
        if norm < tol:
            break

        img_old = img_new

    return img_new
