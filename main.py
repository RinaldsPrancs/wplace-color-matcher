import numpy as np
import matplotlib.image as mpimg
from PIL import Image


def hex_to_rgb(hex_code: str):
    s = hex_code.lstrip('#')
    return tuple(int(s[i:i+2], 16) for i in (0, 2, 4))

def srgb_to_linear(u):
    return np.where(u <= 0.04045, u / 12.92, ((u + 0.055) / 1.055) ** 2.4)

def rgb_to_lab(rgb_float): 
    r, g, b = srgb_to_linear(rgb_float[..., 0]), srgb_to_linear(rgb_float[..., 1]), srgb_to_linear(rgb_float[..., 2])

    X = 0.4124564*r + 0.3575761*g + 0.1804375*b
    Y = 0.2126729*r + 0.7151522*g + 0.0721750*b
    Z = 0.0193339*r + 0.1191920*g + 0.9503041*b

    Xn, Yn, Zn = 0.95047, 1.00000, 1.08883
    x, y, z = X / Xn, Y / Yn, Z / Zn

    eps = (6/29)**3 
    kappa = (29/3)**2 / 3 
    def f(t):
        return np.where(t > eps, np.cbrt(t), (t * (1/3) * (29/6)**2) + (4/29))

    fx, fy, fz = f(x), f(y), f(z)

    L = 116*fy - 16
    a = 500*(fx - fy)
    b2 = 200*(fy - fz)
    return np.stack([L, a, b2], axis=-1)  

# IMAGE NAME
img = mpimg.imread("download.png") 
has_alpha = (img.shape[-1] == 4)


rgb = img[..., :3]
alpha = None
if has_alpha:
    alpha = img[..., 3]


if rgb.dtype == np.uint8:
    rgb_f = rgb.astype(np.float32) / 255.0
else:
    rgb_f = rgb.astype(np.float32)


# F2P PALETTE
hex_palette = [
    "#000000", "#3c3c3c", "#787878", "#d2d2d2", "#ffffff", "#600018", "#ed1c24",
    "#ff7f27", "#f6aa09", "#f9dd3b", "#fffabc", "#0eb968", "#13e67b", "#87ff5e",
    "#0c816e", "#10aea6", "#13e1be", "#28509e", "#4093e4", "#60f7f2", "#6b50f6",
    "#99b1fb", "#780c99", "#aa38b9", "#e09ff9", "#cb007a", "#ec1f80", "#f38da9",
    "#684634", "#95682a", "#f8b277"
]

# FULL COLOR PALETTE

# hex_palette = [
#     "#000000", "#3c3c3c", "#787878", "#aaaaaa", "#d2d2d2", "#ffffff",
#     "#600018", "#a50e1e", "#ed1c24", "#fa8072",
#     "#e45c1a", "#ff7f27", "#f6aa09", "#f9dd3b", "#fffabc",
#     "#9c8431", "#c5ad31", "#e8d45f",
#     "#4a6b3a", "#5a944a", "#84c573", "#0eb968", "#13e67b", "#87ff5e",
#     "#0c816e", "#10aea6", "#13e1be",
#     "#0f799f", "#60f7f2", "#bbfaf2",
#     "#28509e", "#4093e4", "#7dc7ff",
#     "#4d31b8", "#6b50f6", "#99b1fb",
#     "#4a4284", "#7a71c4", "#b5aef1",
#     "#780c99", "#aa38b9", "#e09ff9",
#     "#cb007a", "#ec1f80", "#f38da9",
#     "#9b5249", "#d18078", "#fab6a4",
#     "#684634", "#95682a", "#dba463",
#     "#7b6352", "#9c846b", "#d6b594",
#     "#d18051", "#f8b277", "#ffc5a5",
#     "#6d643f", "#948c6b", "#cdc59e",
#     "#333941", "#6d758d", "#b3b9d1"
# ]


palette_rgb_u8 = np.array([hex_to_rgb(h) for h in hex_palette], dtype=np.uint8)
palette_rgb_f = palette_rgb_u8.astype(np.float32) / 255.0

palette_lab = rgb_to_lab(palette_rgb_f)  # (P,3)

H, W = rgb_f.shape[:2]
img_lab = rgb_to_lab(rgb_f.reshape(-1, 3)).reshape(H*W, 3)  # (HW,3)

diff = img_lab[:, None, :] - palette_lab[None, :, :]
dist2 = np.sum(diff * diff, axis=-1)

nearest_idx = np.argmin(dist2, axis=1)  # (HW,)
remapped_rgb = palette_rgb_u8[nearest_idx].reshape(H, W, 3)


if has_alpha:
    if alpha.dtype != np.uint8:
        alpha_u8 = (alpha * 255.0).clip(0, 255).astype(np.uint8)
    else:
        alpha_u8 = alpha
    out = np.dstack([remapped_rgb, alpha_u8])
    Image.fromarray(out, mode="RGBA").save("image_cmpltd.png")
else:
    Image.fromarray(remapped_rgb, mode="RGB").save("image_cmpltd.png")

print("Saved")
