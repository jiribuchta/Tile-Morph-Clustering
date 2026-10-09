
import openslide, numpy as np, colorsys
from PIL import Image

p = '/mnt/projects/breast_cancer/tile_morph_clustering/masks_k32_v2/2020_00109-01-N.tiff'

out = 'mask_thumb.png'

s = openslide.OpenSlide(p)
# level ~4096 px wide: small enough to read instantly, big enough to see the holes
lvl = next((i for i in range(s.level_count - 1, -1, -1)
            if s.level_dimensions[i][0] >= 4096), 0)
w, h = s.level_dimensions[lvl]
a = np.array(s.read_region((0, 0), lvl, (w, h)))
if a.ndim == 3:
    a = a[..., 0]
s.close()

lut = np.zeros((256, 3), np.uint8)                    # 0 = white (background/empty)
for c in range(1, 32):
    r, g, b = colorsys.hsv_to_rgb((c - 1) / 31 * 0.85, 1.0, 1.0)
    lut[c] = (int(r * 255), int(g * 255), int(b * 255))

Image.fromarray(lut[a]).save(out)
print(f'{out}  ({w}x{h}, level {lvl})  — white = empty/cluster0, color = cluster (hue order = id)')