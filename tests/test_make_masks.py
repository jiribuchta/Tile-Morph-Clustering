"""Runnable check for make_masks geometry + write/verify round-trip.

Stubs openslide (not available outside the cluster image) with a vips-backed
fake that reads the exact BigTIFF written by the pipeline, so the tested path
is: paint -> write_big_tiff -> re-open -> verify_mask, at real coordinates.
"""

import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pyvips
import pytest
from omegaconf import OmegaConf

# stub openslide before importing make_masks
stub = types.ModuleType("openslide")
sys.modules.setdefault("openslide", stub)
import make_masks  # noqa: E402


class _Region:
    def __init__(self, img, loc):
        self.img, self.loc = img, loc

    def getpixel(self, xy):
        x, y = self.loc[0] + xy[0], self.loc[1] + xy[1]
        return (int(self.img.crop(x, y, 1, 1).numpy().item()), 0, 0, 255)


class StubSlide:
    """Duck-typed openslide.OpenSlideSlide over a 2-level vips TIFF."""

    def __init__(self, path):
        self.path = str(path)
        self._l0 = pyvips.Image.new_from_file(self.path, access="random")
        self.level_dimensions = [
            (self._l0.width, self._l0.height),
            (self._l0.width // 2, self._l0.height // 2),
        ]
        self.properties = {"openslide.mpp-x": "0.47", "openslide.mpp-y": "0.47"}

    def read_region(self, loc, level, size):
        img = self._l0
        if level:
            img = img.avgdown(level)  # vips pyramid == successive halfing
        return _Region(img, loc)

    def close(self):
        pass


stub.OpenSlideSlide = StubSlide
make_masks.openslide = stub  # in case module import order differs


@pytest.fixture
def slide(tmp_path: Path) -> dict:
    l0_w, l0_h = 4096, 2048
    arr = np.zeros((l0_h, l0_w), dtype=np.uint8)
    f = tmp_path / "wsi.tiff"
    write = pyvips.Image.tiffsave
    write(
        pyvips.Image.new_from_array(arr),
        f,
        bigtiff=True,
        compression=pyvips.enums.ForeignTiffCompression.DEFLATE,
        tile=True,
        tile_width=512,
        tile_height=512,
        xres=1000 / 0.47,
        yres=1000 / 0.47,
        pyramid=True,
    )
    return {
        "path": str(f),
        "extent_x": l0_w // 2,
        "extent_y": l0_h // 2,
        "tile_extent_x": 64,
        "tile_extent_y": 64,
        "mpp_x": 0.94,
        "mpp_y": 0.94,
        "level": 1,
        "carcinoma": False,
        "slide_id": "x",
    }


def test_paint_write_verify_roundtrip(tmp_path: Path, slide: dict):
    tex = (64, 64)
    tiles = [
        (0, 0, 5),
        (128, 64, 7),
        (slide["extent_x"] - 64, slide["extent_y"] - 64, 3),
        (slide["extent_x"] - 32, slide["extent_y"] - 32, 9),  # crosses edge
    ]
    arr, mpp_x, mpp_y = make_masks.build_slide_mask(slide, tiles, 1, tex)
    assert arr.shape == (slide["extent_y"] * 2, slide["extent_x"] * 2)

    dest = tmp_path / "mask.tiff"
    img = pyvips.Image.new_from_array(arr)
    make_masks.write_big_tiff(img, dest, mpp_x, mpp_y)

    # geometry: mask level 0 == WSI level 0
    s = StubSlide(dest)
    assert s.level_dimensions[0] == (slide["extent_x"] * 2, slide["extent_y"] * 2)
    s.close()

    # every tile's center pixel must equal its cluster
    make_masks.verify_mask(dest, tiles, 1)

    # values outside all tiles stay 0
    assert int(StubSlide(dest)._l0.crop(600, 1500, 1, 1).numpy().item()) == 0


def test_paint_clamps_to_bounds():
    arr = np.zeros((100, 100), np.uint8)
    make_masks.paint(arr, 95, 95, 64, 64, 4, (1.0, 1.0))
    assert arr[99, 99] == 4
    assert arr[0, 0] == 0


def test_main_run_sampled_end_to_end(tmp_path: Path):
    """main_run: clustering dir -> one .tiff per slide + manifest.json (sampled mode).

    2 synthetic slides; only slide A has assigned tiles. Verifies the full
    config-driven path: parquet reads, slide limit, per-slide mask, manifest.
    """
    l0_w, l0_h, level, tex_w, tex_h = 2048, 1024, 1, 64, 64

    def _wsi(name: str) -> Path:
        f = tmp_path / name
        pyvips.Image.tiffsave(
            pyvips.Image.new_from_array(np.zeros((l0_h, l0_w), np.uint8)),
            f, bigtiff=True, tile=True, tile_width=512, tile_height=512,
            pyramid=True, compression=pyvips.enums.ForeignTiffCompression.DEFLATE,
        )
        return f

    wsi_a, wsi_b = _wsi("slide_A.mrxs"), _wsi("slide_B.mrxs")

    clu = tmp_path / "clustering"
    clu.mkdir()
    pd.DataFrame([
        {"slide_id": "A", "path": str(wsi_a), "level": level,
         "tile_extent_x": tex_w, "tile_extent_y": tex_h},
        {"slide_id": "B", "path": str(wsi_b), "level": level,
         "tile_extent_x": tex_w, "tile_extent_y": tex_h},
    ]).to_parquet(clu / "slides.parquet", index=False)
    # level-1 coords: tile (0,0) + overlapping tile per slide
    pd.DataFrame([
        {"slide_id": "A", "x": 0, "y": 0, "cluster": 4},
        {"slide_id": "A", "x": 32, "y": 32, "cluster": 9},
        {"slide_id": "B", "x": 0, "y": 0, "cluster": 6},
    ]).to_parquet(clu / "assignments.parquet", index=False)
    np.save(clu / "centroids.npy", np.eye(8, dtype=np.float32))

    out = tmp_path / "masks"
    cfg = OmegaConf.create({
        "clustering_out": str(clu), "mode": "sampled",
        "slides": 2, "out": str(out),
    })
    make_masks.main_run(cfg)

    # one mask per slide, named after the WSI stem (Path(...).stem strips .mrxs)
    assert (out / "slide_A.tiff").exists()
    assert (out / "slide_B.tiff").exists()
    # manifest records both slides + k
    mani = json.loads((out / "manifest.json").read_text())
    assert {s["slide_id"] for s in mani["slides"]} == {"A", "B"}
    assert mani["k"] == 8
    # painted tiles carry their cluster ids; slides are independent
    a = stub.OpenSlideSlide(str(out / "slide_A.tiff"))
    assert a.level_dimensions[0] == (l0_w, l0_h)
    assert int(a._l0.crop(1, 1, 1, 1).numpy().item()) == 4   # A tile (0,0)
    assert int(a._l0.crop(65, 65, 1, 1).numpy().item()) == 9  # A tile (32,32)
    a.close()
    b = stub.OpenSlideSlide(str(out / "slide_B.tiff"))
    assert int(b._l0.crop(1, 1, 1, 1).numpy().item()) == 6   # B tile (0,0)
    b.close()
