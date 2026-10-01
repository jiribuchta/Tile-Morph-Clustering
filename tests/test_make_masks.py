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
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import pyvips
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
stub.OpenSlide = StubSlide  # build_slide_mask/verify_mask call openslide.OpenSlide
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


def test_stream_slide_tiles_assigns_nearest_centroid(tmp_path: Path):
    """stream_slide_tiles: nearest-centroid, N>32 tiles/slide in one batch (axis regression)."""
    K, _D, N = 4, 4, 40  # more tiles than centroids, in a single batch
    C = np.eye(K, dtype=np.float32)  # centroids on the 4 axes

    sids = np.array([b"sA"] * N, dtype=object)  # one wanted slide
    x = np.arange(N, dtype=np.int64) * 64
    y = np.zeros(N, dtype=np.int64)
    expected = np.array([j % K for j in range(N)])  # tile j -> centroid j%K
    emb = C[expected]  # each embedding points at its expected centroid

    pf = tmp_path / "tiles.parquet"
    pd.DataFrame(
        {"slide_id": sids, "x": x, "y": y, "embedding": list(emb)}
    ).to_parquet(pf, index=False)

    got = make_masks.stream_slide_tiles([str(pf)], b"sA".hex(), C)
    assert len(got) == N  # every tile assigned (no index-32 overflow)
    # labels use the +1 offset: cluster c -> label c+1
    assert all(c == e + 1 for (_, _, c), e in zip(got, expected, strict=True))


def test_main_run_end_to_end(tmp_path: Path):
    """main_run in one shot: data -> index -> masks -> manifest -> report conf."""
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
    K = np.eye(8, dtype=np.float32)

    clu = tmp_path / "clustering"
    clu.mkdir()
    pd.DataFrame([
        {"slide_id": "A", "path": str(wsi_a), "level": level,
         "tile_extent_x": tex_w, "tile_extent_y": tex_h},
        {"slide_id": "B", "path": str(wsi_b), "level": level,
         "tile_extent_x": tex_w, "tile_extent_y": tex_h},
    ]).to_parquet(clu / "slides.parquet", index=False)
    np.save(clu / "centroids.npy", K)

    # sharded tile data (the `data.paths.local` dir): one interleaved part
    data_dir = tmp_path / "data"
    (data_dir / "tiles").mkdir(parents=True)
    _shard(data_dir / "tiles" / "p0.parquet", [
        ("A", 0, 0, K[4]),
        ("A", 32, 32, K[7]),
        ("B", 0, 0, K[6]),
    ])

    out = tmp_path / "masks"
    cfg = OmegaConf.create({
        "clustering_out": str(clu),
        "data": {"paths": {"local": str(data_dir)}},
        "slides": 2, "out": str(out),
    })
    make_masks.main_run(cfg)

    # one mask per slide, named after the WSI stem (Path(...).stem strips .mrxs)
    assert (out / "slide_A.tiff").exists()
    assert (out / "slide_B.tiff").exists()
    # index was built + cached (the no-re-scan win)
    assert (out / "slide_parts_index.json").exists()
    idx = json.loads((out / "slide_parts_index.json").read_text())
    assert set(idx) == {"A", "B"}
    # manifest records both slides + k
    mani = json.loads((out / "manifest.json").read_text())
    assert {s["slide_id"] for s in mani["slides"]} == {"A", "B"}
    assert mani["k"] == 8
    # report config written, patched to SlideRetriever over the 2 WSIs
    rpt = out / "report_conf" / "reporter" / "tile_morph_k32.yaml"
    assert rpt.exists()
    y = rpt.read_text()
    assert "report.masks.SlideRetriever" in y
    assert str(wsi_a) in y and str(wsi_b) in y
    # painted tiles carry cluster id + 1 (nearest-centroid); slides independent
    a = stub.OpenSlideSlide(str(out / "slide_A.tiff"))
    assert a.level_dimensions[0] == (l0_w, l0_h)
    assert int(a._l0.crop(1, 1, 1, 1).numpy().item()) == 5   # A tile (0,0) -> cluster 4
    assert int(a._l0.crop(65, 65, 1, 1).numpy().item()) == 8  # A tile (32,32) -> cluster 7
    a.close()
    b = stub.OpenSlideSlide(str(out / "slide_B.tiff"))
    assert int(b._l0.crop(1, 1, 1, 1).numpy().item()) == 7   # B tile (0,0) -> cluster 6
    b.close()

    # re-run: masks already on disk -> all skipped (resumable, no re-scan)
    make_masks.main_run(cfg)
    mani2 = json.loads((out / "manifest.json").read_text())
    assert {s["slide_id"] for s in mani2["slides"]} == set()  # none newly written


def test_select_slides_count_and_name():
    """slides=0|'all' = all; N = first N (carcinoma order); <name/ID> = the match."""
    df = pd.DataFrame({
        "slide_id": [b"sA", b"sB", b"sC"],
        "path": ["/wsi/A.mrxs", "/wsi/B.mrxs", "/wsi/C.mrxs"],
    })
    all_ids = df["slide_id"].tolist()
    assert make_masks._select_slides(df, 0)["slide_id"].tolist() == all_ids
    assert make_masks._select_slides(df, "all")["slide_id"].tolist() == all_ids
    assert make_masks._select_slides(df, 2)["path"].tolist() == ["/wsi/A.mrxs", "/wsi/B.mrxs"]
    assert make_masks._select_slides(df, "B.mrxs")["path"].tolist() == ["/wsi/B.mrxs"]
    assert make_masks._select_slides(df, "a.mrxs")["path"].tolist() == ["/wsi/A.mrxs"]


def test_index_build_and_cache_roundtrip(tmp_path: Path):
    """build_index -> save_index -> load_index round-trip (interleaved parts)."""
    p0, p1 = tmp_path / "p0.parquet", tmp_path / "p1.parquet"
    _shard(p0, [(b"\x01" * 8, 0, 0, [1, 0, 0, 0]), (b"\x02" * 8, 0, 0, [0, 1, 0, 0])])
    _shard(p1, [(b"\x01" * 8, 0, 0, [0, 0, 1, 0]), (b"\x02" * 8, 0, 0, [0, 0, 0, 1])])
    parts = [("p0", str(p0), 0), ("p1", str(p1), 0)]
    index = make_masks.build_index(parts)
    assert set(index) == {b"\x01".hex() * 8, b"\x02".hex() * 8}
    assert sorted(index[b"\x01".hex() * 8]) == [str(p0), str(p1)]  # both parts

    ipath = tmp_path / "idx.json"
    make_masks.save_index(ipath, index)
    assert make_masks.load_index(ipath) == index
    assert make_masks.load_index(tmp_path / "nope.json") is None


def test_write_report_conf(tmp_path: Path):
    """write_report_conf: masks+WSI map -> patched SlideRetriever; missing -> None."""
    masks = tmp_path / "masks"
    masks.mkdir()
    (masks / "slide_A.tiff").write_bytes(b"x")
    (masks / "slide_B.tiff").write_bytes(b"x")

    ws = tmp_path / "ws"
    ws.mkdir()
    wsi_a, wsi_b = ws / "slide_A.mrxs", ws / "slide_B.mrxs"
    clu = tmp_path / "clu"
    clu.mkdir()
    pd.DataFrame({"path": [str(wsi_a), str(wsi_b)]}).to_parquet(
        clu / "slides.parquet", index=False
    )
    # all WSIs missing on this machine -> skipped, not fatal
    assert make_masks.write_report_conf(masks, clu / "slides.parquet", tmp_path / "rc") is None

    # WSIs that exist -> yaml written + patched
    wsi_a.touch()
    wsi_b.touch()
    dest = make_masks.write_report_conf(masks, clu / "slides.parquet", tmp_path / "rc2")
    y = dest.read_text()
    assert "report.masks.SlideRetriever" in y
    assert "paths:" in y and str(wsi_a) in y


# ---- make_masks: per-slide, resumable, failure-isolated --------------------


def _fake_wsi(tmp_path: Path, name: str, l0_w: int, l0_h: int) -> Path:
    f = tmp_path / name
    pyvips.Image.tiffsave(
        pyvips.Image.new_from_array(np.zeros((l0_h, l0_w), np.uint8)),
        f, bigtiff=True, tile=True, tile_width=256, tile_height=256,
        pyramid=True, compression=pyvips.enums.ForeignTiffCompression.DEFLATE,
    )
    return f


def _shard(path: Path, rows) -> None:
    pq.write_table(pa.table({
        "slide_id": [r[0] for r in rows],
        "x": [r[1] for r in rows],
        "y": [r[2] for r in rows],
        "embedding": [list(r[3]) for r in rows],
    }), path)


def test_stream_slide_tiles_reads_only_its_parts(tmp_path: Path):
    """stream_slide_tiles: nearest-centroid cluster, only the target slide, only
    the given parts.
    """
    K, D = 4, 4
    C = np.eye(K, dtype=np.float32)
    a, b = b"\x01" * 8, b"\x02" * 8  # two slides, bytes -> hex join key
    # A: 3 tiles -> centroids 0,1,2 ; B: 1 tile -> centroid 3
    shard0 = tmp_path / "p0.parquet"
    _shard(shard0, [(a, 0, 0, C[0]), (a, 64, 0, C[1]), (b, 0, 0, C[3])])
    shard1 = tmp_path / "p1.parquet"
    _shard(shard1, [(a, 32, 32, C[2])])  # A's 3rd tile lives in the 2nd part

    tiles = make_masks.stream_slide_tiles(
        [str(shard0), str(shard1)], a.hex(), C
    )
    # only A's tiles (3), never B's; clusters 0,1,2 (+1 offset) at the right coords
    assert sorted(tiles) == [(0, 0, 1), (32, 32, 3), (64, 0, 2)]


def test_process_writes_skips_and_isolates(tmp_path: Path):
    """process_slides(): writes a mask per slide; skips existing; a WSI that fails
    to open is logged+skipped (not fatal); a slide with no parts is failed.
    """
    l0_w, l0_h, level, tex_w, tex_h = 1024, 512, 1, 64, 64
    K, D = 4, 4
    C = np.eye(K, dtype=np.float32)
    sid_a, sid_b, sid_c = b"\x01" * 8, b"\x02" * 8, b"\x03" * 8

    wsi_a = _fake_wsi(tmp_path, "slide_A.mrxs", l0_w, l0_h)
    wsi_b = _fake_wsi(tmp_path, "slide_B.mrxs", l0_w, l0_h)
    # slide_C points at a WSI that does NOT exist (simulates not-mounted)

    masks = tmp_path / "masks"
    shards = tmp_path / "shards"
    shards.mkdir()
    s0 = shards / "p0.parquet"
    _shard(s0, [(sid_a, 0, 0, C[0]), (sid_b, 0, 0, C[1])])
    s1 = shards / "p1.parquet"
    _shard(s1, [(sid_a, 32, 32, C[2])])

    clu = tmp_path / "clu"
    clu.mkdir()
    slides_df = pd.DataFrame([
        {"slide_id": sid_a, "path": str(wsi_a), "level": level,
         "tile_extent_x": tex_w, "tile_extent_y": tex_h},
        {"slide_id": sid_b, "path": str(wsi_b), "level": level,
         "tile_extent_x": tex_w, "tile_extent_y": tex_h},
        {"slide_id": sid_c, "path": str(tmp_path / "missing.mrxs"), "level": level,
         "tile_extent_x": tex_w, "tile_extent_y": tex_h},
    ])
    slides_df.to_parquet(clu / "slides.parquet", index=False)
    np.save(clu / "centroids.npy", C)

    # index: A -> [s0, s1], B -> [s0]; C has NO parts (tests the no-parts fail)
    index_path = tmp_path / "slide_parts_index.json"
    index_path.write_text(json.dumps(
        {sid_a.hex(): [str(s0), str(s1)], sid_b.hex(): [str(s0)]}
    ))
    parts_by_slide = make_masks.load_index(index_path)

    targets = list(slides_df.itertuples(index=False))
    ok, skip, fail, _rows = make_masks.process_slides(
        targets, level, (tex_w, tex_h), C, parts_by_slide, masks
    )
    assert (ok, skip, fail) == (2, 0, 1)  # A+B written, C failed (no parts)
    assert (masks / "slide_A.tiff").exists()
    assert (masks / "slide_B.tiff").exists()

    # A's painted tiles carry cluster + 1 (A tile (0,0)->cluster 0->1, (32,32)->2->3)
    a = stub.OpenSlideSlide(str(masks / "slide_A.tiff"))
    assert a.level_dimensions[0] == (l0_w, l0_h)
    assert int(a._l0.crop(1, 1, 1, 1).numpy().item()) == 1
    assert int(a._l0.crop(65, 65, 1, 1).numpy().item()) == 3
    a.close()

    # re-run: both already written -> all skipped, still 0 failed
    ok, skip, fail, _rows = make_masks.process_slides(
        targets, level, (tex_w, tex_h), C, parts_by_slide, masks
    )
    assert (ok, skip, fail) == (0, 2, 1)  # 2 skipped, C fails again


def test_process_isolates_unopenable_wsi(tmp_path: Path):
    """A WSI that can't be opened (e.g. not mounted) is skipped, not fatal, and the
    other slide still gets its mask.
    """
    l0_w, l0_h, level, tex_w, tex_h = 1024, 512, 1, 64, 64
    C = np.eye(4, dtype=np.float32)
    sid_a, sid_b = b"\x01" * 8, b"\x02" * 8

    wsi_b = _fake_wsi(tmp_path, "slide_B.mrxs", l0_w, l0_h)
    # A points at a missing file -> OpenSlide(StubSlide) will fail to open it
    masks = tmp_path / "masks"
    shards = tmp_path / "shards"
    shards.mkdir()
    s0 = shards / "p0.parquet"
    _shard(s0, [(sid_a, 0, 0, C[0]), (sid_b, 0, 0, C[1])])

    slides_df = pd.DataFrame([
        {"slide_id": sid_a, "path": str(tmp_path / "nope.mrxs"), "level": level,
         "tile_extent_x": tex_w, "tile_extent_y": tex_h},
        {"slide_id": sid_b, "path": str(wsi_b), "level": level,
         "tile_extent_x": tex_w, "tile_extent_y": tex_h},
    ])
    parts_by_slide = {sid_a.hex(): [str(s0)], sid_b.hex(): [str(s0)]}

    ok, skip, fail, _rows = make_masks.process_slides(
        list(slides_df.itertuples(index=False)), level, (tex_w, tex_h), C,
        parts_by_slide, masks,
    )
    # A fails to open (missing file) -> counted failed; B still written
    assert (ok, fail) == (1, 1)
    assert (masks / "slide_B.tiff").exists()
