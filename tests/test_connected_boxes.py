# -*- coding: utf-8 -*-
"""连通域切框 —— 一块完整锈斑必须是一个框，不能炸成几十处。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

import numpy as np
import pytest

from aircraft_vqa import geometry as G


def _force_bfs(monkeypatch):
    monkeypatch.setattr(G, "_components", lambda small: None)


def test_one_solid_blob_is_one_box():
    """真实踩到的坑：一行的候选像素列表在 BFS 开始前就算好了，
    循环里不复查 visited，于是同一连通域内每个已访问像素都被当成新起点，
    各自吐出一个 1 像素的框 —— 30x30 的方块变成 12 处缺陷。"""
    m = np.zeros((64, 64), dtype=bool)
    m[10:40, 10:40] = True
    assert G.connected_boxes(m) == [[10, 10, 40, 40]]


def test_one_solid_blob_is_one_box_bfs_path(monkeypatch):
    _force_bfs(monkeypatch)
    m = np.zeros((64, 64), dtype=bool)
    m[10:40, 10:40] = True
    assert G.connected_boxes(m) == [[10, 10, 40, 40]]


def test_separate_blobs_stay_separate():
    m = np.zeros((128, 128), dtype=bool)
    m[10:30, 10:30] = True
    m[60:100, 60:100] = True
    m[10:24, 80:110] = True
    boxes = G.connected_boxes(m)
    assert len(boxes) == 3, boxes
    assert boxes == sorted(boxes,
                           key=lambda b: -(b[2] - b[0]) * (b[3] - b[1])), "应按面积降序"


def test_full_frame_blob_is_single_box():
    """整幅几乎全是锈的图，只能给一个框。"""
    m = np.ones((512, 512), dtype=bool)
    assert len(G.connected_boxes(m)) == 1


@pytest.mark.parametrize("seed", range(40))
def test_scipy_and_bfs_paths_agree(seed, monkeypatch):
    """有没有 scipy 都必须切出一模一样的框。"""
    rng = np.random.default_rng(seed)
    s = int(rng.integers(24, 160))
    m = rng.random((s, s)) > rng.uniform(0.3, 0.9)
    fast = G.connected_boxes(m)
    _force_bfs(monkeypatch)
    slow = G.connected_boxes(m)
    assert fast == slow, (seed, s, fast[:3], slow[:3])


def test_components_helpers_agree_directly():
    rng = np.random.default_rng(7)
    for _ in range(60):
        s = int(rng.integers(16, 90))
        m = rng.random((s, s)) > rng.uniform(0.3, 0.95)
        a = G._components(m)
        if a is None:
            pytest.skip("环境里没有 scipy")
        assert sorted(a) == sorted(G._components_bfs(m))


def test_tiny_defect_still_yields_a_box():
    """缺陷小到被面积阈值滤光时，退化为整体外接框，不能丢样本。"""
    m = np.zeros((512, 512), dtype=bool)
    m[100, 100] = True
    boxes = G.connected_boxes(m)
    assert len(boxes) == 1
    assert boxes[0] == [100, 100, 101, 101]


def test_empty_mask_gives_no_boxes():
    assert G.connected_boxes(np.zeros((32, 32), dtype=bool)) == []


def test_box_count_is_capped():
    """碎点再多也不能无限出框，否则清点类问答直接教废。"""
    rng = np.random.default_rng(3)
    m = rng.random((256, 256)) > 0.5      # 大量互不相连的碎点
    assert len(G.connected_boxes(m)) <= 12
