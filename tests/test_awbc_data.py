"""Exercise temporal alignment and split/padding semantics with real MP4/Parquet."""
import json

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from torch.utils.data import DataLoader

from awbc_data import build_datasets, ImagePreprocessConfig
from awbc_data.dataset import IMAGENET_MEAN, IMAGENET_STD


@pytest.fixture
def root(tmp_path):
    (tmp_path / "meta/episodes").mkdir(parents=True)
    (tmp_path / "data/chunk-000").mkdir(parents=True)
    (tmp_path / "videos").mkdir()
    info = {"codebase_version": "v3.0", "fps": 30, "total_episodes": 2, "total_frames": 28,
            "video_path": "videos/{video_key}-{chunk_index}-{file_index}.mp4",
            "features": {"observation.images.top": {"dtype": "video"},
                         "observation.state": {"dtype": "float32", "shape": [8]},
                         "action": {"dtype": "float32", "shape": [8]}}}
    (tmp_path / "meta/info.json").write_text(json.dumps(info))
    # Two episodes share a video. Episode 1 starts at video frame 20, not frame 0.
    with av.open(str(tmp_path / "videos/observation.images.top-0-0.mp4"), "w") as video:
        stream = video.add_stream("libx264", rate=30)
        stream.width, stream.height, stream.pix_fmt = 32, 24, "yuv420p"
        for t in range(34):
            image = np.full((24, 32, 3), t * 6, dtype=np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
                video.mux(packet)
        for packet in stream.encode():
            video.mux(packet)
    episodes, records = [], []
    for eid in range(2):
        episodes.append({"episode_index": eid, "length": 14, "dataset_from_index": eid*14,
                         "dataset_to_index": (eid+1)*14,
                         "videos/observation.images.top/chunk_index": 0,
                         "videos/observation.images.top/file_index": 0,
                         "videos/observation.images.top/from_timestamp": eid*20/30})
        for t in range(14):
            records.append({"episode_index": eid, "frame_index": t, "index": eid*14+t,
                            "timestamp": t/30, "observation.state": [eid*100+t]*8,
                            "action": [9999 if t == 13 else eid*100+t]*8})
    pq.write_table(pa.Table.from_pylist(episodes), tmp_path / "meta/episodes/episodes.parquet")
    # Episode 0 crosses a numeric file boundary; row order isn't assumed.
    pq.write_table(pa.Table.from_pylist(records[:7][::-1]), tmp_path / "data/chunk-000/file-000.parquet")
    pq.write_table(pa.Table.from_pylist(records[7:]), tmp_path / "data/chunk-000/file-001.parquet")
    return tmp_path


def test_chunks_statistics_and_alignment(root):
    train, val = build_datasets(root, image_size=(12, 16))
    assert train.episode_ids == (0,) and val.episode_ids == (1,)
    assert len(train) == len(val) == 2
    assert train.stats is val.stats
    assert train.stats["observation.state"]["max"] == [13]*8
    assert train.stats["action"]["max"] == [12]*8
    assert train.stats["action"]["count"] == 13
    tail = train[1]
    assert tail["valid_length"] == 3
    assert tail["action_is_pad"].tolist() == [False]*3 + [True]*7
    raw = train.denormalize("action", tail["action"])
    torch.testing.assert_close(raw[:, 0], torch.tensor([10, 11]+[12]*8, dtype=torch.float32))
    assert tail["observation"]["images"].shape == (1, 3, 12, 16)
    torch.testing.assert_close(train.denormalize("observation.state", tail["observation"]["state"]), torch.full((8,), 10.))
    # Check decoded pixel content as well as returned timestamps.
    for dataset, expected_frame in [(train, 10), (val, 30)]:
        sample = dataset[1]
        image = sample["observation"]["images"][0] * IMAGENET_STD + IMAGENET_MEAN
        assert abs(image.mean().item()*255 - expected_frame*6) < 4
        torch.testing.assert_close(sample["video_timestamp"], sample["video_target_timestamp"])
    assert not tail["advantage_valid"]
    assert val[0]["action"].max() > 1  # Validation isn't clipped or used to fit stats.
    train.close()
    val.close()


def test_awbc_annotations(root):
    with pytest.raises(ValueError, match="requires an annotated"):
        build_datasets(root, require_advantage=True)
    for path in sorted((root / "data").rglob("*.parquet")):
        records = pq.read_table(path).to_pylist()
        for r in records:
            t = r["frame_index"]
            r.update({"annotation.advantage": -0.1 if t < 10 else 0.2,
                      "annotation.advantage_valid": t < 13,
                      "annotation.action_chunk_start": t in (0, 10),
                      "annotation.action_chunk_valid_length": 10 if t < 10 else 3})
        pq.write_table(pa.Table.from_pylist(records), path)
    train, _ = build_datasets(root, require_advantage=True)
    assert len(train) == 2  # Repeated per-row labels don't multiply samples.
    assert train[0]["advantage_valid"]
    assert train[0]["advantage"].item() == pytest.approx(-0.1)
    assert train[1]["advantage"].item() == pytest.approx(0.2)
    train.close()


def test_spawn_worker(root):
    train, _ = build_datasets(root)
    train[0]  # Open a decoder before pickle; handles must not be serialized.
    loader = DataLoader(train, batch_size=2, num_workers=1, multiprocessing_context="spawn")
    batch = next(iter(loader))
    assert batch["action"].shape == (2, 10, 8)
    train.close()


def test_dataset_uses_configured_letterbox(root):
    config = ImagePreprocessConfig(size=(32, 32), mode="letterbox", padding_rgb=(10, 20, 30))
    train, val = build_datasets(root, image_preprocess=config)
    assert train.image_size == (32, 32)
    assert train.image_preprocessor.config == val.image_preprocessor.config == config
    image = train[0]["observation"]["images"][0]
    raw = image * IMAGENET_STD + IMAGENET_MEAN
    torch.testing.assert_close(raw[:, 0, 0], torch.tensor([10, 20, 30])/255)
    train.close()
