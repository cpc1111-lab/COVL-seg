from pathlib import Path


def test_replay_buffer_priority_order_and_capacity():
    from covl_seg.continual.replay_buffer import ReplayItem, SACRReplayBuffer

    buffer = SACRReplayBuffer(max_total_items=3, max_per_class=2)
    items = [
        ReplayItem("img_a.jpg", "lbl_a.png", class_id=1, priority=0.3),
        ReplayItem("img_b.jpg", "lbl_b.png", class_id=1, priority=0.9),
        ReplayItem("img_c.jpg", "lbl_c.png", class_id=2, priority=0.5),
        ReplayItem("img_d.jpg", "lbl_d.png", class_id=3, priority=0.1),
    ]
    for item in items:
        buffer.add(item)

    kept = buffer.items()
    assert len(kept) == 3
    assert all(x.priority >= 0.3 for x in kept)
    assert kept[0].priority >= kept[1].priority >= kept[2].priority


def test_replay_buffer_enforces_per_class_limit():
    from covl_seg.continual.replay_buffer import ReplayItem, SACRReplayBuffer

    buffer = SACRReplayBuffer(max_total_items=10, max_per_class=2)
    for idx, score in enumerate([0.1, 0.4, 0.8, 0.2]):
        buffer.add(ReplayItem(f"img_{idx}.jpg", f"lbl_{idx}.png", class_id=4, priority=score))

    class_items = [x for x in buffer.items() if x.class_id == 4]
    assert len(class_items) == 2
    assert {x.priority for x in class_items} == {0.8, 0.4}


def test_replay_buffer_round_trip_serialization(tmp_path: Path):
    from covl_seg.continual.replay_buffer import ReplayItem, SACRReplayBuffer

    buffer = SACRReplayBuffer(max_total_items=5, max_per_class=3)
    buffer.add(ReplayItem("img_x.jpg", "lbl_x.png", class_id=7, priority=0.77))
    buffer.add(ReplayItem("img_y.jpg", "lbl_y.png", class_id=8, priority=0.12))

    out_file = tmp_path / "buffer.json"
    buffer.save(out_file)

    loaded = SACRReplayBuffer.load(out_file)
    loaded_items = loaded.items()
    assert len(loaded_items) == 2
    assert loaded_items[0].priority == 0.77
    assert loaded_items[1].class_id == 8
