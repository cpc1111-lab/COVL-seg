from covl_seg.scripts.train_open_continual import build_parser


def test_parser_accepts_clip_and_method_modes():
    parser = build_parser()
    args = parser.parse_args(
        [
            "--config",
            "covl_seg/configs/covl_seg_vitb_ade15.yaml",
            "--output-dir",
            "work_dirs/dev",
            "--datasets-root",
            "datasets",
            "--col-method",
            "covl",
            "--clip-finetune",
            "attention",
            "--seg-net",
            "swin_b",
            "--enable-ctr",
            "--ewc-lambda",
            "12.5",
        ]
    )
    assert args.col_method == "covl"
    assert args.datasets_root == "datasets"
    assert args.clip_finetune == "attention"
    assert args.seg_net == "swin_b"
    assert args.enable_ctr is True
    assert args.ewc_lambda == 12.5


def test_main_d2_bootstraps_datasets(monkeypatch, tmp_path):
    from covl_seg.scripts import train_open_continual as script

    calls = {"bootstrap": 0, "run": 0}

    def _fake_bootstrap(*, datasets_root, runtime_root, force_download=False):
        calls["bootstrap"] += 1

    class _FakeTrainer:
        def run(self, max_tasks=None):
            calls["run"] += 1
            return {"tasks_executed": 1.0, "last_task": 1.0}

    monkeypatch.setattr(script, "ensure_coco_stuff_ready_for_training", _fake_bootstrap)
    monkeypatch.setattr(script.OpenContinualTrainer, "from_args", lambda _args: _FakeTrainer())
    monkeypatch.setattr(
        "sys.argv",
        [
            "train_open_continual.py",
            "--config",
            "covl_seg/configs/covl_seg_vitb_ade15.yaml",
            "--output-dir",
            str(tmp_path / "run"),
            "--engine",
            "d2",
            "--datasets-root",
            str(tmp_path / "datasets"),
            "--max-tasks",
            "1",
        ],
    )

    script.main()

    assert calls["bootstrap"] == 1
    assert calls["run"] == 1
