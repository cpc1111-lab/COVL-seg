from pathlib import Path


def test_train_script_parser_has_core_args():
    from covl_seg.scripts.train_continual import build_parser

    parser = build_parser()
    args = parser.parse_args([
        "--config",
        "covl_seg/configs/covl_seg_vitb_ade15.yaml",
        "--output-dir",
        "work_dirs/dev",
    ])
    assert args.config.endswith("covl_seg_vitb_ade15.yaml")
    assert args.output_dir == "work_dirs/dev"
    assert args.max_tasks is None
    assert args.engine == "auto"


def test_eval_script_parser_has_resume_task_arg():
    from covl_seg.scripts.eval_continual import build_parser

    parser = build_parser()
    args = parser.parse_args([
        "--config",
        "covl_seg/configs/covl_seg_vitb_coco.yaml",
        "--output-dir",
        "work_dirs/dev",
        "--resume-task",
        "3",
    ])
    assert args.resume_task == 3
    assert args.engine == "auto"


def test_config_files_exist():
    root = Path(__file__).resolve().parents[2]
    cfg_dir = root / "covl_seg" / "configs"
    expected = {
        "covl_seg_vitb_ade15.yaml",
        "covl_seg_vitb_ade100.yaml",
        "covl_seg_vitb_coco.yaml",
        "covl_seg_vitl_ade15.yaml",
    }
    found = {p.name for p in cfg_dir.glob("*.yaml")}
    assert expected.issubset(found)
