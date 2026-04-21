import io
import sys
import zipfile
from pathlib import Path

import pytest


def test_resolve_dataset_root_precedence(tmp_path, monkeypatch):
    from covl_seg.scripts.bootstrap_coco_train import resolve_datasets_root

    repo_root = tmp_path / "repo"
    env_root = tmp_path / "env-datasets"
    monkeypatch.setenv("DETECTRON2_DATASETS", str(env_root))

    assert resolve_datasets_root(None, repo_root=repo_root) == env_root

    cli_root = tmp_path / "cli-datasets"
    assert resolve_datasets_root(str(cli_root), repo_root=repo_root) == cli_root

    monkeypatch.delenv("DETECTRON2_DATASETS", raising=False)
    assert resolve_datasets_root(None, repo_root=repo_root) == repo_root / "datasets"


def test_resolve_dataset_root_empty_env_falls_back(monkeypatch):
    from covl_seg.scripts.bootstrap_coco_train import resolve_datasets_root

    repo_root = Path("/repo")
    monkeypatch.setenv("DETECTRON2_DATASETS", "")
    assert resolve_datasets_root(None, repo_root=repo_root) == repo_root / "datasets"


def test_parser_rejects_force_and_skip_download_together():
    from covl_seg.scripts.bootstrap_coco_train import build_parser

    parser = build_parser()

    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(
            [
                "--output-dir",
                "work_dirs/dev",
                "--force-download",
                "--skip-download",
            ]
        )

    assert isinstance(exc_info.value.code, int)
    assert exc_info.value.code != 0


def test_main_dry_run_does_not_execute_subprocess(monkeypatch, capsys):
    from covl_seg.scripts.bootstrap_coco_train import main

    def _unexpected_subprocess(*_args, **_kwargs):
        raise AssertionError("subprocess.run should not be called in dry-run")

    monkeypatch.setattr(
        "covl_seg.scripts.bootstrap_coco_train.subprocess.run",
        _unexpected_subprocess,
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "bootstrap_coco_train.py",
            "--output-dir",
            "work_dirs/dev",
            "--dry-run",
        ],
    )

    main()
    output = capsys.readouterr().out
    assert "[dry-run]" in output
    assert "train_continual" in output


def test_validate_coco_layout_lists_missing_paths(tmp_path):
    from covl_seg.scripts.bootstrap_coco_train import validate_coco_layout

    with pytest.raises(ValueError, match="Missing required COCO-Stuff paths") as exc_info:
        validate_coco_layout(tmp_path)

    message = str(exc_info.value)
    assert str(tmp_path / "coco-stuff" / "images" / "train2017") in message
    assert str(tmp_path / "coco-stuff" / "images" / "val2017") in message
    assert str(tmp_path / "coco-stuff" / "annotations" / "train2017") in message
    assert str(tmp_path / "coco-stuff" / "annotations" / "val2017") in message
    assert str(tmp_path / "coco-stuff" / "annotations_detectron2" / "train2017") in message
    assert str(tmp_path / "coco-stuff" / "annotations_detectron2" / "val2017") in message


def test_validate_coco_layout_accepts_complete_layout(tmp_path):
    from covl_seg.scripts.bootstrap_coco_train import validate_coco_layout

    (tmp_path / "coco-stuff" / "images" / "train2017").mkdir(parents=True)
    (tmp_path / "coco-stuff" / "images" / "val2017").mkdir(parents=True)
    (tmp_path / "coco-stuff" / "annotations" / "train2017").mkdir(parents=True)
    (tmp_path / "coco-stuff" / "annotations" / "val2017").mkdir(parents=True)
    (tmp_path / "coco-stuff" / "annotations_detectron2" / "train2017").mkdir(parents=True)
    (tmp_path / "coco-stuff" / "annotations_detectron2" / "val2017").mkdir(parents=True)

    validate_coco_layout(tmp_path)


def test_build_train_command_enforces_d2_engine(tmp_path):
    from covl_seg.scripts.bootstrap_coco_train import build_train_command

    command = build_train_command(
        config_path="covl_seg/configs/covl_seg_vitb_ade15.yaml",
        output_dir=tmp_path / "work_dirs" / "bootstrap",
        seed=3,
        max_tasks=2,
    )

    assert command[0:3] == [sys.executable, "-m", "covl_seg.scripts.train_continual"]
    assert "--engine" in command
    assert command[command.index("--engine") + 1] == "d2"


def test_build_train_command_omits_max_tasks_when_none(tmp_path):
    from covl_seg.scripts.bootstrap_coco_train import build_train_command

    command = build_train_command(
        config_path="covl_seg/configs/covl_seg_vitb_ade15.yaml",
        output_dir=tmp_path / "work_dirs" / "bootstrap",
        seed=3,
        max_tasks=None,
    )

    assert "--max-tasks" not in command


def test_prepare_command_uses_runtime_root_and_dataset_env(monkeypatch, tmp_path):
    from covl_seg.scripts.bootstrap_coco_train import run_prepare_coco_stuff

    calls = []

    def _fake_run(cmd, check, env):
        calls.append((cmd, check, env))

    monkeypatch.setattr("covl_seg.scripts.bootstrap_coco_train.subprocess.run", _fake_run)

    runtime_root = tmp_path / "covl_seg_d2_runtime"
    datasets_root = tmp_path / "datasets"
    run_prepare_coco_stuff(runtime_root=runtime_root, datasets_root=datasets_root)

    assert len(calls) == 1
    cmd, check, env = calls[0]
    assert check is True
    assert cmd == [
        sys.executable,
        str(runtime_root / "datasets" / "prepare_coco_stuff.py"),
    ]
    assert env["DETECTRON2_DATASETS"] == str(datasets_root)


def test_training_command_launcher_sets_dataset_env(monkeypatch, tmp_path):
    from covl_seg.scripts.bootstrap_coco_train import run_training_command

    calls = []

    def _fake_run(cmd, check, env):
        calls.append((cmd, check, env))

    monkeypatch.setattr("covl_seg.scripts.bootstrap_coco_train.subprocess.run", _fake_run)

    cmd = [sys.executable, "-m", "covl_seg.scripts.train_continual"]
    datasets_root = tmp_path / "datasets"
    run_training_command(cmd=cmd, datasets_root=datasets_root)

    assert len(calls) == 1
    called_cmd, check, env = calls[0]
    assert called_cmd == cmd
    assert check is True
    assert env["DETECTRON2_DATASETS"] == str(datasets_root)


def test_download_invalid_existing_zip_triggers_redownload(monkeypatch, tmp_path):
    from covl_seg.scripts.bootstrap_coco_train import download_file_with_retries

    dest = tmp_path / "train2017.zip"
    dest.write_bytes(b"not-a-valid-zip")

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as archive:
        archive.writestr("train2017/sample.txt", "ok")
    payload = zip_buffer.getvalue()

    calls = {"count": 0}

    class _FakeResponse(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            self.close()
            return False

    def _fake_urlopen(*_args, **_kwargs):
        calls["count"] += 1
        return _FakeResponse(payload)

    monkeypatch.setattr("covl_seg.scripts.bootstrap_coco_train.urlopen", _fake_urlopen)

    result = download_file_with_retries(
        url="https://example.com/train2017.zip",
        dest=dest,
        force_download=False,
        retries=1,
    )

    assert result == dest
    assert calls["count"] == 1
    with zipfile.ZipFile(dest) as archive:
        assert archive.namelist() == ["train2017/sample.txt"]


def test_download_retry_exhaustion_raises_runtimeerror(monkeypatch, tmp_path):
    from covl_seg.scripts.bootstrap_coco_train import download_file_with_retries

    def _failing_urlopen(*_args, **_kwargs):
        raise OSError("network down")

    monkeypatch.setattr("covl_seg.scripts.bootstrap_coco_train.urlopen", _failing_urlopen)

    with pytest.raises(RuntimeError, match="Failed to download .* after 2 attempts") as exc_info:
        download_file_with_retries(
            url="https://example.com/train2017.zip",
            dest=tmp_path / "train2017.zip",
            retries=2,
        )

    assert isinstance(exc_info.value.__cause__, OSError)


def test_extract_archive_rejects_zip_slip_entries(tmp_path):
    from covl_seg.scripts.bootstrap_coco_train import extract_archive

    zip_path = tmp_path / "evil.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("../escape.txt", "pwnd")

    target_dir = tmp_path / "coco"
    with pytest.raises(ValueError, match="Unsafe archive entry"):
        extract_archive(zip_path=zip_path, target_dir=target_dir)

    assert not (tmp_path / "escape.txt").exists()


def test_archive_urls_use_coco_stuff_annotations_url():
    from covl_seg.scripts.bootstrap_coco_train import COCO_ARCHIVES

    urls = [spec["url"] for spec in COCO_ARCHIVES]
    assert (
        "http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/"
        "stuffthingmaps_trainval2017.zip"
    ) in urls
    assert "http://images.cocodataset.org/annotations/annotations_trainval2017.zip" not in urls


def test_archive_extract_targets_match_coco_stuff_layout():
    from covl_seg.scripts.bootstrap_coco_train import COCO_ARCHIVES

    by_name = {spec["name"]: spec["extract_subdir"] for spec in COCO_ARCHIVES}
    assert by_name["train2017.zip"] == Path("coco-stuff") / "images"
    assert by_name["val2017.zip"] == Path("coco-stuff") / "images"
    assert by_name["stuffthingmaps_trainval2017.zip"] == Path("coco-stuff") / "annotations"


def test_main_skips_extraction_when_coco_stuff_layout_exists(monkeypatch, tmp_path):
    from covl_seg.scripts.bootstrap_coco_train import main

    repo_root = tmp_path / "repo"
    (repo_root / "covl_seg" / "scripts").mkdir(parents=True)
    datasets_root = tmp_path / "datasets"

    (datasets_root / "coco-stuff" / "images" / "train2017").mkdir(parents=True)
    (datasets_root / "coco-stuff" / "images" / "val2017").mkdir(parents=True)
    (datasets_root / "coco-stuff" / "annotations" / "train2017").mkdir(parents=True)
    (datasets_root / "coco-stuff" / "annotations" / "val2017").mkdir(parents=True)
    (datasets_root / "coco-stuff" / "annotations_detectron2" / "train2017").mkdir(parents=True)
    (datasets_root / "coco-stuff" / "annotations_detectron2" / "val2017").mkdir(parents=True)

    download_calls = []
    extract_calls = []
    prepare_calls = []
    train_calls = []

    def _fake_download(url, dest, force_download=False, retries=3, timeout=60):
        download_calls.append((url, dest, force_download, retries, timeout))
        return dest

    def _fake_extract(zip_path, target_dir):
        extract_calls.append((zip_path, target_dir))

    def _fake_prepare(*, runtime_root, datasets_root):
        prepare_calls.append((runtime_root, datasets_root))

    def _fake_train(*, cmd, datasets_root):
        train_calls.append((cmd, datasets_root))

    monkeypatch.setattr(
        "covl_seg.scripts.bootstrap_coco_train.download_file_with_retries",
        _fake_download,
    )
    monkeypatch.setattr("covl_seg.scripts.bootstrap_coco_train.extract_archive", _fake_extract)
    monkeypatch.setattr(
        "covl_seg.scripts.bootstrap_coco_train.run_prepare_coco_stuff",
        _fake_prepare,
    )
    monkeypatch.setattr(
        "covl_seg.scripts.bootstrap_coco_train.run_training_command",
        _fake_train,
    )
    monkeypatch.setattr(
        "covl_seg.scripts.bootstrap_coco_train.Path.resolve",
        lambda self: repo_root / "covl_seg" / "scripts" / "bootstrap_coco_train.py",
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "bootstrap_coco_train.py",
            "--output-dir",
            "work_dirs/dev",
            "--datasets-root",
            str(datasets_root),
        ],
    )

    main()

    assert download_calls == []
    assert extract_calls == []
    assert len(prepare_calls) == 0
    assert len(train_calls) == 1


def test_resolve_d2_runtime_root_defaults_to_internal_vendor(tmp_path):
    from covl_seg.scripts.bootstrap_coco_train import _resolve_d2_runtime_root

    root = _resolve_d2_runtime_root(repo_root=tmp_path)
    assert root == tmp_path / "covl_seg" / "vendor" / "covl_seg_d2_runtime"


def test_ensure_coco_stuff_ready_for_training_auto_downloads_when_missing(monkeypatch, tmp_path):
    from covl_seg.scripts import bootstrap_coco_train as bootstrap

    calls = {"download": 0, "extract": 0, "prepare": 0, "validate": 0}

    def _fake_exists(_root):
        return False

    def _fake_download(url, dest, force_download=False, retries=3, timeout=60):
        calls["download"] += 1
        return dest

    def _fake_extract(_zip_path, _target_dir):
        calls["extract"] += 1

    def _fake_prepare(*, runtime_root, datasets_root):
        calls["prepare"] += 1
        assert runtime_root == tmp_path / "runtime"
        assert datasets_root == tmp_path / "datasets"

    def _fake_validate(_root):
        calls["validate"] += 1

    monkeypatch.setattr(bootstrap, "_coco_stuff_tree_exists", _fake_exists)
    monkeypatch.setattr(bootstrap, "download_file_with_retries", _fake_download)
    monkeypatch.setattr(bootstrap, "extract_archive", _fake_extract)
    monkeypatch.setattr(bootstrap, "run_prepare_coco_stuff", _fake_prepare)
    monkeypatch.setattr(bootstrap, "validate_coco_layout", _fake_validate)

    bootstrap.ensure_coco_stuff_ready_for_training(
        datasets_root=tmp_path / "datasets",
        runtime_root=tmp_path / "runtime",
    )

    assert calls["download"] == len(bootstrap.COCO_ARCHIVES)
    assert calls["extract"] == len(bootstrap.COCO_ARCHIVES)
    assert calls["prepare"] == 1
    assert calls["validate"] == 1


def test_ensure_coco_stuff_ready_skips_prepare_when_prepared_exists(monkeypatch, tmp_path):
    from covl_seg.scripts import bootstrap_coco_train as bootstrap

    calls = {"download": 0, "extract": 0, "prepare": 0, "validate": 0}

    monkeypatch.setattr(bootstrap, "_coco_stuff_tree_exists", lambda _root: True)
    monkeypatch.setattr(bootstrap, "_coco_stuff_prepared_tree_exists", lambda _root: True)
    monkeypatch.setattr(
        bootstrap,
        "download_file_with_retries",
        lambda *args, **kwargs: calls.__setitem__("download", calls["download"] + 1),
    )
    monkeypatch.setattr(
        bootstrap,
        "extract_archive",
        lambda *args, **kwargs: calls.__setitem__("extract", calls["extract"] + 1),
    )
    monkeypatch.setattr(
        bootstrap,
        "run_prepare_coco_stuff",
        lambda **kwargs: calls.__setitem__("prepare", calls["prepare"] + 1),
    )
    monkeypatch.setattr(
        bootstrap,
        "validate_coco_layout",
        lambda _root: calls.__setitem__("validate", calls["validate"] + 1),
    )

    bootstrap.ensure_coco_stuff_ready_for_training(
        datasets_root=tmp_path / "datasets",
        runtime_root=tmp_path / "runtime",
    )

    assert calls["download"] == 0
    assert calls["extract"] == 0
    assert calls["prepare"] == 0
    assert calls["validate"] == 1


def test_stream_copy_with_progress_prints_percentage(capsys):
    from covl_seg.scripts.bootstrap_coco_train import _stream_copy_with_progress

    source = io.BytesIO(b"a" * 100)
    target = io.BytesIO()
    _stream_copy_with_progress(
        source=source,
        target=target,
        total_bytes=100,
        label="sample.zip",
        phase="download",
        chunk_size=20,
    )

    assert target.getvalue() == b"a" * 100
    output = capsys.readouterr().out
    assert "Download progress [sample.zip]:" in output
    assert "100%" in output
