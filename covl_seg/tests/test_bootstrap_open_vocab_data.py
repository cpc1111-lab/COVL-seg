import io
import zipfile
from pathlib import Path


def test_parser_accepts_datasets_root_and_force_download():
    from covl_seg.scripts.bootstrap_open_vocab_data import build_parser

    parser = build_parser()
    args = parser.parse_args(["--datasets-root", "datasets", "--force-download"])
    assert args.datasets_root == "datasets"
    assert args.force_download is True


def test_ensure_open_vocab_data_skips_when_outputs_ready(monkeypatch, tmp_path):
    from covl_seg.scripts import bootstrap_open_vocab_data as script

    calls = {"download": 0, "extract": 0, "prepare": 0}

    monkeypatch.setattr(script, "_voc_outputs_ready", lambda _root: True)
    monkeypatch.setattr(script, "_pc59_ready", lambda _root: True)
    monkeypatch.setattr(script, "_pc459_ready", lambda _root: True)
    monkeypatch.setattr(
        script,
        "download_file",
        lambda *args, **kwargs: calls.__setitem__("download", calls["download"] + 1),
    )
    monkeypatch.setattr(
        script,
        "_safe_extract_tar",
        lambda *args, **kwargs: calls.__setitem__("extract", calls["extract"] + 1),
    )
    monkeypatch.setattr(
        script,
        "_run_prepare_script",
        lambda *args, **kwargs: calls.__setitem__("prepare", calls["prepare"] + 1),
    )

    script.ensure_open_vocab_eval_data_ready(
        datasets_root=tmp_path / "datasets",
        runtime_root=tmp_path / "runtime",
    )

    assert calls["download"] == 0
    assert calls["extract"] == 0
    assert calls["prepare"] == 0


def test_ensure_open_vocab_data_downloads_and_runs_prepare(monkeypatch, tmp_path):
    from covl_seg.scripts import bootstrap_open_vocab_data as script

    calls = {"download": 0, "prepare": 0}

    monkeypatch.setattr(script, "_voc_outputs_ready", lambda _root: False)
    monkeypatch.setattr(script, "_pc59_ready", lambda _root: False)
    monkeypatch.setattr(script, "_pc459_ready", lambda _root: False)
    monkeypatch.setattr(script, "_safe_extract_tar", lambda *args, **kwargs: None)
    monkeypatch.setattr(script, "_safe_extract_zip", lambda *args, **kwargs: None)
    monkeypatch.setattr(script, "_ensure_pascal_context_metadata", lambda *args, **kwargs: None)
    monkeypatch.setattr(script.shutil, "copytree", lambda *args, **kwargs: None)
    monkeypatch.setattr(script.shutil, "copy2", lambda *args, **kwargs: None)

    def _fake_download(url, dest, kind, force_download=False):
        calls["download"] += 1
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("ok", encoding="utf-8")
        return dest

    monkeypatch.setattr(script, "download_file", _fake_download)
    monkeypatch.setattr(
        script,
        "_run_prepare_script",
        lambda *args, **kwargs: calls.__setitem__("prepare", calls["prepare"] + 1),
    )

    script.ensure_open_vocab_eval_data_ready(
        datasets_root=tmp_path / "datasets",
        runtime_root=tmp_path / "runtime",
    )

    assert calls["download"] == len(script.OPEN_VOCAB_SOURCES)
    assert calls["prepare"] == 3


def test_resolve_datasets_root_prefers_cli(tmp_path):
    from covl_seg.scripts.bootstrap_open_vocab_data import resolve_datasets_root

    resolved = resolve_datasets_root(str(tmp_path / "custom"), repo_root=tmp_path)
    assert resolved == Path(tmp_path / "custom")


def test_open_vocab_sources_do_not_use_disabled_pc59_host():
    from covl_seg.scripts.bootstrap_open_vocab_data import OPEN_VOCAB_SOURCES

    urls = [spec["url"] for spec in OPEN_VOCAB_SOURCES.values()]
    assert all("codalabuser.blob.core.windows.net" not in url for url in urls)


def test_voc2012_aug_source_uses_direct_zip_download_url():
    from covl_seg.scripts.bootstrap_open_vocab_data import OPEN_VOCAB_SOURCES

    assert "dl=1" in OPEN_VOCAB_SOURCES["voc2012_aug"]["url"]


def test_download_file_redownloads_invalid_existing_zip(monkeypatch, tmp_path):
    from covl_seg.scripts.bootstrap_open_vocab_data import download_file

    dest = tmp_path / "SegmentationClassAug.zip"
    dest.write_text("not-zip", encoding="utf-8")

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as archive:
        archive.writestr("SegmentationClassAug/sample.png", "ok")
    payload = zip_buffer.getvalue()

    calls = {"count": 0}

    class _FakeResponse(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            self.close()
            return False

        @property
        def headers(self):
            return {"Content-Length": str(len(payload))}

    def _fake_urlopen(_url, timeout=120):
        calls["count"] += 1
        return _FakeResponse(payload)

    monkeypatch.setattr("covl_seg.scripts.bootstrap_open_vocab_data.urlopen", _fake_urlopen)

    result = download_file(
        url="https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip?dl=1",
        dest=dest,
        kind="zip",
        force_download=False,
    )

    assert result == dest
    assert calls["count"] == 1
    with zipfile.ZipFile(dest) as archive:
        assert archive.namelist() == ["SegmentationClassAug/sample.png"]


def test_prepare_voc_script_avoids_removed_numpy_str_alias():
    script_path = (
        Path(__file__).resolve().parents[1]
        / "vendor"
        / "covl_seg_d2_runtime"
        / "datasets"
        / "prepare_voc.py"
    )
    text = script_path.read_text(encoding="utf-8")
    assert "dtype=np.str" not in text


def test_ensure_pascal_context_metadata_creates_missing_val_split(tmp_path):
    from covl_seg.scripts.bootstrap_open_vocab_data import _ensure_pascal_context_metadata

    voc2010 = tmp_path / "VOCdevkit" / "VOC2010"
    (voc2010 / "ImageSets" / "Segmentation").mkdir(parents=True, exist_ok=True)
    (voc2010 / "ImageSets" / "Segmentation" / "val.txt").write_text("2008_000001\n", encoding="utf-8")

    _ensure_pascal_context_metadata(tmp_path)

    generated = voc2010 / "pascalcontext_val.txt"
    assert generated.exists()
    assert generated.read_text(encoding="utf-8") == "2008_000001\n"
