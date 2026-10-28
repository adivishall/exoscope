import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from exoscope.data.synthetic import SyntheticParams, generate_synthetic  # noqa: E402
from exoscope.processing import preprocess  # noqa: E402


@pytest.fixture(scope="session")
def synth():
    return generate_synthetic(SyntheticParams())


@pytest.fixture(scope="session")
def synth_processed(synth):
    return preprocess(synth)


@pytest.fixture(autouse=True)
def _tmp_db(tmp_path, monkeypatch):
    monkeypatch.setenv("EXOSCOPE_DB_PATH", str(tmp_path / "t.db"))
