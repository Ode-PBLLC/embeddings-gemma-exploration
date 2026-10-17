import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures" / "ingest"


@pytest.fixture
def fixture_json():
    return lambda name: json.loads((FIXTURES / name).read_text(encoding="utf-8"))


@pytest.fixture
def fixture_text():
    return lambda name: (FIXTURES / name).read_text(encoding="utf-8")
