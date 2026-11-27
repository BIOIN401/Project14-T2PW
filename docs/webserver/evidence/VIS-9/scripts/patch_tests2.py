from pathlib import Path
p = Path("C:/t/vis-pw9/tests/test_pwml_vis9_pwc_id.py"); s = p.read_text(encoding="utf-8")
def rep(old, new):
    global s
    assert s.count(old) == 1, old[:70]
    s = s.replace(old, new)
rep('import pytest\n', 'import json\n\nimport pytest\n')
rep('from t2pw.pwml import writer\n', 'from t2pw.pwml import refresh_pwc_id_skip, writer\n')
rep('("pantoate", "(R)-pantoate"), ("β-alanine", "beta-Alanine"), ("L-serine", "L-Serine"),\n])',
    '("pantoate", "(R)-pantoate"), ("β-alanine", "beta-Alanine"), ("L-serine", "L-Serine"), ("limonene", "(+)-Limonene"),\n])')
rep('("MK", "Melleolide K"), ("noreugenin", "Isobiflorin"),\n])',
    '("MK", "Melleolide K"), ("noreugenin", "Isobiflorin"),\n    ("(+)-limonene", "(-)-Limonene"),\n])')
rep('''    (40974, "4'-phosphopantetheine"), (2580, "Geraniol"), (442, "Cinnamic acid"),''',
    '''    (40974, "4'-phosphopantetheine"), (2580, "Geraniol"), (442, "Cinnamic acid"), (41655, "Ferric enterobactin"),''')
rep('''def test_missing_skip_list_derives_nothing(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(writer, "_PWC_ID_SKIP_PATH", tmp_path / "absent.json")
    writer._pwc_id_skip.cache_clear()
    try:
        assert _pwc([_legacy("chorismate", 7748, "Chorismate")]) == [None]
    finally:
        writer._pwc_id_skip.cache_clear()
''', '''@pytest.mark.parametrize("content", [None, '{"derivation_mismatch": [], "structure_errors": {}}', "not json"])
def test_missing_or_empty_skip_list_derives_nothing(monkeypatch, tmp_path, caplog, content) -> None:
    path = tmp_path / "pwc_id_skip.json"
    if content is not None:
        path.write_text(content, encoding="utf-8")
    monkeypatch.setattr(writer, "_PWC_ID_SKIP_PATH", path)
    writer._pwc_id_skip.cache_clear()
    try:
        with caplog.at_level("WARNING", logger=writer.__name__):
            assert _pwc([_legacy("chorismate", 7748, "Chorismate")]) == [None]
            assert _pwc([_legacy("chorismate", 7748, "Chorismate")]) == [None]
        assert sum("no pwc-id is derived" in r.getMessage() for r in caplog.records) == 1
    finally:
        writer._pwc_id_skip.cache_clear()


class _FakeDb:
    def __init__(self, rows, error=""):
        self.rows, self.error, self.last_error = rows, error, ""

    def available(self):
        return True

    def _query(self, sql, params):
        self.last_error = self.error
        return [] if self.error else list(self.rows)


ROWS = [{"id": 40738, "name": "x", "pwc_id": "PW_C040736"}, {"id": 42951, "name": "y", "pwc_id": "PW_C042948"}]


def test_refresh_writes_the_rows_and_the_structure_errors(tmp_path) -> None:
    out = tmp_path / "skip.json"
    refresh_pwc_id_skip.refresh(out, db=_FakeDb(ROWS))
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["derivation_mismatch"] == [40738, 42951] and "41655" in data["structure_errors"]


def test_refresh_refuses_a_failed_query_without_naming_the_host(tmp_path) -> None:
    out = tmp_path / "skip.json"
    with pytest.raises(RuntimeError) as err:
        refresh_pwc_id_skip.refresh(out, db=_FakeDb(ROWS, error="db_query_failed:(2003, \\"Can't connect to db.example.org\\")"))
    assert "db_query_failed" in str(err.value) and "example.org" not in str(err.value) and not out.exists()


def test_refresh_refuses_no_rows_and_a_shrunken_list(tmp_path) -> None:
    out = tmp_path / "skip.json"
    with pytest.raises(RuntimeError):
        refresh_pwc_id_skip.refresh(out, db=_FakeDb([]))
    assert not out.exists()
    out.write_text(json.dumps({"derivation_mismatch": list(range(1, 101))}), encoding="utf-8")
    with pytest.raises(RuntimeError):
        refresh_pwc_id_skip.refresh(out, db=_FakeDb(ROWS))
    assert len(json.loads(out.read_text(encoding="utf-8"))["derivation_mismatch"]) == 100
    refresh_pwc_id_skip.refresh(out, db=_FakeDb(ROWS), allow_shrink=True)
    assert json.loads(out.read_text(encoding="utf-8"))["derivation_mismatch"] == [40738, 42951]
''')
p.write_text(s, encoding="utf-8"); print("ok")
