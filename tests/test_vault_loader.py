import pytest

from raptor_qdrant.vault import VaultLoader, diff_vault
from raptor_qdrant.vault.loader import VaultNote


@pytest.fixture
def vault(tmp_path):
    (tmp_path / "메모").mkdir()
    (tmp_path / ".obsidian").mkdir()
    (tmp_path / ".obsidian" / "config.md").write_text("무시", encoding="utf-8")
    return tmp_path


def write(vault, name: str, text: str):
    path = vault / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_strips_frontmatter_from_the_body(vault):
    write(vault, "a.md", "---\ntags: [운영, 쿠버네티스]\n---\n본문이다.")

    note = VaultLoader(vault).load()[0]

    assert note.text == "본문이다."
    assert note.tags == ("운영", "쿠버네티스")


def test_keeps_body_when_frontmatter_is_malformed(vault):
    write(vault, "a.md", "---\n: : :\n---\n본문이다.")

    assert "본문이다." in VaultLoader(vault).load()[0].text


def test_flattens_wikilinks_and_records_targets(vault):
    write(vault, "a.md", "[[대상 노트]] 와 [[다른 것|표시 이름]] 을 본다.")

    note = VaultLoader(vault).load()[0]

    assert note.text == "대상 노트 와 표시 이름 을 본다."
    assert note.links == ("대상 노트", "다른 것")


def test_strips_heading_anchors_from_links(vault):
    write(vault, "a.md", "[[노트#섹션]] 참고. 충분한 길이의 본문.")

    note = VaultLoader(vault).load()[0]

    assert note.links == ("노트",)
    assert "#섹션" not in note.text


def test_drops_embeds(vault):
    write(vault, "a.md", "앞 ![[그림.png]] 뒤. 충분한 길이의 본문이다.")

    assert "그림.png" not in VaultLoader(vault).load()[0].text


def test_collects_inline_tags(vault):
    write(vault, "a.md", "본문에 #운영 과 #k8s/네트워크 태그가 있다.")

    assert VaultLoader(vault).load()[0].tags == ("운영", "k8s/네트워크")


def test_skips_excluded_dirs_and_empty_notes(vault):
    write(vault, "메모/b.md", "내용 있음")
    write(vault, "empty.md", "---\ntags: [x]\n---\n   ")

    paths = {n.path for n in VaultLoader(vault).load()}

    assert paths == {"메모/b.md"}


def test_path_is_vault_relative_posix(vault):
    write(vault, "메모/정리/c.md", "내용 있음")

    assert VaultLoader(vault).load()[0].path == "메모/정리/c.md"


def test_hash_tracks_raw_file_content(vault):
    path = write(vault, "a.md", "처음 내용")
    first = VaultLoader(vault).load()[0].content_hash

    path.write_text("바뀐 내용", encoding="utf-8")

    assert VaultLoader(vault).load()[0].content_hash != first


def test_missing_vault_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="vault not found"):
        VaultLoader(tmp_path / "없음")


def note(path: str, digest: str) -> VaultNote:
    return VaultNote(path=path, title=path, text="본문", content_hash=digest)


def test_diff_classifies_every_note():
    current = [note("a.md", "h1"), note("b.md", "new"), note("c.md", "h3")]
    indexed = {"a.md": "h1", "b.md": "old", "d.md": "h4"}

    diff = diff_vault(current, indexed)

    assert [n.path for n in diff.added] == ["c.md"]
    assert [n.path for n in diff.changed] == ["b.md"]
    assert diff.removed == ("d.md",)
    assert [n.path for n in diff.unchanged] == ["a.md"]
    assert diff.has_changes


def test_diff_reports_no_changes_when_hashes_match():
    diff = diff_vault([note("a.md", "h1")], {"a.md": "h1"})

    assert not diff.has_changes
