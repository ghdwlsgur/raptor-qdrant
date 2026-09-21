import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

EXCLUDED_DIRS = frozenset({".obsidian", ".trash", ".git", "node_modules"})

FRONTMATTER_RE = re.compile(r"\A---\r?\n(.*?)\r?\n---\r?\n?", re.DOTALL)
EMBED_RE = re.compile(r"!\[\[[^\]]*\]\]")
WIKILINK_RE = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]*)?(?:\|([^\]]*))?\]\]")
INLINE_TAG_RE = re.compile(r"(?:(?<=\s)|\A)#([\w가-힣][\w가-힣/_-]*)")


@dataclass(frozen=True)
class VaultNote:
    path: str
    title: str
    text: str
    content_hash: str
    tags: tuple[str, ...] = ()
    links: tuple[str, ...] = ()
    frontmatter: dict = field(default_factory=dict, compare=False)

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


def _parse_frontmatter(raw: str) -> tuple[dict, str]:
    match = FRONTMATTER_RE.match(raw)
    if not match:
        return {}, raw

    try:
        parsed = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError:
        return {}, raw

    if not isinstance(parsed, dict):
        return {}, raw

    return parsed, raw[match.end() :]


def _as_tag_tuple(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return tuple(t.strip() for t in value.split(",") if t.strip())
    if isinstance(value, list):
        return tuple(str(t).strip() for t in value if str(t).strip())
    return ()


def _flatten_wikilinks(body: str) -> tuple[str, tuple[str, ...]]:
    targets: list[str] = []

    def replace(match: re.Match[str]) -> str:
        target = match.group(1).strip()
        display = (match.group(2) or "").strip()
        if target:
            targets.append(target)
        return display or target

    return WIKILINK_RE.sub(replace, body), tuple(dict.fromkeys(targets))


class VaultLoader:
    """옵시디언 볼트를 읽어 임베딩 가능한 노트 목록으로 바꾼다."""

    def __init__(
        self,
        vault_path: str | Path,
        excluded_dirs: frozenset[str] = EXCLUDED_DIRS,
    ):
        self.vault_path = Path(vault_path).expanduser().resolve()
        if not self.vault_path.is_dir():
            raise ValueError(f"vault not found: {self.vault_path}")
        self.excluded_dirs = excluded_dirs

    def note_paths(self) -> list[Path]:
        return sorted(
            path
            for path in self.vault_path.rglob("*.md")
            if not self.excluded_dirs.intersection(path.parts)
        )

    def load(self) -> list[VaultNote]:
        notes = []
        skipped = 0

        for path in self.note_paths():
            note = self.load_note(path)
            if note.is_empty:
                skipped += 1
                continue
            notes.append(note)

        logger.info(
            f"loaded {len(notes)} notes from {self.vault_path}"
            + (f", skipped {skipped} empty" if skipped else "")
        )
        return notes

    def load_note(self, path: Path) -> VaultNote:
        raw = path.read_text(encoding="utf-8", errors="replace")
        frontmatter, body = _parse_frontmatter(raw)

        body = EMBED_RE.sub("", body)
        body, links = _flatten_wikilinks(body)

        tags = _as_tag_tuple(frontmatter.get("tags")) + tuple(
            INLINE_TAG_RE.findall(body)
        )

        return VaultNote(
            path=path.relative_to(self.vault_path).as_posix(),
            title=path.stem,
            text=body.strip(),
            content_hash=hashlib.sha256(raw.encode("utf-8")).hexdigest(),
            tags=tuple(dict.fromkeys(tags)),
            links=links,
            frontmatter=frontmatter,
        )
