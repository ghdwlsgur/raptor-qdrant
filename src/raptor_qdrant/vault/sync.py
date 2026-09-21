import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from raptor_qdrant.vault.loader import VaultNote

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VaultDiff:
    added: tuple[VaultNote, ...] = ()
    changed: tuple[VaultNote, ...] = ()
    removed: tuple[str, ...] = ()
    unchanged: tuple[VaultNote, ...] = field(default=(), repr=False)

    @property
    def has_changes(self) -> bool:
        return bool(self.added or self.changed or self.removed)

    @property
    def summary(self) -> str:
        return (
            f"added {len(self.added)}, changed {len(self.changed)}, "
            f"removed {len(self.removed)}, unchanged {len(self.unchanged)}"
        )


def diff_vault(
    notes: Iterable[VaultNote], indexed_hashes: Mapping[str, str]
) -> VaultDiff:
    """볼트의 현재 상태와 이미 인덱싱된 해시를 비교한다."""
    added, changed, unchanged = [], [], []
    seen = set()

    for note in notes:
        seen.add(note.path)
        indexed = indexed_hashes.get(note.path)
        if indexed is None:
            added.append(note)
        elif indexed != note.content_hash:
            changed.append(note)
        else:
            unchanged.append(note)

    removed = tuple(sorted(set(indexed_hashes) - seen))

    return VaultDiff(
        added=tuple(added),
        changed=tuple(changed),
        removed=removed,
        unchanged=tuple(unchanged),
    )
