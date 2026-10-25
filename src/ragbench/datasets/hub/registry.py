"""What the hub knows: one `HubDataset` per public benchmark, with where its files are, how to check them and under what terms they come."""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field

from ragbench.errors import UsageError


class PullError(UsageError):
    """The dataset cannot be pulled as asked."""


@dataclass(frozen=True)
class HubFile:
    name: str  # the name it is cached under and handed to the reader by
    url: str
    size: int  # bytes; checked after the download as well as shown before it
    sha256: str | None = None
    git_sha1: str | None = None  # a git blob id, for files whose host publishes only that (small, non-LFS files on Hugging Face)

    def __post_init__(self) -> None:
        if (self.sha256 is None) == (self.git_sha1 is None):
            raise ValueError(f"Hub file {self.name} needs exactly one of sha256 and git_sha1.")


@dataclass(frozen=True)
class HubDataset:
    name: str
    title: str
    description: str
    license: str
    license_url: str
    source_url: str
    citation: str
    revision: str  # the commit (or tag) the files are pinned to
    files: tuple[HubFile, ...]
    reader: str  # key of `readers.READERS`
    shared_corpus: bool  # one corpus serves every question (False: each question brings its own passages)
    options: dict[str, str] = field(default_factory=dict)  # handed to the reader (e.g. which RAGBench subset)
    notes: str = ""  # terms or caveats worth reading before using the results

    @property
    def total_bytes(self) -> int:
        return sum(item.size for item in self.files)

    @property
    def config_name(self) -> str:
        return self.name.replace("-", "_")


HUB: dict[str, HubDataset] = {}


def register(entry: HubDataset) -> HubDataset:
    HUB[entry.name] = entry
    return entry


def get_dataset(name: str) -> HubDataset:
    import ragbench.datasets.hub.catalog  # noqa: F401  (registers the built-in datasets)

    if name in HUB:
        return HUB[name]
    close = difflib.get_close_matches(name, list(HUB), n=3, cutoff=0.5)
    hint = f" Did you mean {' or '.join(repr(item) for item in close)}?" if close else ""
    raise PullError(f"Unknown dataset {name!r}.{hint} `ragbench datasets list` shows what is available.")
