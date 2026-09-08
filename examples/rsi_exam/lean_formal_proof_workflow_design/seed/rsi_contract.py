"""Pure artifact contract helpers shared by the importer and replay adapter."""

import stat
from pathlib import Path, PurePosixPath


def artifact_sources(config: dict) -> list[str]:
    """Keep Harbor's destinations intact; the workspace mirrors sources below /app."""
    sources = []
    for artifact in config.get("artifacts", []):
        source = artifact if isinstance(artifact, str) else artifact["source"]
        if isinstance(artifact, dict) and artifact.get("service") not in (None, "main"):
            raise ValueError("Sidecar artifacts are not supported by this adapter")
        path = PurePosixPath(source)
        if ".." in path.parts or not path.is_relative_to("/app") or path == PurePosixPath("/app"):
            raise ValueError(f"Expected an artifact below /app, got {source!r}")
        sources.append(str(path))
    if not sources:
        raise ValueError("The task must declare at least one submission artifact")
    for i, source in enumerate(sources):
        if any(PurePosixPath(source).is_relative_to(p) for p in sources[:i]):
            raise ValueError("Overlapping artifact sources are unsupported")
        if any(PurePosixPath(p).is_relative_to(source) for p in sources[:i]):
            raise ValueError("Overlapping artifact sources are unsupported")
    return sources


def submission_path(root: Path, source: str) -> Path:
    # Revalidate even when called directly by a Harbor agent constructor.
    artifact_sources({"artifacts": [source]})
    relative = PurePosixPath(source).relative_to("/app")
    path = root.joinpath(*relative.parts)
    for parent in [path, *path.parents]:
        if parent == root:
            break
        if parent.is_symlink():
            raise ValueError(f"Submission symlinks are not allowed: {relative}")
    if path.exists():
        for entry in [path, *path.rglob("*")] if path.is_dir() else [path]:
            mode = entry.lstat().st_mode
            if not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
                raise ValueError(f"Submission must contain only regular files: {relative}")
    return path
