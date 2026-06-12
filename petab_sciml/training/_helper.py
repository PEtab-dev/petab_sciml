"""Shared function for training strategies"""

from pathlib import Path


def _resolve_output_dir(
    yaml: Path | str,
    output_dir: Path | str | None,
    default_name: str,
) -> Path:
    """Resolve and prepare the output directory for an export.

    If ``output_dir`` is ``None``, defaults to a subdirectory of the source
    YAML's directory named ``default_name``. Raises if the resolved directory
    matches the source YAML's directory, since exporting in place would
    overwrite the source. Creates the directory if it does not exist.
    """
    yaml_dir = Path(yaml).parent
    if output_dir is None:
        output_dir = yaml_dir / default_name
    output_dir = Path(output_dir)

    if output_dir.resolve() == yaml_dir.resolve():
        raise ValueError(
            f"output_dir ({output_dir}) must differ from the source problem's "
            f"directory ({yaml_dir}); exporting in place would overwrite the source."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
