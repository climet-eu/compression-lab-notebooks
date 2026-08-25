import argparse
import json
import re
from pathlib import Path
from typing import List, Optional

try:
    import gribscan
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency: gribscan. Install it into the Python environment used to "
        "run this script with:\n\n"
        "    python -m pip install gribscan\n\n"
    ) from exc


DEFAULT_REMOTE_BASE_URL = (
    "https://object-store.os-api.cci1.ecmwf.int/esiwacebucket/ERA5_07_2026"
)


class Era5Magician(gribscan.magician.IFSMagician):
    def m2dataset(self, meta):
        attrs = meta["attrs"]
        npoints = meta.get("array", {}).get("shape", ["unknown"])[0]
        return (
            f"{attrs.get('gridType', 'unknown')}_"
            f"{attrs.get('typeOfLevel', 'unknown')}_"
            f"{npoints}"
        )

    def variable_hook(self, key, info):
        var = super().variable_hook(key, info)
        var["attrs"] = dict(**var.get("attrs", {}), **info.get("extra", {}))
        return var


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def _decode_json(value):
    return json.loads(value) if isinstance(value, str) else value


def _ref_store(ref: dict) -> dict:
    return ref.get("refs", ref)


def _variables_in_ref(ref: dict) -> List[str]:
    metadata = _decode_json(_ref_store(ref).get(".zmetadata", {})).get("metadata", {})
    variables = []

    for key, value in metadata.items():
        if not key.endswith("/.zattrs"):
            continue

        name = key[: -len("/.zattrs")]
        attrs = _decode_json(value)
        dims = attrs.get("_ARRAY_DIMENSIONS", [])
        if dims != [name]:
            variables.append(name)

    return sorted(variables)


def _replace_reference_path(ref: dict, grib_path: Path, remote_url: Optional[str]) -> dict:
    if remote_url is None:
        return ref

    return json.loads(
        json.dumps(ref).replace(json.dumps(str(grib_path)), json.dumps(remote_url))
    )


def generate_ref(
    grib_path: Path,
    output_path: Optional[Path] = None,
    remote_url: Optional[str] = None,
    group: Optional[str] = None,
) -> List[Path]:
    if output_path is None:
        output_path = Path(f"{grib_path}.ref")

    index_path = output_path.with_suffix(output_path.suffix + ".index")
    written = []

    try:
        gribscan.write_index(str(grib_path), index_path, force=True)
        refs_by_group = gribscan.grib_magic([index_path], Era5Magician())

        if group is not None:
            if group not in refs_by_group:
                raise ValueError(
                    f"Unknown group {group!r}. Available groups: {sorted(refs_by_group)}"
                )
            refs_by_group = {group: refs_by_group[group]}

        if len(refs_by_group) == 1:
            group_name, ref = next(iter(refs_by_group.items()))
            ref = _replace_reference_path(ref, grib_path, remote_url)
            output_path.write_text(json.dumps(ref, indent=4))
            written.append(output_path)

            catalog_path = output_path.with_suffix(".refs.json")
            catalog = {
                "source": str(remote_url or grib_path),
                "groups": {
                    group_name: {
                        "ref": output_path.name,
                        "variables": _variables_in_ref(ref),
                    }
                },
            }
            catalog_path.write_text(json.dumps(catalog, indent=4))
            written.append(catalog_path)
            return written

        stem = output_path.name[: -len(".ref")] if output_path.name.endswith(".ref") else output_path.stem
        catalog = {"source": str(remote_url or grib_path), "groups": {}}

        for group_name, ref in sorted(refs_by_group.items()):
            ref = _replace_reference_path(ref, grib_path, remote_url)
            group_path = output_path.with_name(f"{stem}.{_safe_name(group_name)}.ref")
            group_path.write_text(json.dumps(ref, indent=4))
            written.append(group_path)
            catalog["groups"][group_name] = {
                "ref": group_path.name,
                "variables": _variables_in_ref(ref),
            }

        catalog_path = output_path.with_suffix(".refs.json")
        catalog_path.write_text(json.dumps(catalog, indent=4))
        written.append(catalog_path)
        return written
    finally:
        index_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Generate gribscan .ref metadata for a GRIB file.")
    parser.add_argument("grib_path", type=Path, help="Path to the local GRIB file.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output .ref path. Defaults to <grib_path>.ref.",
    )
    parser.add_argument(
        "--remote-url",
        help="URL to store in the references instead of the local GRIB path.",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_REMOTE_BASE_URL,
        help=(
            "Base URL used to infer --remote-url from the GRIB filename. "
            f"Defaults to {DEFAULT_REMOTE_BASE_URL}."
        ),
    )
    parser.add_argument(
        "--local-paths",
        action="store_true",
        help="Keep references pointing to the local GRIB path instead of a URL.",
    )
    parser.add_argument(
        "--group",
        help="Write only one dataset group. Without this, all groups are written.",
    )
    args = parser.parse_args()

    if args.local_paths:
        remote_url = None
    else:
        remote_url = args.remote_url or f"{args.base_url.rstrip('/')}/{args.grib_path.name}"

    output_path = args.output or Path(f"{args.grib_path}.ref")
    for path in generate_ref(args.grib_path, output_path, remote_url, args.group):
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
