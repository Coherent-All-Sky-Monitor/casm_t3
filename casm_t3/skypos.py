"""Beam index -> sky position for plot annotation.

The deployed beamforming weights file carries each synthesised beam's fixed
(alt, az) pointing. The array doesn't track, so a beam's RA/Dec depends only
on the event time: AltAz at OVRO -> ICRS via astropy.

The weights .h5 is ~400 MB and lives only on corr1, so the 512-row pointing
table is extracted into ``data/beam_pointings.json`` inside this package and
travels to corr2 with the normal repo rsync. Regenerate it whenever a new
weights file is deployed:

    python -m casm_t3.skypos /path/to/new_weights.h5
"""

from __future__ import annotations

import json
from datetime import datetime
from functools import lru_cache
from pathlib import Path

# Matches bf_weights_generator.config — the weights themselves were built
# against these coordinates, so they must agree.
OVRO_LAT_DEG = 37.2339
OVRO_LON_DEG = -118.2820
OVRO_ALT_M = 1222.0

POINTINGS_JSON = Path(__file__).parent / "data" / "beam_pointings.json"


@lru_cache(maxsize=1)
def _pointings() -> dict | None:
    try:
        return json.loads(POINTINGS_JSON.read_text())
    except (OSError, ValueError):
        return None


def beam_altaz(beam: int) -> tuple[float, float] | None:
    """(alt_deg, az_deg) of a global beam, or None if the table is missing."""
    table = _pointings()
    if table is None or not 0 <= beam < len(table["alt_deg"]):
        return None
    return table["alt_deg"][beam], table["az_deg"][beam]


def beam_radec(beam: int, utc: datetime) -> dict | None:
    """Sky position of a beam at an event time.

    Returns {"alt_deg", "az_deg", "ra_deg", "dec_deg", "ra_hms", "dec_dms"}
    or None when the pointing table or astropy is unavailable — callers
    annotate plots and must never fail on a missing position.
    """
    pos = beam_altaz(beam)
    if pos is None:
        return None
    try:
        from astropy import units as u
        from astropy.coordinates import AltAz, EarthLocation, SkyCoord
        from astropy.time import Time
        from astropy.utils import iers

        # corr2 has no internet and the bundled IERS table ends before today;
        # extrapolating it costs milliarcseconds against degree-scale beams.
        iers.conf.auto_download = False
        iers.conf.auto_max_age = None

        loc = EarthLocation(lat=OVRO_LAT_DEG * u.deg, lon=OVRO_LON_DEG * u.deg,
                            height=OVRO_ALT_M * u.m)
        coord = SkyCoord(alt=pos[0] * u.deg, az=pos[1] * u.deg,
                         frame=AltAz(obstime=Time(utc), location=loc)).icrs
        return {
            "alt_deg": pos[0], "az_deg": pos[1],
            "ra_deg": float(coord.ra.deg), "dec_deg": float(coord.dec.deg),
            "ra_hms": coord.ra.to_string(unit=u.hourangle, sep="hms",
                                         precision=1, pad=True),
            "dec_dms": coord.dec.to_string(unit=u.deg, sep="dms",
                                           precision=0, alwayssign=True),
        }
    except Exception:  # noqa: BLE001 - annotation only, never plot-fatal
        return None


def extract_pointings(weights_h5: str | Path, out_json: Path = POINTINGS_JSON) -> Path:
    """Pull pointings/{alt,az}_deg out of a weights file into the sidecar."""
    import h5py

    with h5py.File(weights_h5, "r") as f:
        table = {
            "source_file": str(weights_h5),
            "created_utc": f.attrs.get("created_utc", ""),
            "alt_deg": [round(float(v), 4) for v in f["pointings/alt_deg"][:]],
            "az_deg": [round(float(v), 4) for v in f["pointings/az_deg"][:]],
        }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(table))
    return out_json


if __name__ == "__main__":
    import sys

    path = extract_pointings(sys.argv[1])
    print(f"wrote {path} from {sys.argv[1]}")
