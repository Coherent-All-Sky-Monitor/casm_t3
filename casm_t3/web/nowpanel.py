"""Live "now at OVRO" snapshot for the stats page.

Computed server-side with astropy (IERS auto-download off, so it works on
the offline corr nodes); the page's JS ticks the clocks between refreshes,
advancing LST at the sidereal rate from the anchor computed here. For a
drift-scan array the meridian RA (= LST) and hours-to-transit of the bright
calibrators are the numbers an observer actually wants.
"""

from __future__ import annotations

from datetime import datetime, timezone

from ..skypos import OVRO_ALT_M, OVRO_LAT_DEG, OVRO_LON_DEG

SIDEREAL_RATE = 1.00273790935  # sidereal / solar time

# J2000 (ra_deg, dec_deg); None means "ask astropy" (the Sun moves).
SOURCES = [
    ("Sun", None),
    ("Cas A", (350.850, 58.815)),
    ("Cyg A", (299.868, 40.734)),
    ("Tau A (Crab)", (83.633, 22.015)),
    ("B0329+54", (53.247, 54.579)),
]


def _transit_str(ha_h: float) -> str:
    """Hour angle (sidereal hours past transit) -> friendly phrase."""
    solar = abs(ha_h) / SIDEREAL_RATE
    if solar < 0.05:
        return "on meridian"
    return f"in {solar:.1f} h" if ha_h < 0 else f"{solar:.1f} h ago"


def snapshot() -> dict | None:
    """Clocks + where the bright things are; None if astropy is unhappy."""
    try:
        from astropy import units as u
        from astropy.coordinates import AltAz, EarthLocation, SkyCoord, get_sun
        from astropy.time import Time
        from astropy.utils import iers

        iers.conf.auto_download = False
        iers.conf.auto_max_age = None

        now = datetime.now(timezone.utc)
        t = Time(now)
        loc = EarthLocation(lat=OVRO_LAT_DEG * u.deg, lon=OVRO_LON_DEG * u.deg,
                            height=OVRO_ALT_M * u.m)
        lst_h = float(t.sidereal_time("apparent",
                                      longitude=OVRO_LON_DEG * u.deg).hourangle)
        frame = AltAz(obstime=t, location=loc)
        rows = []
        for name, radec in SOURCES:
            c = (get_sun(t) if radec is None
                 else SkyCoord(ra=radec[0] * u.deg, dec=radec[1] * u.deg))
            aa = c.transform_to(frame)
            ha = (lst_h - float(c.ra.hourangle) + 12) % 24 - 12
            rows.append(dict(name=name, alt=float(aa.alt.deg),
                             az=float(aa.az.deg), dec=float(c.dec.deg),
                             transit=_transit_str(ha)))
        return dict(epoch_ms=int(now.timestamp() * 1000), lst_h=lst_h,
                    sources=rows)
    except Exception:  # noqa: BLE001 - the page must render without it
        return None
