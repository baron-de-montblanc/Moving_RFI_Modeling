import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import EarthLocation
 
plt.style.use("seaborn-v0_8")


MU = 398600.4418  # km^3/s^2, G*M_earth

# MWA array centre. In practice pull this from your uvfits
# (UVData.telescope_location_lat_lon_alt) to stay consistent with your sim.
mwa = EarthLocation(lat=-26.701335*u.deg, lon=116.670963*u.deg, height=370.0272*u.m)
-26.701335, 116.670963

def solve_kepler(M, e, tol=1e-12, maxiter=50):
    """Solve M = E - e*sin(E) for eccentric anomaly E (radians)."""
    M = np.mod(M, 2*np.pi)
    E = M if e < 0.8 else np.full_like(M, np.pi)   # initial guess
    for _ in range(maxiter):
        dE = (E - e*np.sin(E) - M) / (1 - e*np.cos(E))
        E = E - dE
        if np.all(np.abs(dE) < tol):
            break
    return E

def kepler_to_gcrs(a, e, i, raan, argp, M0, t, t0):
    """
    Two-body propagation of classical elements to geocentric inertial xyz.
    a [km]; e [-]; i, raan, argp, M0 [rad]; t, t0 astropy Time. Returns (N,3) km.
    """
    dt = (t - t0).to(u.s).value          # seconds since epoch, shape (N,)
    n  = np.sqrt(MU / a**3)              # mean motion [rad/s]
    M  = M0 + n*dt
    E  = solve_kepler(M, e)

    nu = 2*np.arctan2(np.sqrt(1+e)*np.sin(E/2), np.sqrt(1-e)*np.cos(E/2))
    r  = a*(1 - e*np.cos(E))

    r_pqw = np.stack([r*np.cos(nu), r*np.sin(nu), np.zeros_like(r)], axis=-1)

    # perifocal -> inertial:  Rz(raan) Rx(i) Rz(argp)  (constant in time)
    cO, sO = np.cos(raan), np.sin(raan)
    ci, si = np.cos(i),    np.sin(i)
    cw, sw = np.cos(argp), np.sin(argp)
    R = np.array([
        [cO*cw - sO*sw*ci, -cO*sw - sO*cw*ci,  sO*si],
        [sO*cw + cO*sw*ci, -sO*sw + cO*cw*ci, -cO*si],
        [sw*si,             cw*si,             ci   ],
    ])
    return r_pqw @ R.T                    # (N,3) km, GCRS-aligned inertial

def satellite_radec(a, e, i, raan, argp, M0, t, t0, location):
    """Topocentric RA/Dec [deg] and range [km] of the satellite from `location`."""
    r_sat = kepler_to_gcrs(a, e, i, raan, argp, M0, t, t0)   # (N,3)

    obs_pos, _ = location.get_gcrs_posvel(t)                 # observer in same frame
    r_obs = obs_pos.xyz.to(u.km).value.T                     # (N,3)

    rho = r_sat - r_obs                                      # topocentric vector
    x, y, z = rho[:, 0], rho[:, 1], rho[:, 2]
    ra   = np.mod(np.arctan2(y, x), 2*np.pi)
    dec  = np.arctan2(z, np.hypot(x, y))
    
    dist = np.linalg.norm(rho, axis=1)
    up = r_obs / np.linalg.norm(r_obs, axis=1, keepdims=True)
    alt = np.degrees(np.arcsin(np.sum(rho * up, axis=1) / dist))
    
    return np.degrees(ra), np.degrees(dec), alt
    
# example orbit: ~550 km, 53° inclination, near-circular
a    = 6921.0               # km  (≈ 6371 + 550)
e    = 0.001
i    = np.radians(-83.0)
raan = np.radians(200.0)
argp = np.radians(20000.0)
M0   = np.radians(10.0)

t0 = Time("2025-01-01T12:00:00", scale="utc")   # epoch of your elements
t  = t0 + np.arange(0, 120, 0.1)*u.s            # 2-min window, 1 s cadence
times_sec = (t - t0).to(u.s).value

ra, dec, alt = satellite_radec(a, e, i, raan, argp, M0, t, t0, mwa)

fig, ax = plt.subplots(figsize=(6, 6))
sc = ax.scatter(ra, dec, c=alt,
                cmap="viridis", s=25, zorder=3, label="Simulated samples")
ax.set_xlabel("Right Ascension (deg)")
ax.set_ylabel("Declination (deg)")
ax.set_title("Simulated satellite trajectory in RA/Dec")
ax.grid(color="gray", alpha=0.3, linestyle="solid")
cbar = fig.colorbar(sc, ax=ax)
cbar.set_label("Altitude (deg. above/below horizon)")
ax.legend()
plt.tight_layout()
plt.show()