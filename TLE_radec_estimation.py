import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import EarthLocation, get_sun, AltAz
from tqdm import tqdm
 
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
    dt = (t).to(u.s).value          # seconds since epoch, shape (N,)
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

    obs_pos, _ = location.get_gcrs_posvel(t + t0)                 # observer in same frame
    r_obs = obs_pos.xyz.to(u.km).value.T                     # (N,3)

    rho = r_sat - r_obs                                      # topocentric vector
    x, y, z = rho[:, 0], rho[:, 1], rho[:, 2]
    ra   = np.mod(np.arctan2(y, x), 2*np.pi)
    dec  = np.arctan2(z, np.hypot(x, y))
    
    dist = np.linalg.norm(rho, axis=1)
    up = r_obs / np.linalg.norm(r_obs, axis=1, keepdims=True)
    alt = np.degrees(np.arcsin(np.sum(rho * up, axis=1) / dist))
    
    return np.degrees(ra), np.degrees(dec), alt

# Precompute the MWA zenith direction over one sidereal day (orbit-independent).
day  = Time("2025-07-01T00:00:00") + np.arange(0, 86164, 60.0)*u.s
temp_zen, _ = mwa.get_gcrs_posvel(day)
zen_dir   = temp_zen.xyz.to(u.km).value.T
zen_dir  /= np.linalg.norm(zen_dir, axis=1, keepdims=True)      # (Ng, 3) unit zenith vectors
tsec = (day - day[0]).to(u.s).value

def target_zenith(a, e, i, raan, argp):
    """Return (t0, M0) that puts this orbit at the MWA zenith mid-window,
    or None if the orbit's plane never reaches the zenith declination."""
    cO, sO = np.cos(raan), np.sin(raan)
    ci, si = np.cos(i),    np.sin(i)
    asc_dir = np.array([cO, sO, 0.0])                       # node direction
    mot_dir = np.array([-sO*ci, cO*ci, si])                 # direction of orbital motion
    orb_norm_dir = np.cross(asc_dir, mot_dir)               # orbit normal

    f = zen_dir @ orb_norm_dir                              # zenith is in-plane where f = 0
    s = np.signbit(f)                                       # flag negative f
    zenith_crossings = np.where(s[:-1] != s[1:])[0]
    if len(zenith_crossings) == 0:
        return None                                         # inclination too low -> no pass
    cr = zenith_crossings[0]
    tc = tsec[cr] + 30.0
    zen_cr = (zen_dir[cr] + zen_dir[cr+1])/2

    u_lat = np.arctan2(zen_cr @ mot_dir, zen_cr @ asc_dir)  # argument of latitude at zenith
    nu = u_lat - argp                                       # true anomaly there
    E  = 2*np.arctan2(np.sqrt(1-e)*np.sin(nu/2), np.sqrt(1+e)*np.cos(nu/2))
    Mc = E - e*np.sin(E)                                    # mean anomaly at crossing
    n  = np.sqrt(MU / a**3)
    t0 = tc - 60.0 + rng.uniform(-500, 500)                 # approx. zenith crossing time
    M0 = Mc - n*60.0
    return t0, M0
    
satlist = pd.read_csv('active_satellites.csv')
n_rad_s = satlist["MEAN_MOTION"] * 2*np.pi / 86400.0   # revolutions/day -> radians/sec
satlist["SEMIMAJOR_AXIS"] = (MU / n_rad_s**2)**(1/3)                # radians/sec -> semi-major axis (km)
deg_cols = ["INCLINATION", "RA_OF_ASC_NODE", "ARG_OF_PERICENTER", "MEAN_ANOMALY"]
satlist[deg_cols] = np.deg2rad(satlist[deg_cols])
kepler_params = satlist[["SEMIMAJOR_AXIS", "ECCENTRICITY", "INCLINATION",
            "RA_OF_ASC_NODE", "ARG_OF_PERICENTER", "MEAN_ANOMALY"]]

a, e, i, raan, argp, M0 = kepler_params.values.T
rng = np.random.default_rng(42)
t0 = np.zeros_like(M0)
for k in range(len(a)):
    result = target_zenith(a[k], e[k], i[k], raan[k], argp[k])
    if result is None:
        t0[k], M0[k] = np.nan, np.nan
    else:
        t0[k], M0[k] = result
        
SIDEREAL_DAY = 86164.0905                              # s, Earth rotation wrt inertial space
REF_EPOCH    = Time("2025-07-01T00:00:00", scale="utc")  # epoch of day/tsec/zen_dir grid

# spread each pass across 2025 by a WHOLE number of sidereal days
rng_days    = np.random.default_rng(43)
day_offsets = rng_days.integers(-181, 182, size=len(a))  # ~Jan 1 .. Dec 28, 2025
t0 = t0 + day_offsets * SIDEREAL_DAY

t0 *= u.s
t0 = REF_EPOCH + t0
t  = np.arange(0, 120, 0.1)*u.s            # 2-min window, one ra/dec point every 0.1 seconds

ra  = []
dec = []
alt = []

for j in tqdm(range(len(a))):
    if np.isnan(M0[j]):
        ra_j, dec_j, alt_j = np.nan, np.nan, np.nan
    else:
        ra_j, dec_j, alt_j = satellite_radec(a[j], e[j], i[j], raan[j], argp[j], M0[j], t, t0[j], mwa)
    ra.append(ra_j)
    dec.append(dec_j)
    alt.append(alt_j)
    
df = kepler_params[["SEMIMAJOR_AXIS", "ECCENTRICITY", "INCLINATION", "RA_OF_ASC_NODE", "ARG_OF_PERICENTER"]]
df["MEAN_ANOMALY"] = M0
df["StartTime"] = t0
df["RightAscensions"] = ra
df["Declinations"] = dec

clean_df = df.dropna()

times = Time(clean_df["StartTime"].tolist())   # reconstruct a Time array from the column
sun_alt = get_sun(times).transform_to(AltAz(obstime=times, location=mwa)).alt.to_value(u.deg)

is_night = sun_alt < -10.0                        # Sun well below the horizon
night_df = clean_df[is_night].reset_index(drop=True)

clean_df.to_pickle("clean_trajectories.pkl")
night_df.to_pickle("clean_night_trajectories.pkl")

clean_ra, clean_dec, clean_alt = [], [], []

for j in tqdm(range(len(a))):
    if np.isnan(M0[j]):        # invalid satellite -> skip
        continue
    clean_ra.append(ra[j])
    clean_dec.append(dec[j])
    clean_alt.append(alt[j])

fig, ax = plt.subplots(figsize=(6, 6))
sc = ax.scatter(clean_ra[0:15000], clean_dec[0:15000], c=clean_alt[0:15000],
                cmap="viridis", s=10, zorder=3, label="Simulated samples")
ax.set_xlabel("Right Ascension (deg)")
ax.set_ylabel("Declination (deg)")
ax.set_title("Simulated satellite trajectory in RA/Dec")
ax.grid(color="gray", alpha=0.3, linestyle="solid")
cbar = fig.colorbar(sc, ax=ax)
cbar.set_label("Altitude (deg. above/below horizon)")
ax.legend()
plt.tight_layout()
plt.savefig("simulated_trajectories.png")