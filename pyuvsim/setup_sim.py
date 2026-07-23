import os
from pathlib import Path
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import yaml
from tqdm import tqdm

import astropy.units as u
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord, EarthLocation, AltAz


## ============================================================================================================
## 
## Functions to load, plot, and convert trajectories to catalogs.
##
## ============================================================================================================


def load_trajectory(pkl, idx):
    """Load a single trajectory from a pickled trajectory table.

    The pickle file is expected to contain a pandas object supporting
    row selection through ``.loc``. The trajectory corresponding to ``idx``
    is returned without additional processing.

    Args:
        pkl (str or os.PathLike): Path to the pickle file containing the
            trajectory table.
        idx (int or str): Row index identifying the trajectory to load.

    Returns:
        pandas.Series: Row containing the selected trajectory data.
    """
    with open(pkl, "rb") as file:
        all_trajectories = pickle.load(file)
    return all_trajectories.loc[idx]


def plot_trajectory(
    traj_ra, 
    traj_dec, 
    label, 
    date,
    duration, 
    npix=1000, 
    savepath=None,
):
    """Plot a celestial trajectory in an MWA-centered WCS projection.

    The plot is centered on the Murchison Widefield Array zenith at the
    specified time. The trajectory coordinates are converted from sky
    coordinates to WCS pixel coordinates before plotting.

    Args:
        traj_ra (astropy.units.Quantity): Right ascension
            coordinates of the trajectory.
        traj_dec (astropy.units.Quantity): Declination
            coordinates of the trajectory.
        label (str): Label used to identify the trajectory in the plot title.
        date (astropy.time.Time): Observation time used to
            calculate the MWA zenith.
        duration (float): Duration of the trajectory in seconds.
        npix (int, optional): Number of pixels along each axis of the square
            WCS projection. Defaults to 1000.
        savepath (str or os.PathLike, optional): Output path for the saved
            figure. If ``None``, the figure is not written to disk.
            Defaults to ``None``.

    Returns:
        None
    """
    zenith = _get_mwa_zenith(date)
    wcs = _init_wcs(zenith, npix)

    plt.style.use('seaborn-v0_8')
    fig, ax = plt.subplots(figsize=(8, 4),
                 subplot_kw={'projection': wcs, 'slices': ['x', 'y']},
                 )

    ra_pix, dec_pix   = _deg_to_pix(wcs, traj_ra.value, traj_dec.value)
    zra_pix, zdec_pix = _deg_to_pix(wcs, zenith.ra.deg, zenith.dec.deg)

    ax.scatter(ra_pix, dec_pix, s=1, label=f"Satellite tracks (duration {round(duration)}s)")
    ax.scatter(zra_pix, zdec_pix, marker='*', s=100, color='red', label="Zenith from MWA on start time")

    ax.set_xlim(200, npix-200)
    ax.set_ylim(200, npix-200)

    ax.set_xlabel("Right ascension")
    ax.set_ylabel("Declination")

    ax.set_title(f"Satellite {label} trajectory, start time {date}", y=1.02)
    ax.legend(loc='upper left')

    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches='tight')

    plt.clf()


def create_sim_setup_files(
    obspar_template, 
    setup_path, 
    time_array, 
    ra_array, 
    dec_array, 
    label,
    src_flux=1000, 
    src_freq=175,
):
    """Create source catalogs and pyuvsim observation parameter files.

    One catalog and one observation parameter file are generated for each
    trajectory sample. Each catalog contains a single point source located at
    the corresponding right ascension and declination. The observation
    parameter template is copied and updated with the corresponding catalog,
    output filename, and observation start time.

    The following directories are created under ``setup_path``:

    - ``_catalogs/cat_<label>/``
    - ``_obsparams/obspar_<label>/``
    - ``_runs/runs_<label>/``

    Args:
        obspar_template (dict): pyuvsim observation parameter template.
        setup_path (str or os.PathLike): Base directory containing the
            telescope configuration and array layout files.
        time_array (array-like): Observation start times expressed as Julian
            Dates. Must contain one value per trajectory sample.
        ra_array (astropy.units.Quantity): Right ascension coordinates for
            the simulated point source.
        dec_array (astropy.units.Quantity): Declination coordinates for the
            simulated point source.
        label (str): Label used to name the catalog, observation parameter,
            and simulation-output directories.
        src_flux (float, optional): Flux density assigned to each simulated
            source (units Jansky). Defaults to 1000.
        src_freq (float, optional): Reference frequency assigned to each
            source (units MHz). Defaults to 175.

    Returns:
        None
    """
    # Create necessary directories
    cat_dir = os.path.join(setup_path, f"_catalogs/cat_{label}/")
    par_dir = os.path.join(setup_path, f"_obsparams/obspar_{label}/")
    run_dir = os.path.join(setup_path, f"_runs/runs_{label}/")

    os.makedirs(cat_dir, exist_ok=True)
    os.makedirs(par_dir, exist_ok=True)
    os.makedirs(run_dir, exist_ok=True)

    for i, t in enumerate(tqdm(time_array)):

        pdx = f"{i:04d}"
        
        df = pd.DataFrame(
            columns = ["source_id", "ra_J2000 [deg]", "dec_J2000 [deg]", "Flux [Jy]", "Frequency [MHz]"],
            data    = [[f"src{pdx}", float(ra_array[i].value), float(dec_array[i].value), src_flux, src_freq]],
        )

        # Save the CATALOGS to disk
        df.to_csv(os.path.join(cat_dir, f"cat{pdx}.txt"), sep='\t', index=False)

        # Generate obsparam
        obs = copy.deepcopy(obspar_template)

        obs['filing']['outfile_name'] = os.path.join(run_dir, f"run{pdx}.uvh5")
        obs['sources']['catalog']     = os.path.join(cat_dir, f"cat{pdx}.txt")
        obs['time']['start_time']     = float(t)

        obs['telescope']['array_layout']          = os.path.join(setup_path,"layout_phase1.csv")
        obs['telescope']['telescope_config_name'] = os.path.join(setup_path,"telescope.yaml")

        # Save the OBSPARAMS to disk
        with open(os.path.join(par_dir, f"obsparam{pdx}.yaml"), 'w') as file:
            yaml.dump(obs, file, default_flow_style=False)


## ============================================================================================================
## 
## Helper functions
##
## ============================================================================================================


def _get_mwa_zenith(date):
    """Calculate the ICRS coordinates of the MWA zenith.

    Args:
        date (astropy.time.Time or datetime-like): Time at which to calculate
            the local zenith above the Murchison Widefield Array.

    Returns:
        astropy.coordinates.SkyCoord: ICRS right ascension and declination of
        the MWA zenith at ``date``.
    """
    mwa = EarthLocation.of_site('MWA')
    return SkyCoord(alt=90*u.deg, az=0*u.deg, frame=AltAz(obstime=date, location=mwa)).icrs


def _init_wcs(zenith, npix):
    """Initialize a zenith-centered two-dimensional ARC projection.

    The WCS uses an ICRS right-ascension and declination coordinate system.
    The zenith is placed at the center of a square image, with right ascension
    increasing toward the left.

    Args:
        zenith (astropy.coordinates.SkyCoord): Sky coordinate to place at the
            center of the WCS projection.
        npix (int): Number of pixels along each axis of the square image.

    Returns:
        astropy.wcs.WCS: Configured two-dimensional celestial WCS object.
    """
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ['RA---ARC', 'DEC--ARC']
    wcs.wcs.crpix = [npix/2 + 0.5, npix/2 + 0.5]     # 1-based: center of the grid
    wcs.wcs.crval = [zenith.ra.deg, zenith.dec.deg]
    wcs.wcs.cdelt = [-90.0/(npix/2), 90.0/(npix/2)]  # deg/pix; RA increases left
    wcs.wcs.cunit = ['deg', 'deg']
    wcs.wcs.radesys = 'ICRS'
    wcs.pixel_shape = (npix, npix)
    return wcs


def _deg_to_pix(wcs, xdeg, ydeg):
    """Convert celestial coordinates in degrees to WCS pixel coordinates.

    Args:
        wcs (astropy.wcs.WCS): Celestial WCS used for the coordinate
            transformation.
        xdeg (float or array-like): Right ascension coordinate or coordinates
            in degrees.
        ydeg (float or array-like): Declination coordinate or coordinates in
            degrees.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray]: Pixel coordinates along the first
        and second image axes, respectively.
    """
    coord = SkyCoord(xdeg, ydeg, unit=u.deg)
    pix_coords = wcs.wcs_world2pix(np.column_stack((coord.ra.degree, coord.dec.degree)), 0)
    return pix_coords[:,0], pix_coords[:,1]


## ============================================================================================================
## 
## Run code
##
## ============================================================================================================


if __name__ == "__main__":


    abs_path      = Path(__file__).resolve().parent.parent
    setup_path    = os.path.join(abs_path, "pyuvsim/setup_files/")
    pkl_path      = os.path.join(abs_path, "clean_night_trajectories.pkl")
    obspar_file   = os.path.join(setup_path, "sample_obsparam.yaml")

    idx         = 1        # specify which satellite to load in
    label       = "test02" # label for loaded satellite
    plot_savename = os.path.join(abs_path, f"trajectory_{label}.png")

    print(f"Loading satellite {idx}, labeled {label}, from {pkl_path}.")

    trajectory = load_trajectory(pkl_path, idx)
    traj_ra    = trajectory['RightAscensions']*u.deg
    traj_dec   = trajectory['Declinations']*u.deg

    time_res    = 0.1   # seconds; TODO: this should be read in from the pkl
    num_samples = traj_ra.size
    duration    = num_samples * time_res

    start_time  = trajectory['StartTime']
    time_array  = start_time.jd + np.arange(num_samples) * time_res / 86400.0  # seconds to days

    print("Start time:", start_time)
    print("Trajectory duration:", duration, "seconds at", time_res, "second cadence.")

    # Load observation parameter template
    with open(obspar_file, 'r') as file:
        obspar_template = yaml.safe_load(file)

    print(f"Plotting satellite tracks... (saving to: {plot_savename})")
    plot_trajectory(traj_ra, traj_dec, label, start_time, duration, savepath=plot_savename)

    print(f"Generating catalog and obsparam files... (saving to: {setup_path})")
    create_sim_setup_files(obspar_template, setup_path, time_array, traj_ra, traj_dec, label)
