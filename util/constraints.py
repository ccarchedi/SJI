""" Constraints on the inversion.

This includes observed phase velocities and constraints pulled from receiver
function observations.
"""
import re
import os

import warnings
import numpy as np
import pandas as pd
import xarray as xr # for loading netcdf
import math
from scipy.interpolate import interp1d
import netCDF4 as nc

warnings.simplefilter(action='ignore',category=FutureWarning)

# =============================================================================
# Set up classes for commonly used variables
# =============================================================================

# =============================================================================
#       Extract the observations of interest for a given location
# =============================================================================

def extract_observations(location:tuple, id:str, boundaries:tuple, vpvs:float,
        randomize:bool=False, dohw:bool=False, bmult:float=1):
    """ Make an Observations object. #jsb added randomize, dohw, bmult, smooth_Sp
    """

    surface_waves = _extract_phase_vels(location)
    rfs = _extract_rf_constraints(location, id, boundaries, vpvs, bmult)

    if rfs is None:
        print('No observation sufficiently close to ' + str(location[0]) + ',' + str(location[1]))
        return

    # Put in fake Moho info!!!! #
    # rfs.loc[0, ('dv', 'dvstd')] = [0.085, 0.1]

    # Write to file
    if not os.path.exists('output/' + id):
        os.mkdir('output/' + id)
    savedir = 'output/{0}/{0}_'.format(id)
    surface_waves.to_csv('{}_surface_wave_constraints.csv'.format(savedir, id))
    rfs.to_csv('{}_RF_constraints.csv'.format(savedir, id))

    ph_vel  = surface_waves['ph_vel'].to_numpy()
    std     = surface_waves['std'].to_numpy() #hardwrited by jsb if 0.385 is there (not atm)
    periods = surface_waves['period'].to_numpy()

    tt         = rfs['tt'].to_numpy()
    dv         = rfs['dv'].to_numpy()
    ttstd      = rfs['ttstd'].to_numpy()
    dvstd      = rfs['dvstd'].to_numpy()
    breadth    = rfs['breadth'].to_numpy()

    if not dohw:

        #jsb d changed to dvec
        #dvec = np.vstack((surface_waves['ph_vel'][:, np.newaxis],
        #               rfs['tt'][:, np.newaxis],
        #               rfs['dv'][:, np.newaxis]))
        #std = np.vstack((surface_waves['std'][:, np.newaxis],
        #               rfs['ttstd'][:, np.newaxis],
        #               rfs['dvstd'][:, np.newaxis]))

        dvec = np.vstack((ph_vel[:, np.newaxis],
                       tt[:, np.newaxis],
                       dv[:, np.newaxis]))
        std = np.vstack((std[:, np.newaxis],
                       ttstd[:, np.newaxis],
                       dvstd[:, np.newaxis]))
        #periods = surface_waves['period'].values

    elif dohw:

        hw = _extract_hw_constraints(location, vpvs) #added by jsb
        #hw.to_csv('{}_HW_constraints.csv'.format(savedir, id)) #added by jsb
        #vsn = hw['Vp']/vpvs
        vsn = hw/vpvs

        if np.isnan(vsn):
            vsn = 4.5
            vsnstd = 1e9#won't effect inv this way
        else:
            vsnstd = 0.1#its a guess

        dvec = np.vstack((ph_vel[:, np.newaxis],
                       tt[:, np.newaxis],
                       dv[:, np.newaxis],
                       vsn))

        std = np.vstack((std[:, np.newaxis],
                       ttstd[:, np.newaxis],
                       dvstd[:, np.newaxis],
                       vsnstd))

    #jsb added randomize
    if randomize:
        for k in np.arange(0,len(dvec)):
            dvec[k] = dvec[k] + np.random.normal(loc=0, scale=std[k])

    return dvec, std, periods, len(rfs), breadth

def _extract_rf_constraints(location:tuple, id:str, boundaries:pd.DataFrame,
                            vpvs:float, bmult:float=1):
    pd.options.mode.chained_assignment = None
    """
    Note that some of the reported standard deviation on values are
    unrealistically low (i.e. zero), so we will assume the minimum standard
    deviation on a value is the 10% quantile of the total data set.
    """
    bnames, bwidths, smoothing = boundaries

    #jsb many edits to include breadth and smoothing

    # Load in receiver function constraints
    #all_rfs = pd.read_csv('data/RFconstraints/a_priori_constraints.csv')
    #all_rfs = pd.read_csv('data/RFconstraints/a_priori_constraints_breadth.csv')
    #all_rfs = pd.read_csv('data/RFconstraints/a_priori_constraints_sed.csv')
    # all_rfs = pd.read_csv('data/RFconstraints/a_priori_constraints_Continent.csv')
    all_rfs = pd.read_csv('data/RFconstraints/a_priori_constraints_BIMA.csv')

    ind,flag = _find_closest_lat_lon(all_rfs.copy(), location, 0.25)

    obs = all_rfs.loc[ind]

    rfs = pd.DataFrame(
        columns = ['lat', 'lon', 'tt', 'ttstd', 'dv', 'dvstd', 'breadth']
    )

    ib = 0
    for bound,bw,radius in zip(bnames, bwidths, smoothing):

        if radius > 0:
            #first, do the smoothing if necessary
            #most code copied and pasted from _find_closest_lat_lon, terrible programing form by jsb
            df = all_rfs.copy()
            lat, lon = location
            # Make sure all longitudes are in range -180 to 180
            if lon > 180:
                lon -= 360
            df.loc[df['lon'] > 180, 'lon'] -= 360

            df['distance_squared'] = (df['lon'] - lon)**2 + (df['lat'] - lat)**2
            #half normal, though I doubt the constant terms matter here
            weights = math.sqrt(2)*np.exp( -(df['distance_squared'])**2/(2*radius**2))/(math.sqrt(math.pi)*radius)

            if np.sum(weights)<0.1:
                return

            obs['tt' + bound]      = np.sum(all_rfs['tt' + bound].to_numpy()*weights)/(np.sum(weights))
            obs['breadth' + bound] = np.sum(all_rfs['breadth' + bound].to_numpy()*weights)/(np.sum(weights))

            #not exactly correct but oh well
            obs['tt' + bound + 'std']     = np.sum(all_rfs['tt' + bound + 'std'].to_numpy()*weights)/(np.sum(weights))

            try:
                obs['dv' + bound]         = np.sum(all_rfs['dv' + bound].to_numpy()*weights)/(np.sum(weights))
                obs['dv' + bound + 'std'] = np.sum(all_rfs['dv' + bound + 'std'].to_numpy()*weights)/(np.sum(weights))
            except:
                obs['amp' + bound]         = np.sum(all_rfs['amp' + bound].to_numpy()*weights)/(np.sum(weights))
                obs['amp' + bound + 'std'] = np.sum(all_rfs['amp' + bound + 'std'].to_numpy()*weights)/(np.sum(weights))

        # Extract travel time information
        try:
            tt = obs['tt' + bound]
            ttstd = obs['tt' + bound + 'std']

            min_allowed_ttstd = all_rfs['tt' + bound + 'std'].quantile(0.1)
            ttstd = max((ttstd, min_allowed_ttstd))

            breadth = obs['breadth' + bound]

            #now test if you need to scale or change it
            breadth = max( (breadth*bmult, bw) )

        except:
            print('No RF constraints on travel time for ' + bound)
            return

        # If necessary, scale to Vs travel time
        try:
            rftype = obs['type' + bound]
        except:
            rftype = 'Ps'
            print('RF type unspecified for ' + bound + ' - assuming Ps')
        if rftype == 'Sp': # Scale travel time from travelling at Vp to at Vs
            tt *= vpvs
            # for constant a, variable A: sigma_aA = |a| * sigma_A
            ttstd *= vpvs

        # Extract velocity contrast information
        try:
            dv = obs['dv' + bound]
            dvstd = obs['dv' + bound + 'std']

            min_allowed_dvstd = all_rfs['dv' + bound + 'std'].quantile(0.1)
            #ampstd = max((dvstd, min_allowed_dvstd))
            dvstd = max((dvstd, min_allowed_dvstd))
        except:
            try:
                amp = obs['amp' + bound]
                ampstd = obs['amp' + bound + 'std']

                min_allowed_ampstd = all_rfs['amp' + bound + 'std'].quantile(0.1)

                dv, dvstd = _convert_amplitude_to_dv(
                    amp, ampstd, rftype, breadth
                )

                _, dvstdmin = _convert_amplitude_to_dv(
                    0.1, min_allowed_ampstd, rftype, breadth
                )#note dummy value jsb

                dvstd = max((ampstd, dvstdmin))

            except:
                print('No RF constraints on dV for ' + bound)
                return rfs

        if (rftype == 'Sp') & (flag==1): #some points don't have Sp phases
            print('No RF constraints on dV for Sp')
            return rfs

        if (rftype == 'Ps'): #some points don't have Sp phases
            #dvstd = dvstd/2#See SR16 for why you should use factor of four. 2 for test
            dvstd = dvstd/4#See SR16 for why you should use factor of four.
            ttstd = ttstd/4

        lat, lon = location

        if np.isnan(ind):
            tt = 0

        rfs = rfs.append(
            pd.Series([lat, lon, tt, ttstd, dv, dvstd, breadth], index=rfs.columns),
            ignore_index=True,
        )

        ib += 1

    return rfs

def _convert_amplitude_to_dv(amp, ampstd, rftype, boundary_width):
    """

    Calculated the Sp synthetics in MATLAB for a variety of dV (where
    dV = (1 - v_bottom / v_top) * 100 given the width of the boundary
    layer (here labelled 'breadth').  These are saved as individual .csv
    files, where the breadth of the synthetic layer is in the file name,
    and each row represents a different input dV and output phase amplitude.

    synthvals = pd.DataFrame(columns = ['breadth', 'dv', 'amplitude'])
    for b in range(0, 51, 5):
        a = pd.read_csv('synthvals_' + str(b) + '.0.csv', header=None,
                        names = ['dv', 'amplitude'])
        a['breadth'] = b
        a['dv'] *= -1 / 100
        synthvals = synthvals.append(a, ignore_index=True, sort=False)
    synthvals.to_csv('data/RFconstraints/synthvals_Sp.csv', index=False)
    """

    try:
        synth = pd.read_csv('data/RFconstraints/synthvals_' + rftype +'.csv')
    except:
        print('No synthetic amplitudes calculated for ' + rftype + '!')
        return

    x = np.unique(synth.breadth.to_numpy())
    boundary_width_rounded = x[np.where(min(abs(x - boundary_width)) == abs(x - boundary_width))[0].item()]

    synth = synth[synth.breadth == boundary_width_rounded]
    # Note that numpy default for interpolation is x < x[0] returns y[0]
    #dv = np.round(np.interp(amp, synth.amplitude, synth.dv), 3)
    #dvstd = abs(np.round(np.interp(ampstd, synth.amplitude, synth.dv), 3))
    f = interp1d(synth.amplitude, synth.dv, kind='linear', fill_value="extrapolate")
    dv = np.round(f(amp), 3)
    dvstd = abs(np.round(f(ampstd), 3))

    return dv, dvstd

def _extract_phase_vels(location:tuple, phv_preloaded:tuple=(0,)):

    if phv_preloaded[0]:
        phv = phv_preloaded[1]
    else:
        #print('Loading phase velocities') jsb
        phv = _load_observed_sw_constraints()

        #remove NaN, jsb
        x   = np.where(np.isnan(phv.values))
        phv = phv.drop(x[0]); #jsb works, idk how to use python elegantly

    surface_waves = pd.DataFrame()
    for period in phv['period'].unique():
        ind,flag = _find_closest_lat_lon(
            phv[phv['period'] == period].copy(), location)
        surface_waves = surface_waves.append(phv.loc[ind])
    surface_waves = (surface_waves.sort_values(by=['period'], ascending=True)
        .reset_index(drop=True))
    # Should actually load in some std!!!!  Going to do a random estimate
    for index, row in surface_waves.iterrows():
        # Conservative estimate: below 50s, assume +- 0.025
        # Else, assume that you will be able to see a 1% difference in phase
        # over the 140 km (2 station spacings) distance travelled at about 4 km/s
        surface_waves.loc[index, 'std'] = (
            max(0.025, abs(140 / (140 / 4 + row.period / 100) - 4) / 2)
        )
        if row.period <=8:
            surface_waves.loc[index, 'std'] *= 2
        #     phv[phv.period == row.period].ph_vel.std() / 2

    return surface_waves.reset_index(drop=True)

def _load_observed_sw_constraints():
    """ Load surface wave constraints into pandas.

    Very specific to the way the data is currently stored!  See READMEs in the
    relevent directories.
    """

    # Load in surface waves
    data_dir = 'data/obs_dispersion/'
    phvel = pd.DataFrame()
    # Have ASWMS data with filenames 'helmholtz_stack_LHZ_[period].xyz',
    # longer period surface wave data with filenames 'c_[period]s_BD19',
    # and ambient noise data with filenames 'R[period]_USANT15.txt'
    for file in sorted(os.listdir(data_dir), reverse=True, key=str.lower):
        if 'USANT15' in file: # Ambient noise
            phvel = phvel.append(_load_ambient_noise(data_dir, file))
        if 'helmholtz' in file: # ASWMS data
            phvel = phvel.append(_load_earthquake_sw(data_dir, file))
        if 'eikonal' in file: # ASWMS data. Same format
            phvel = phvel.append(_load_earthquake_sw(data_dir, file))
        if 'ambientnoise' in file: # data formatted like the ASWMS data
            phvel = phvel.append(_load_earthquake_sw(data_dir, file))
        if 'BD19' in file: # Longer T surface waves: Babikoff & Dalton, 2019
            c = _load_earthquake_sw_BD19(data_dir, file)
            if c.period[0] not in phvel.period.unique():
                phvel = phvel.append(c)
    phvel = phvel.sort_values(by=['period', 'lat', 'lon']).reset_index(drop=True)

    return phvel

def _load_ambient_noise(data_dir:str, file:str):
    """
    """
    # Find period of observations
    # Filenames are R[period]_USANT15.pix, so split on R or _
    period = float(re.split('R|_', file)[1])

    # Find reference phase velocity
    with open(data_dir + file, 'r') as fid:
        for line in fid:
            if 'PVELREF' in line:
                break
    ref_vel = float(line.split()[1])

    # Load file
    # data structure: 1. geocentric latitude, longitude, pixel size, deviation
    ambient_noise = pd.read_csv(data_dir + file, header=None,
        skiprows=11, sep='\s+')
    ambient_noise.columns = ['geocentric_lat', 'lon', 'size', 'dV']
    ambient_noise['ph_vel'] = (1 + ambient_noise['dV'] / 100) * ref_vel
    # convert to geodetic latitude,
    #       tan(geocentric_lat) = (1 - f)**2 * tan(geodesic_lat)
    # https://en.wikipedia.org/wiki/Latitude#Geocentric_latitude
    WGS84_f = 1 / 298.257223563  # flattening for WGS84 ellipsoid
    ambient_noise['lat'] = np.degrees(np.arctan(
        np.tan(np.radians(ambient_noise['geocentric_lat'])) / (1 - WGS84_f)**2
    ))
    ambient_noise['period'] = period

    return ambient_noise[['period', 'lat', 'lon', 'ph_vel']]

def _load_earthquake_sw(data_dir:str, file:str):
    """
    """
    # Find period of observations
    # Filenames are R[period]_USANT15.pix, so split on R or _
    # period = float(re.split('_|\.', file)[-2])
    pp = (re.split("_",file)[-1]) #assumes format ends with PERIOD.PERIOD.xyz
    period = float(pp[:-4])

    surface_waves = pd.read_csv(data_dir + file, sep='\s+', header=None)
    surface_waves.columns = ['lat', 'lon', 'ph_vel']
    surface_waves['period'] = period

    return surface_waves[['period', 'lat', 'lon', 'ph_vel']]

def _load_earthquake_sw_BD19(data_dir:str, file:str):
    """
    """
    # Find period of observations
    # Filenames are c_[period]s_BD19, so split on _ and s
    period = float(re.split('_|s', file)[1])

    surface_waves = pd.read_csv(data_dir + file, sep='\s+', header=None)
    surface_waves.columns = ['lat', 'lon', 'ph_vel']
    surface_waves['period'] = period

    # Phase velocities are reported in m/s - convert to km/s
    surface_waves['ph_vel'] /= 1000

    return surface_waves[['period', 'lat', 'lon', 'ph_vel']]

def _find_closest_lat_lon(df:pd.DataFrame, location:tuple, mindistance:int=1e20):
    """ Find index in dataframe of closest point to lat, lon.

    Assumes that the dataframe has (at least) two columns, Lat and Lon, which
    are in °N and °E respectively.

    """
    flag = 0
    lat, lon = location
    # Make sure all longitudes are in range -180 to 180
    if lon > 180:
        lon -= 360
    df.loc[df['lon'] > 180, 'lon'] -= 360

    df['distance_squared'] = (df['lon'] - lon)**2 + (df['lat'] - lat)**2
    min_ind = df['distance_squared'].idxmin()

    if df.loc[min_ind, 'distance_squared'] > mindistance:
        print('!!!!!! Closest observation at {}°N, {}°E !!!!!!'.format(
            df.loc[min_ind, 'lat'], df.loc[min_ind, 'lon']))
        #min_ind = np.nan #jsb
        flag = 1

    return min_ind,flag

def get_vels_Crust1(location):
    """
    Crust 1.0 is given in a 1 degree x 1 degree grid (i.e. 360 lon points, 180
    lat points).  The downloads are structured as
        crust1.bnds  (360 * 180) x 9 depths to top of each layer
                             0. water (i.e. topography)
                             1. ice (i.e. bathymetry)
                             2. upper sediments (i.e. depth to rock)
                             3. middle sediments
                             4. lower sediments
                             5. upper crust (i.e. depth to bedrock)
                             6. middle crust
                             7. lower crust
                             8. mantle (i.e. Moho depth)
    Note that for places where a layer doesn't exist, the difference between
    bnds[i, n] and bnds[i, n+1] = 0; i.e. for continents, the top of the ice is
    the same as the top of the water; where there are no glaciers, the top of
    sediments is the same as the top of the ice, etc.

        crust1.[rho|vp|vs]  (360 * 180) x 9 values of density, Vp, Vs for each
                            of the layers specified in bnds

    Each row in these datasets steps first in longitude (from -179.5 to +179.5)
    then in latitude (from 89.5 to -89.5).
        i.e. index of (lat, lon) will be at (lon + 179.5) + (89.5 - lat) * 360


    """
    lat, lon = location

    all_lons = np.arange(-179.5,180,1)
    all_lats = np.arange(89.5,-90,-1)
    i = int((lon - all_lons[0]) + ((all_lats[0] - lat) // 1) * len(all_lons))

    nm = 'data/earth_models/crust1/crust1.'
    try:
        cb = pd.read_csv(nm + 'bnds', skiprows=i, nrows=1, header=None, sep='\s+'
            ).values.flatten()
        vs = pd.read_csv(nm + 'vs', skiprows=i, nrows=1, header=None, sep='\s+'
            ).values.flatten()
    except:
        crust1url = 'http://igppweb.ucsd.edu/~gabi/crust1/crust1.0.tar.gz'
        print(
            'You need to download (and extract) the Crust1.0 model'
            + ' from \n\t{} \nand save to \n\t{}'.format(crust1url, nm[:-7])
        )
        return
    # vp = pd.read_csv(nm + 'vp', skiprows=i, nrows=1, header=None, sep='\s+'
    #     ).values.flatten()
    # rho = pd.read_csv(nm + 'rho', skiprows=i, nrows=1, header=None, sep='\s+'
    #     ).values.flatten()

    thickness = -np.diff(cb)
    ib = 0
    m_t = [0]
    m_vs = []
    for t in thickness:
        if t > 0:
            m_vs += [vs[ib]]
            m_t += [t]
        ib += 1
    m_vs += [vs[ib]]#[m_vs[-1]]

    return m_t, m_vs


def get_vels_ShenRitzwoller2016(location):
    """
    To get a more accurate starting model for the crust in the US, load in
    data from Shen & Ritzwoller, 2016.  Note that this is the updated model
    of Shen et al., 2013 - the model used (at time of writing) for Moho depth
    and velocity contrast constraints.  (Note we used an older model for that
    as the download included explicit Moho depth, dVs and error information).

    Download is available from https://doi.org/10.17611/DP/EMCUS2016

    The downloaded file - 'US.2016.nc' - is a NetCDF file with coordinates
        'depth'         0 - 150 km, spacing 0.5 km
        'latitude'      20 - 50 °N, spacing 0.25°
        'longitude'     235 - 295 °E, spacing 0.25°
    and fields
        'vsv'           (depth, latitude, longitude)
        'vp'            (depth, latitude, longitude)
        'rho'           (depth, latitude, longitude)

    """
    lat, lon = location
    if lon < 0: # Convert to °E if in °W
        lon += 360

    ds = load_literature_vel_model('SR16')
    if not ds:
        return

    i_lat = np.argmin(np.abs(ds.latitude.values - lat))
    i_lon = np.argmin(np.abs(ds.longitude.values - lon))

    thickness = np.hstack((np.array([0]), np.diff(ds.depth.values)))
    vsv = ds.vsv.values[:, i_lat, i_lon]

    return list(thickness), list(vsv)

def load_literature_vel_model(ref:str):

    if ref == 'SR16':
        nm = 'data/earth_models/US.2016.nc'
        url = 'https://doi.org/10.17611/DP/EMCUS2016'
        ref = 'Shen & Ritzwoller, 2016 (10.1002/2016JB012887)'
        v_field = 'vsv'
        ref_v = np.array([])
    elif ref == 'S15':
        nm = 'data/earth_models/US-CrustVs-2015_kmps.nc'
        url = 'https://doi.org/10.17611/DP/EMCUSCRUSTVS2015'
        ref = 'Schmandt et al., 2015 (10.1002/2015GL066593)'
        v_field = 'vs'
        ref_v = np.array([])
    elif ref == 'P14':
        nm = 'data/earth_models/DNA13_percent.nc'
        url = 'https://doi.org/10.17611/DP/9991615'
        ref = 'Porrit et al., 2014 (10.1016/j.epsl.2013.10.034)'
        v_field = 'vsvj'
        ref_v = np.array([ # WUS model, Pollitz (2008) Fig. 17: 10.1029/2007JB005556
            [2.40, 0], [2.45, 0.5], [3.18, 4.5], [3.20, 18], [3.90, 20],
            [3.90, 33], [4.30, 35], [4.30, 60], [4.26, 65], [4.26, 215],
            [4.65, 220], [4.65, 240], [4.74, 242], [4.74, 300]
            ])
    elif ref == 'P15':
        nm = 'data/earth_models/US-Crust-Upper-mantle-Vs.Porter.Liu.Holt.2015_kmps.nc'
        url = 'https://doi.org/10.17611/DP/EMCUCUMVPLH15MLROWCU'
        ref = 'Porter et al., 2015 (10.1002/2015GL066950)'
        v_field = 'vs'
        ref_v = np.array([])
    elif ref == 'F18':
        nm = 'data/earth_models/csem-north-america-2019.12.01.nc'
        url = 'https://doi.org/10.17611/dp/emccsemnamatl20191201'
        ref = 'Fichtner et al., 2018 (10.1029/2018GL077338)'
        v_field = 'vsv'
        ref_v = np.array([])
    elif ref == 'Y14':
        nm = 'data/earth_models/SEMum-NA14_kmps.nc'
        url = 'https://doi.org/10.17611/dp/EMCSEMUMNA14'
        ref = 'Yuan et al., 2014 (10.1016/j.epsl.2013.11.057)'
        v_field = 'Vs'
        ref_v = np.array([])
    elif ref == 'C15':
        nm = 'data/earth_models/WUS-CAMH-2015.nc'
        url = 'https://doi.org/10.17611/dp/EMCWUSCAMH2015'
        ref = 'Chai et al., 2015 (10.1002/2015GL063733)'
        v_field = 'vs'
        ref_v = np.array([])

    else:
        print('Unknown reference - try again')
        return

    try:
        ds = xr.open_dataset(nm)
    except:

        print(
            'You need to download the {} model from'.format(ref)
            + ' IRIS EMC\n\t{} \nand save to \n\t{}'.format(url, nm)
        )
        return

    if ref_v.any(): # Convert perturbations to absolute values
        imax = np.argmax(ds.depth.values > ref_v[-1, 1])
        ref_vz = np.interp(ds.depth.values[:imax], ref_v[:, 1], ref_v[:, 0])
        dvs = ds[v_field].values[:imax, :, :]
        vs = np.zeros_like(dvs)
        for ila in range(dvs.shape[1]):
            for ilo in range(dvs.shape[2]):
                vs[:, ila, ilo] = (1 + (dvs[:, ila, ilo] / 100)) * ref_vz

        ds = xr.Dataset(
            {'vs': (['depth', 'latitude', 'longitude'],  vs)},
            coords={'longitude': (['longitude'], ds.longitude.values),
                    'latitude': (['latitude'], ds.latitude.values),
                    'depth': ds.depth.values[:imax],
                    }
        )
        v_field = 'vs'

    if v_field != 'vs':
        ds['vs'] = ds[v_field]

    return ds

def interpolate_lit_model(ref, z, lats, lons):
    # Put the literature model on the same grid as the inversion model
    ds = load_literature_vel_model(ref)
    z_a = ds.depth.values
    lats_a = ds.latitude.values
    lons_a = ds.longitude.values
    if lons_a[0] > 0:
        lons_a -= 360

    vs_a = np.zeros((len(lats), len(lons), len(z)))
    ila = 0
    ilo = 0
    for lat in lats:
        for lon in lons:

            i_lat = np.argmin(np.abs(lats_a - lat))
            i_lon = np.argmin(np.abs(lons_a - lon))
            if abs(lats_a[i_lat] - lat) > 1:
                print('Nearest latitude is {}'.format(lats_a[i_lat]))
            if abs(lons_a[i_lon] - lon) > 1:
                print('Nearest longitude is {}'.format(lons_a[i_lon]))

            vs_a[ila, ilo, :] = np.interp(z, z_a, ds.vs.values[:, i_lat, i_lon])

            ilo += 1
        ila += 1
        ilo = 0
    # print('Model {} depth limits: {:.0f}-{:.0f} km'.format(ref, z_a[0], z_a[-1]))

    return vs_a

def _extract_hw_constraints(location:tuple, vpvs:float): #added by jsb

    #pn = pd.read_csv('data/PnUS_2016.txt', sep=',', header=None)
    #pn.columns = ['lon', 'lat', 'Vp']

    ds = nc.Dataset('./data/PnUS_2016.nc')

    lt = ds['latitude'][:]
    ln = ds['longitude'][:]

    LN,LT = np.meshgrid(ln.data, lt.data)

    d2 = (LT - location[0])**2 + (LN - location[1])**2
    #ind = np.unravel_index(d2.argmin(), d2.shape)

    weights = math.sqrt(2)*np.exp( -(d2)**2/(2*0.5**2))/(math.sqrt(math.pi)*0.5)
    #ind, _ = _find_closest_lat_lon(pn, location)
    vp      = ds['Vp'][:].data
    #anismag = ds['AnisMag'][:].data

    val = np.nansum(weights*vp)/np.sum(weights)

    #radial anisotropy correction. Not a high confidence correction but might do the job.
    return math.sqrt(2.5 - 1.5*1.04)*val#math.sqrt(2.5 - 1.5*1.05)*vp[ind[0], ind[1]]#vp[ind[0], ind[1]]*(1 - anismag[ind[0], ind[1]]/200)
    #return val
