# %%
import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import calendar
import cartopy.crs as ccrs
import rioxarray
import pandas as pd
import regionmask
import geopandas as gpd

from const import *
from mypath import *


def clip_region_mask(ds, region_name="SEA"):
    # print(region_name)

    region_mask = regionmask.defined_regions.srex[[region_name]]
    gdf = gpd.GeoDataFrame([1], geometry=region_mask.polygons, crs=WORLD_SHP.crs)

    return ds.rio.clip(gdf.geometry, crs=gdf.crs)


def get_model_name(path):
    """Extract the model name from a nc path
    Param:
        Path of the nc file
    Return:
        Name of the model used to generate the nc file
    """
    return (
        path.split("\\")[-1].split("AERmon")[-1].split("historical")[0].replace("_", "")
    )


def get_list_model_by_name(list_path):
    """Extract the list of the nc files which generated by the equivalent model
    Param:
        List of the nc files
    Return:
        Dictionary with keys are the model names and values are the list of nc files that are generated by the equivalent key model
    """
    list_model = list(set([get_model_name(p) for p in list_path]))

    model_dict = {}
    for model in list_model:
        model_dict[model] = []
        for p in list_path:
            if model in p:
                model_dict[model].append(p)
    return model_dict


def merge_by_model(list_path):
    """Merge all the nc files generated by single model to one ds
    Param:
        list of nc files
    Return:
        Merged ds - xarray
    """
    list_ds = []
    for p in list_path:
        list_ds.append(xr.open_dataset(p))

    return xr.concat(list_ds, dim=DIM_TIME)


def merge_all_model(list_path):
    """Merge all the nc files generated by all the models to xarray ds
    Param:
        List of all the nc files
    Return:
        Dictionary of all the models. Keys are the model names, values are the equivalent xarray ds
    """

    l_model_dict = get_list_model_by_name(list_path)
    l_model_name = l_model_dict.keys()
    merged_dict = {}

    for model_name in l_model_name:
        merged_dict[model_name] = merge_by_model(l_model_dict[model_name])

    return merged_dict


def get_month_days(ds):
    """Calculate the list of months for 360-day calendar and noleap calendar
    Param:
        Single xarray ds
    Return:
        Equivalent month array with the length of xarray ds
    """
    n_month = 12

    day_360 = [30] * n_month
    no_leap = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    l = int(len(ds.time) / n_month)

    calendar = ds[DIM_TIME].dt.calendar

    if calendar == "360_day":
        month_arr = np.array(day_360 * l)
    month_arr = np.array(no_leap * l)
    month_days = month_arr

    return month_days


def get_equiv_area(model_name):
    """Get the equivalent area xarray ds
    Param:
        Model name
    Return:
        xarray ds of the area
    """
    for f in AREA_LIST:
        if model_name in f:
            return xr.open_dataset(f)


def get_equiv_lf(model_name):
    """Get the equivalent area xarray ds
    Param:
        Model name
    Return:
        xarray ds of the area
    """
    for f in LF_LIST:
        if model_name in f:
            return xr.open_dataset(f)


def cal_monthly_emi(all_ds_dict, var=VAR_ISOP):
    """Calculate monthly emission for the equivalent variable
    Param:
        Dictionary of all the xarray ds
        Varibale name
    Return:
        Monthly emi for the var

    """
    l_m_name = list(all_ds_dict.keys())

    month_emi = {}
    for m_name in l_m_name:
        print(m_name)
        model = all_ds_dict[m_name]
        month_days = get_month_days(model)
        ds_area = get_equiv_area(m_name)
        all_ds_dict[m_name] = (
            ds_area[VAR_AREA] * TG_RATE * ISOP_2_C * model[var] * month_days
        ).sum(dim=[DIM_LAT, DIM_LON])

    return month_emi


def cal_annual_emi(ds):
    """Calculate the annual emisison from monthly emisison xarray ds
    Param:
        Annual array ds
    Return:
        Time array ds
    """
    time = list(set(t.year for t in ds[DIM_TIME].values))

    month = 12
    l = int(len(ds.month_rate.values) / 12)
    annual_arr = []
    for i in range(0, l):
        annual_arr.append(np.sum(ds.month_rate.values[month * i : month * (i + 1)]))

    return annual_arr, time


def plot_annual_ds(dict_ds):
    """Plot the annual emission ds of the all models
    Param:
        Dictionary with keys are the model names, values are the equiv xarray ds
    Return:
        Nothing
    """
    model_name = dict_ds.keys()

    fig, ax = plt.subplots(figsize=(6, 2.7), layout="constrained")
    axbox = ax.get_position()
    for name in model_name:
        y, x = cal_annual_emi(dict_ds[name])

        ax.plot(x, y, label=name)
    ax.set_xlabel("Year")
    ax.set_ylabel("BVOC rate")
    ax.set_title(f"BVOC trends")
    ax.legend(
        loc="center",
        ncol=3,
        bbox_to_anchor=[axbox.x0 + 0.5 * axbox.width, axbox.y0 - 0.6],
    )


def cal_season_emi(ds, months):
    """Calculate the seasonal emission
    Param:
        Xarray ds
    Return:
        Xarray ds for seasonal emission for each single year
        List of years
    """
    year = list(set(t.year for t in ds[DIM_TIME].values))

    ds_month_dict = {}
    for m in months:
        ds_month = ds.where(ds.time.dt.month == m)
        ds_month_dict[calendar.month_name[m]] = ds_month[ds_month > 0].values

    return ds_month_dict, year


def plot_season_single_model(org_dict_ds, model_name, months=[1, 4, 7, 10]):
    """Plot seasonal emission trends for single model
    Param:
        Annual emission dictionary
        Model name
    Return:
        Nothing
    """

    ss_ds, year = cal_season_emi(org_dict_ds, months)
    months = ss_ds.keys()
    fig, ax = plt.subplots(figsize=(6, 2.7), layout="constrained")
    axbox = ax.get_position()

    for m in months:
        x = ss_ds[m]
        ax.plot(year, x, label=m)

    ax.set_xlabel("Year")
    ax.set_ylabel("BVOC rate")  # Add a y-label to the axes.
    ax.set_title(f"BVOC seasonal trends by {model_name}")
    ax.legend(
        loc="center",
        ncol=4,
        bbox_to_anchor=[axbox.x0 + 0.5 * axbox.width, axbox.y0 - 0.6],
    )


def cal_change_his2pre(ds, var=VAR_ISOP):
    """Calculate the change between the history and present
    Param:
        Input xarray ds
        Variable to calculate the change
    Return:
        Xarray ds with added variable for the change
    """

    ds = ds.assign(change_his2pre=ds.isel(time=-1)[var] - ds.isel(time=0)[var])
    return ds


def plot_map(ds_dict):
    """Plot the map from the dictionary of xarray ds
    Param:
        Dictionary with keys are the model names, values are the xarray ds
    Return:
        Nothing
    """
    l_m_name = ds_dict.keys()

    for i, m_name in enumerate(l_m_name):
        fig = plt.figure(1 + i, figsize=(30, 13))
        # ds = cal_change_his2pre(ds_dict[m_name])
        ds = ds_dict[m_name]
        ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())
        ax.coastlines()
        # ds.change_his2pre.plot.pcolormesh(ax=ax, cmap="coolwarm")
        ds.isel(time=0).plot.pcolormesh(ax=ax, cmap="coolwarm")
        plt.title(m_name, fontsize=18)


def clip_to_roi(ds, var=VAR_ISOP):
    """Clip the xarray ds to ROIgit sta
    Param:
        Original xarray ds
        Boundary dictionary of the corners
    Return:
        clipped xarray ds
    """

    boundary_dict = {"min_lon": 50, "max_lon": 150, "min_lat": -10, "max_lat": 20}
    # ds[var] = ds[var].rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
    ds[var] = ds[var].rio.write_crs("epsg:4326", inplace=True)
    subset = ds[var].rio.clip_box(
        minx=boundary_dict["min_lon"],
        miny=boundary_dict["min_lat"],
        maxx=boundary_dict["max_lon"],
        maxy=boundary_dict["max_lat"],
        crs="EPSG:4326",
    )
    # subset.isel(time=0).plot()
    return subset


def interpolate(ds, lat, lon):
    interpolated_ds = ds.interp(lat=lat, lon=lon)

    return interpolated_ds


def y2p(p):
    s, e = p
    ys = 1850

    m_s = (s - ys) * 12
    m_e = (e - ys + 1) * 12 - 1
    return m_s, m_e


def txt_2_nc(txt_file):
    # convert from txt file to netCDF4 at original resolution

    txt_file = np.loadtxt(
        "/mnt/dg3/ngoc/data_other/IGBP-SurfaceProducts_569/data/IGBP_wp.dat",
        delimiter=" ",
    )
    wiltpoint = []
    wiltpoint[:] = txt_file[:]
    latitudes = np.arange(-56.5, 84.0, 1 / 12)
    latitudes = latitudes[::-1]
    longitudes = np.arange(-180.0, 180.0, 1 / 12)
    ds = xr.Dataset(
        data_vars=dict(
            wiltpoint=(["lat", "lon"], wiltpoint),
        ),
        coords=dict(
            lon=(longitudes),
            lat=(latitudes),
        ),
        attrs=dict(description="Global IGBP wilting point at 0.0833333 degree"),
    )
    ds = ds.where(ds["wiltpoint"] != -2)
    ds = ds.fillna(-9999.0)
    org_nc_file = ds.to_netcdf(
        path="/mnt/dg3/ngoc/data_other/IGBP-SurfaceProducts_569/data/IGBP_wp_org.nc",
        mode="w",
        format="NETCDF4",
    )

    # convert to VISIT resolution
    ds = ds.where(ds["wiltpoint"] != -9999)
    interp_lat = np.load(VISIT_LAT_FILE)
    interp_lon = np.load(VISIT_LONG_FILE)

    interpolated_ds = ds["wiltpoint"].interp(lat=interp_lat, lon=interp_lon)
    interpolated_ds = interpolated_ds / 1000
    interpolated_ds = interpolated_ds.fillna(-9999.0)
    inter_ds = xr.Dataset(
        data_vars=dict(
            wiltpoint=(["lat", "lon"], interpolated_ds.values),
        ),
        coords=dict(
            lon=(interp_lon),
            lat=(interp_lat),
        ),
        attrs=dict(
            description="Global IGBP wilting point at 0.5 degree, fillna = -9999.0"
        ),
    )
    inter_nc_file = inter_ds.to_netcdf(
        path="/mnt/dg3/ngoc/data_other/IGBP-SurfaceProducts_569/data/IGBP_wp_0.5.nc",
        mode="w",
        format="NETCDF4",
    )
    df = inter_ds["wiltpoint"].values
    df = pd.DataFrame(df.flatten())
    inter_txt_file = df.to_csv(
        "/mnt/dg3/ngoc/data_other/IGBP-SurfaceProducts_569/data/wp_2visit.dat",
        header=None,
        index=None,
    )

    return org_nc_file, inter_nc_file, inter_txt_file


def resample(folder):
    latp = os.path.join(DATA_DIR, "gfdl_esm4_latlon", "lat.npy")
    lonp = latp.replace("lat.npy", "lon.npy")
    int_folder = "1x125deg"

    lat, lon = np.load(latp), np.load(lonp)

    path = os.path.join(DATA_SERVER, folder)

    for p, sd, fs in os.walk(path):
        for name in fs:
            file_path = os.path.join(p, name)
            var_name = file_path.split("\\")[-1].split(".nc")[0].split("_")[-1]
            new_path = file_path.replace(folder, f"{int_folder}/{folder}")
            new_folder = p.replace(folder, f"{int_folder}/{folder}")

            if not os.path.exists(new_path):
                if "VISIT" in file_path:
                    ds = xr.open_dataset(file_path, decode_times=False)
                else:
                    ds = xr.open_dataset(file_path)
                resampled_ds = interpolate(ds, lat, lon)

                if not os.path.isdir(new_folder):
                    os.makedirs(new_folder)

                resampled_ds.to_netcdf(new_path)


# %%
