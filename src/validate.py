#%%
import xarray as xr
import xskillscore as xskill
import copy

from utils import *
from const import *

from visit_preprocess import visit_t2cft


DICT_MODEL_NAMES = get_list_model_by_name(glob.glob("../data/var/emiisop/*.nc"))

TOPDOWN_DS = xr.open_dataset(CONCAT_TOPDOWN_PATH)

AVAIL_TIME = slice("2005-01", "2014-12")


test_model = "CESM2-WACCM"


def interpolate(ds):
    interp_lat = np.load(VISIT_LAT_FILE)
    interp_lon = np.load(VISIT_LONG_FILE)

    interpolated_ds = ds.interp(lat=interp_lat, lon=interp_lon)

    return interpolated_ds


def val_single_model(model_name=test_model, var_name="emiisop"):

    if "VISIT" not in model_name:
        list_nc_files = DICT_MODEL_NAMES[model_name]
        model_ds = merge_by_model(list_nc_files)
    else:
        model_ds = visit_t2cft(VISIT_DICT_PATH[model_name], case=model_name)

    sliced_model_ds = model_ds.sel(time=AVAIL_TIME)

    # fix coords
    sliced_model_ds = copy.deepcopy(sliced_model_ds[var_name])
    sliced_model_ds = sliced_model_ds.rio.write_crs("epsg:4326", inplace=True)
    sliced_model_ds.coords["lon"] = (sliced_model_ds.coords["lon"] + 180) % 360 - 180
    sliced_model_ds = sliced_model_ds.sortby(sliced_model_ds.lon)
    sliced_model_ds = sliced_model_ds.rio.set_spatial_dims("lon", "lat", inplace=True)

    interpolated_ds = interpolate(sliced_model_ds)

    TOPDOWN_DS["time"] = interpolated_ds.time
    test_score = xskill.pearson_r(
        TOPDOWN_DS.emiisop / TOPDOWN_DS.Grid_area, interpolated_ds, dim="time"
    )
    return test_score


def plot_map(test_score, i):

    fig = plt.figure(1 + i, figsize=(30, 13))
    ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.gridlines(
        xlocs=range(-180, 180, 40),
        ylocs=range(-80, 81, 20),
        draw_labels=True,
        linewidth=1,
        edgecolor="dimgrey",
    )
    test_score.plot.pcolormesh(
        ax=ax,
        # # levels=21,    #customize for individual variable if needed
        # vmin= 0,
        # vmax= 35,
        extend="both",
        # cbar_kwargs={"label": VIZ_OPT[self.var_name]["map_unit"]},
    )


def main():

    # not visit
    for t, model_name in enumerate(DICT_MODEL_NAMES.keys()):
        print(model_name)
        test_score = val_single_model(model_name)
        plot_map(test_score, t)
        plt.title(model_name, fontsize=18)

    # visit
    for i, model_name in enumerate(VISIT_DICT_PATH):
        print(model_name)
        test_score = val_single_model(model_name)
        plot_map(test_score, i + t)
        plt.title(model_name, fontsize=18)


# %%
