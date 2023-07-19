# %%
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import cartopy.crs as ccrs

from const import *
from mypath import *
from mk import *
from mulLinear import cal_actual_rate


def cal_mk(ds, var_name):
    ds_var = ds[var_name]
    x = ds_var
    y = xr.DataArray(
        np.arange(len(ds_var["year"])) + 1,
        dims="year",
        coords={"year": ds_var["year"]},
    )
    slope = kendall_correlation(x, y, "year")
    return slope


class VisitSenAls:
    base_dir = "/mnt/dg3/ngoc/cmip6_bvoc_als/data/processed_data/VISIT-sensitivity/"
    list_var = ["co2", "co2_met", "co2_met_luc"]
    list_drivers = ["co2", "luc", "met", "co2_met_luc"]

    def __init__(self, var_name="emiisop") -> None:
        self.var_name = var_name

        self.files = {
            var: os.path.join(self.base_dir, f"VISIT-SH{i}_{self.var_name}.nc")
            for i, var in enumerate(self.list_var, 1)
        }

        self.ds_vars = {}
        self.mask = None
        self.org_mk = {}

        # for cao we refer to the change of 25 recent years and 25 past years
        self.ds_cao_pixel = {}
        self.driver_glob_cao = pd.DataFrame()

        self.ds_mk_pixel = {}
        self.driver_glob_mk = pd.DataFrame()

        self.load_data()
        self.cal_mask()
        self.cal_driver_mk()
        self.cal_driver_cao()
        self.plt_glob_rate_drivers()
        # self.plt_glob_driver_pixel(mode="mk")
        self.plt_glob_driver_pixel(mode="cao")

    def load_data(self):
        for v in self.list_var:
            self.ds_vars[v] = xr.open_dataset(self.files[v])

        met = xr.Dataset({})
        luc = xr.Dataset({})

        met[self.var_name] = (
            self.ds_vars["co2_met"][self.var_name] - self.ds_vars["co2"][self.var_name]
        )

        luc[self.var_name] = (
            self.ds_vars["co2_met_luc"][self.var_name]
            - self.ds_vars["co2_met"][self.var_name]
        )

        self.ds_vars["met"] = met
        self.ds_vars["luc"] = luc

    def cal_mask(self):
        mask = self.ds_vars["co2"][self.var_name].mean("year").values
        mask[mask > 0] = 1
        mask[mask <= 0] = np.nan
        self.mask = mask

    def cal_driver_mk(self):
        mk_rates = []
        for v in self.list_drivers:
            print(v)
            file_path = os.path.join(self.base_dir, "mk", f"{v}.nc")
            if os.path.exists(file_path):
                ds = xr.open_dataset(file_path)
                ds = ds.rename(name_dict={list(ds.keys())[0]: v})
                self.org_mk[v] = ds[v]
            else:
                mk_ds = xr.Dataset({})
                mk_ds[v] = cal_mk(self.ds_vars[v], self.var_name)
                mk_ds.to_netcdf(file_path)
                self.org_mk[v] = mk_ds[v]

            glob_rate, glob_change = cal_actual_rate(self.org_mk[v], "VISIT")
            mk_rates.append(glob_rate)
            self.ds_mk_pixel[v] = glob_change

        self.driver_glob_mk["driver"] = self.list_drivers
        self.driver_glob_mk["rate"] = mk_rates
        (
            self.driver_mk_sign_pixel,
            self.driver_mk_abs_pixel,
        ) = self.cal_drivers_pixel_level(self.ds_mk_pixel)

    def cal_driver_cao(self):
        rates = []
        for v in self.list_drivers:
            ds_pi = (
                self.ds_vars[v][self.var_name].sel(year=slice(1850, 1875)).mean("year")
            )
            ds_pd = (
                self.ds_vars[v][self.var_name].sel(year=slice(1990, 2014)).mean("year")
            )

            glob_rate, glob_change = cal_actual_rate((ds_pd - ds_pi), "VISIT")

            rates.append(glob_rate)
            self.ds_cao_pixel[v] = glob_change

        self.driver_glob_cao["driver"] = self.list_drivers
        self.driver_glob_cao["rate"] = rates
        (
            self.driver_cao_sign_pixel,
            self.driver_cao_abs_pixel,
        ) = self.cal_drivers_pixel_level(self.ds_cao_pixel)

    def cal_drivers_pixel_level(self, ds_pixel):
        met = ds_pixel["met"].values
        luc = ds_pixel["luc"].values
        co2 = ds_pixel["co2"].values

        # co2 met luc
        cml = ds_pixel["co2_met_luc"].values

        stacked = np.stack((co2 * cml, luc * cml, met * cml), axis=-1)

        abs_stacked = np.absolute(stacked)

        max_stacked = np.argmax(stacked, axis=2)
        max_abs_stacked = np.argmax(abs_stacked, axis=2)

        valid_sign = max_stacked == max_abs_stacked

        driver_arr = valid_sign * max_stacked
        coords = {
            "lon": ds_pixel["met"].lon.values,
            "lat": ds_pixel["met"].lat.values,
        }

        driver_sign_pixel = xr.Dataset(
            {"driver": (("lat", "lon"), driver_arr * self.mask)}, coords=coords
        )
        driver_abs_pixel = xr.Dataset(
            {"driver": (("lat", "lon"), max_abs_stacked * self.mask)}, coords=coords
        )
        return driver_sign_pixel, driver_abs_pixel

    def plt_glob_rate_drivers(self):
        rs, cs = 2, 1
        w, h = 6 * cs, 6 * rs
        fig, axes = plt.subplots(rs, cs, figsize=(w, h))
        for i, (df, tit) in enumerate(
            zip([self.driver_glob_cao, self.driver_glob_mk], ["cao", "mk"])
        ):
            ax = axes[i]
            sns.barplot(
                df,
                x="driver",
                y="rate",
                ax=ax,
                palette=sns.color_palette(["#8dd3c7", "#ffffb3", "#bebada", "#b3de69"]),
            ).set(title=f"{tit}")
            ax.set_xlabel("Drivers of change")
            ax.set_ylabel("Isoprene emission [TgC]")

    def plt_glob_driver_pixel(self, mode="cao"):
        # if mode == mk
        # sign_driver = self.driver_mk_sign_pixel
        # abs_driver = self.driver_mk_abs_pixel

        if mode == "cao":
            sign_driver = self.driver_cao_sign_pixel
            # abs_driver = self.driver_cao_abs_pixel

        # rs, cs = 2, 1
        # w, h = 6 * cs, 6 * rs
        # fig, axes = plt.subplots(rs, cs, figsize=(w, h))
        # for i, (ds, tit) in enumerate(
        #     zip([sign_driver, abs_driver], ["same sign", "max abs values"])
        # ):
        #     ax = axes[i]
        #     cmap = matplotlib.colors.ListedColormap(
        #         matplotlib.colormaps["Accent"].colors[:4]
        #     )
        #     ds["driver"].plot(levels=4, vmin=0, vmax=3, cmap=cmap, ax=ax)
        #     ax.set(title=f"{mode} {tit}")

        fig = plt.figure(figsize=(12, 9))
        ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())
        ax.coastlines()
        ax.gridlines(
            xlocs=range(-180, 180, 40),
            ylocs=range(-80, 81, 20),
            draw_labels=True,
            linewidth=1,
            edgecolor="dimgrey",
        )
        title = f"Drivers of changes in {self.var_name}"
        cmap = matplotlib.colors.ListedColormap(matplotlib.colormaps["Set3"].colors[:3])
        center = [0.5, 1.5, 2.5]
        cax = sign_driver["driver"].plot(
            cmap=cmap,
            vmin=0,
            vmax=3,
            ax=ax,
            add_colorbar=False,
        )
        cbar = fig.colorbar(
            cax,
            ticks=center,
            orientation="horizontal",
            pad=0.05,
        )
        cbar.ax.set_xticklabels(["co2s", "luc", "met"], size=14)
        cbar.set_label(label="Dominant driver", size=14, weight="bold")
        plt.title(title, fontsize=18)


# %%
