# %%
import pandas as pd
from sklearn.linear_model import LinearRegression

from CMIP6Model import *
import statsmodels.api as sm

sns.set_style("ticks")

predictors = ["lai", "co2s", "tas", "rsds", "pr"]

target = ["emiisop"]

FIG_INDEX = 0


def cal_actual_rate(ds, model_name):
    base_dir = "/mnt/dg3/ngoc/cmip6_bvoc_als/data/axl/areacella"
    fname = (
        "areacella_fx_GFDL-ESM4_historical_r1i1p1f1_gr1.nc"
        if "VISIT" not in model_name
        else "areacella_fx_VISIT_historical_r1i1p1f1_gn.nc"
    )
    ds_area = xr.open_dataset(os.path.join(base_dir, fname))

    reindex_ds_area = ds_area["areacella"].reindex_like(
        ds, method="nearest", tolerance=0.01
    )
    global_change = ds * reindex_ds_area * 1e-12
    global_rate = global_change.sum(dim=["lat", "lon"]).item()
    # global_change = ds
    # global_rate = global_change.mean(dim=["lat", "lon"]).item()

    return global_rate, global_change


def prep_to_clip_reg(ds):
    ds = ds.rio.write_crs("epsg:4326", inplace=True)

    ds.coords["lon"] = (ds.coords["lon"] + 180) % 360 - 180

    ds = ds.sortby(ds.lon)
    ds = ds.rio.set_spatial_dims("lon", "lat", inplace=True)
    return ds


def wrap_long(data):
    # print("Original shape -", data.shape)
    lon = data.coords["lon"]
    lon_idx = data.dims.index("lon")
    wrap_data, wrap_lon = add_cyclic_point(data.values, coord=lon, axis=lon_idx)
    # print("New shape -", wrap_data.shape)
    return wrap_data, wrap_lon


class RegSingleModel:
    def __init__(
        self,
        target_name=target,
        predictor_names=predictors,
        model_name="VISIT",
    ) -> None:
        self.target_name = target_name[0]
        self.predictor_names = predictor_names
        self.model_name = model_name

        self.org_ds = None

        self.read_data()
        self.ml_als()
        self.cal_change_con()
        self.cal_global_change_con()
        self.cal_regional_change_con()

        self.plt_map_change()
        self.plt_global_change_con()
        # self.plt_regional_change_con()

    def read_data(self):
        iters = self.predictor_names + [self.target_name]
        dss = {}
        for p in iters:
            f = os.path.join(RES_DIR, f"{self.model_name}_{p}.nc")
            ds = xr.open_dataset(f)[p]
            # ds = ds.sel(year=slice(1901, 2014)) if "VISIT" in self.model_name else ds
            dss[p] = (("lat", "lon", "year"), ds.transpose("lat", "lon", "year").data)
        lat = ds.lat.values
        lon = ds.lon.values
        year = ds.year.values
        self.org_ds = xr.Dataset(dss, coords={"lat": lat, "lon": lon, "year": year})

    def regression(self, data):
        lai = data["lai"].values
        co2s = data["co2s"].values
        tas = data["tas"].values
        rsds = data["rsds"].values
        pr = data["pr"].values
        X = np.column_stack((lai, co2s, tas, rsds, pr))

        y = data[self.target_name]
        # X[np.isnan(X)] = 0
        # y[np.isnan(y)] = 0
        X = np.nan_to_num(X)
        y = np.nan_to_num(y).reshape(-1)
        est = LinearRegression()
        est.fit(X, y)
        return xr.DataArray(est.coef_)

    def ml_als(self):
        stacked = self.org_ds.stack(allpoints=["lat", "lon"])
        coefs = stacked.groupby("allpoints").map(self.regression)
        self.coefs = coefs.unstack("allpoints")

    def cal_change_con(self):
        clai = self.coefs.isel(dim_0=0).values
        cco2 = self.coefs.isel(dim_0=1).values
        ctas = self.coefs.isel(dim_0=2).values
        crsds = self.coefs.isel(dim_0=3).values
        cpr = self.coefs.isel(dim_0=4).values

        lai = self.org_ds["lai"].values * clai.reshape(clai.shape + (1,))
        co2s = self.org_ds["co2s"].values * cco2.reshape(cco2.shape + (1,))
        tas = self.org_ds["tas"].values * ctas.reshape(ctas.shape + (1,))
        rsds = self.org_ds["rsds"].values * crsds.reshape(crsds.shape + (1,))
        pr = self.org_ds["pr"].values * cpr.reshape(cpr.shape + (1,))

        lai_rate = lai[:, :, -25:-1].mean(axis=2) - lai[:, :, 0:25].mean(axis=2)
        co2s_rate = co2s[:, :, -25:-1].mean(axis=2) - co2s[:, :, 0:25].mean(axis=2)
        tas_rate = tas[:, :, -25:-1].mean(axis=2) - tas[:, :, 0:25].mean(axis=2)
        rsds_rate = rsds[:, :, -25:-1].mean(axis=2) - rsds[:, :, 0:25].mean(axis=2)
        pr_rate = pr[:, :, -25:-1].mean(axis=2) - pr[:, :, 0:25].mean(axis=2)
        est_stacked = np.stack(
            (lai_rate, co2s_rate, tas_rate, rsds_rate, pr_rate),
            axis=-1,
        )
        abslt_est = np.absolute(est_stacked)
        max_contrb = np.argmax(abslt_est, axis=2)

        total_con = lai_rate + co2s_rate + tas_rate + rsds_rate + pr_rate

        abs_con = xr.Dataset(
            {"abs_con": (("lat", "lon"), max_contrb)},
            coords={"lon": self.org_ds.lon.values, "lat": self.org_ds.lat.values},
        )
        real_con = xr.Dataset(
            {
                "total_con": (("lat", "lon"), total_con),
                "lai": (("lat", "lon"), lai_rate),
                "co2s": (("lat", "lon"), co2s_rate),
                "tas": (("lat", "lon"), tas_rate),
                "rsds": (("lat", "lon"), rsds_rate),
                "pr": (("lat", "lon"), pr_rate),
            },
            coords={"lon": self.org_ds.lon.values, "lat": self.org_ds.lat.values},
        )

        self.proj_abs_con = abs_con
        self.proj_real_con = real_con

    def cal_global_change_con(self):
        target_change = self.org_ds[self.target_name].isel(year=slice(-25, -1)).mean(
            "year"
        ) - self.org_ds[self.target_name].isel(year=slice(0, 25)).mean("year")
        target_rate, self.target_change = cal_actual_rate(
            target_change, self.model_name
        )

        df = pd.DataFrame()
        des = []
        rates = []
        prj = 0

        self.pred_change = {}
        for p in self.predictor_names:
            des.append(p)

            rate, self.pred_change[p] = cal_actual_rate(
                self.proj_real_con[p], self.model_name
            )
            prj += rate
            rates.append(rate)

        des.append("regression")
        des.append("model")
        rates.append(prj)
        rates.append(target_rate)
        df["des"] = des
        df["rates"] = rates

        self.global_change_con = df

    def cal_regional_change_con(self):
        df = pd.DataFrame()
        rates = []
        des = []
        scale = []
        for roi in LIST_REGION:
            regression = 0
            target_change = prep_to_clip_reg(self.target_change)
            target_reg_rate = (
                clip_region_mask(target_change, roi).sum(dim=["lat", "lon"]).item()
            )
            for p in self.predictor_names:
                des.append(p)

                pred_change = self.pred_change[p]
                pred_change = prep_to_clip_reg(pred_change)

                rate = clip_region_mask(pred_change, roi).sum(dim=["lat", "lon"]).item()
                rates.append(rate)

                regression += rate

                scale.append(roi)

            des.append("regression")
            rates.append(regression)
            scale.append(roi)

            des.append("model")
            rates.append(target_reg_rate)
            scale.append(roi)

        df["scale"] = scale
        df["des"] = des
        df["rates"] = rates
        self.regional_change_con = df

    def plt_map_change(self):
        global FIG_INDEX
        # rs, cs = 1, 2
        # w, h = 15 * cs, 12 * rs
        # fig, ax = plt.subplots(rs, cs, figsize=(w, h))

        # # self.proj_abs_con["abs_con"].plot(ax=ax[0], levels=5, vmin=0, vmax=4)
        # # self.proj_real_con["total_con"].plot(ax=ax[1], extend="both", cmap="coolwarm")
        # self.proj_abs_con["abs_con"].plot(ax=ax[0])

        # self.proj_real_con["total_con"].plot(
        #     ax=ax[1],
        #     extend="both",
        #     cmap="coolwarm",
        # )
        # ax[0].set_title(f"{self.model_name} - Drivers of change in {self.target_name}")
        # ax[1].set_title(
        #     f"{self.model_name} - Changes in {self.target_name} between PI and PD"
        # )
        mask = self.target_change.values
        mask[mask != 0] = 1
        mask[mask == 0] = np.nan

        mode = ["drivers", "total_change"]
        for m in mode:
            FIG_INDEX += 1
            fig = plt.figure(1 + FIG_INDEX, figsize=(12, 9))
            ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())
            # regionmask.defined_regions.srex.plot(
            #     ax=ax,
            #     projection=ccrs.PlateCarree(),
            #     regions=["SEA", "EAS", "AMZ", "SSA"],
            #     add_label=True,
            #     label="abbrev",
            # )
            ax.coastlines()
            ax.gridlines(
                xlocs=range(-180, 180, 40),
                ylocs=range(-80, 81, 20),
                draw_labels=True,
                linewidth=1,
                edgecolor="dimgrey",
            )
            if m == "drivers":
                data = self.proj_abs_con["abs_con"] * mask
                wrap_data, wrap_lon = wrap_long(data)

                title = f"{self.model_name} - Drivers of changes in {self.target_name}"
                center = [0.5, 1.5, 2.5, 3.5, 4.5]

                cax = ax.pcolormesh(
                    wrap_lon,
                    data.lat,
                    wrap_data,
                    cmap=matplotlib.colors.ListedColormap(
                        matplotlib.colormaps["Accent"].colors[:5]
                    ),
                    vmin=0,
                    vmax=5,
                )

                cbar = fig.colorbar(
                    cax,
                    ticks=center,
                    orientation="horizontal",
                    pad=0.05,
                )
                cbar.ax.set_xticklabels(["lai", "co2s", "tas", "rsds", "pr"], size=14)
                cbar.set_label(label="Dominant driver", size=14, weight="bold")

            if m == "total_change":
                data = self.proj_real_con["total_con"]
                wrap_data, wrap_lon = wrap_long(data)

                title = f"{self.model_name} - Changes in {self.target_name} between PI and PD"
                cax = ax.pcolormesh(
                    wrap_lon,
                    data.lat,
                    wrap_data,
                    cmap="coolwarm",
                    # levels=11,
                    vmin=-12,
                    vmax=12,
                )
                cbar = fig.colorbar(
                    cax,
                    extend="both",
                    orientation="horizontal",
                    pad=0.05,
                )
                cbar.set_label("[$gC/m^{2}/year$]")
            plt.title(title, fontsize=18)

    def plt_global_change_con(self):
        rs, cs = 1, 1
        w, h = 6 * cs, 6 * rs
        fig, axes = plt.subplots(rs, cs, figsize=(w, h))
        sns.barplot(
            self.global_change_con,
            x="des",
            y="rates",
            ax=axes,
            palette=sns.color_palette("Accent", len(self.global_change_con)),
        ).set(title=f"{self.model_name}")
        axes.set_xlabel(" ")
        axes.set_ylabel("Isoprene emission [TgC]")
        axes.set_ylim([-120, 60])

    def plt_regional_change_con(self):
        sns.barplot(self.regional_change_con, x="scale", y="rates", hue="des")


class MultiModelReg:
    def __init__(self) -> None:
        self.model_names = [
            "CESM2-WACCM",
            "GFDL-ESM4",
            "GISS-E2-1-G",
            "NorESM2-LM",
            "UKESM1-0-LL",
        ]
        self.reg_ds = []
        self.glb_ds = []

        self.extr_data()

        self.plt_reg_change_con()

    def extr_data(self):
        global FIG_INDEX
        for mn in self.model_names:
            print(mn)
            reg_single_model = RegSingleModel(model_name=mn)
            FIG_INDEX += 1
            reg_df = reg_single_model.regional_change_con
            reg_df["model"] = [mn] * len(reg_df)

            self.reg_ds.append(reg_df)

        self.reg_ds = pd.concat(self.reg_ds, ignore_index=True)

    def plt_reg_change_con(self):
        rs, cs = 2, 2
        w, h = 10 * cs, 10 * rs
        fig, axes = plt.subplots(rs, cs, figsize=(w, h))
        for i, roi in enumerate(LIST_REGION):
            df = self.reg_ds[self.reg_ds["scale"] == roi]
            ax = axes[i // 2, i % 2]
            sns.barplot(
                df,
                x="model",
                y="rates",
                hue="des",
                ax=ax,
                palette=sns.color_palette("Accent", 7),
            )
            ax.set_title(roi)
            ax.set_ylabel("Isoprene emission [TgC]")


# %%