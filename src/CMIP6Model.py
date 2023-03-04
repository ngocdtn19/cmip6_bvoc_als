#%%

from CMIP6Var import *
from const import *
from mypath import *
from scipy import stats
from visit_preprocess import *
from cartopy.util import add_cyclic_point
import mk
import pymannkendall as pymk
import xskillscore as xs

sns.set_style("ticks")

class Model:
    var_obj_dict = {
        "emiisop": Cmip6TimeSum,
        "emibvoc": Cmip6TimeSum,
        "emiotherbvocs": Cmip6TimeSum,
        "gpp": PP,
        "npp": PP,
        "pr": PR,
        "tas": TAS,
        "rsds": RSDS,
        "emioa": EMIOA,
        "chepsoa": CHEPSOA,
        "emipoa" :EMIPOA,
    }

    def __init__(self, model_name="UKESM1-0-LL") -> None:
        self.model_name = model_name
        self.var_names = list()
        self.var_objs = {}

        self.extract_vars()

    def get_var_ds(self, var_name):

        var_files = glob.glob(os.path.join(VAR_DIR, var_name, "*.nc"))
        model_files = [f for f in var_files if self.model_name in f]
        l_model_ds = []
        for f in model_files:
            var_ds = (
                visit_t2cft(VISIT_DICT_PATH[self.model_name], case=self.model_name) if "VISIT" in self.model_name  else (xr.open_dataset(f))
            )
            l_model_ds.append(var_ds)

        return xr.concat(l_model_ds, dim=DIM_TIME)

    def extract_vars(self):
        all_var_files = list()

        for (dp, dn, fn) in os.walk(VAR_DIR):
            all_var_files += [os.path.join(dp, file) for file in fn]

        # remove files without containing model_name
        all_var_files = [f for f in all_var_files if self.model_name in f]
        self.var_names = list(
            set([f.split("\\")[-1].split("_")[0] for f in all_var_files])
        )

        for v_name in self.var_names:
            if "VISIT" in self.model_name:
                self.var_objs[v_name] = CMIP6Visit(
                    self.model_name, self.get_var_ds(v_name), v_name
                )
            else:
                self.var_objs[v_name] = self.var_obj_dict[v_name](
                    self.model_name, self.get_var_ds(v_name), v_name
                )

    def plot_mk_annual(self, sy, ey, var_name, cmap="RdBu", levels=13, vmin=-0.024, vmax=0.024):
        annual_ds = self.var_objs[var_name].annual_per_area_unit.sel(year=slice(sy, ey))

        x = xr.DataArray(
            np.arange(len(annual_ds["year"])) + 1,
            dims="year",
            coords={"year": annual_ds["year"]},
        )
        slope = mk.kendall_correlation(annual_ds, x, "year")
        lon = slope.coords["lon"]
        lon_idx = slope.dims.index("lon")
        wrap_s, wrap_lon = add_cyclic_point(slope.values, coord=lon, axis=lon_idx)

        fig = plt.figure(1, figsize=(30, 13))
        ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())
        ax.coastlines()
        ax.gridlines(
            xlocs=range(-180, 180, 40),
            ylocs=range(-80, 81, 20),
            draw_labels=True,
            linewidth=1,
            edgecolor="dimgrey",
        )
        my_cmap = matplotlib.cm.get_cmap(cmap)
        title = (
            f"Annual Trends of {var_name} from {sy}-{ey} using the Modified Mann-Kendall method"
        )
        pc = plt.contourf(
            wrap_lon, 
            slope.lat, 
            wrap_s,
            cmap = my_cmap,
            levels = levels,
            # levels=[-0.02, -0.015, -0.01, -0.005, -0.001, -0.0005, 0, 0.0005, 0.001, 0.005, 0.01, 0.015, 0.02],
            # colors=["#053061","#2166ac", "#4393c3","#92c5de","#d1e5f0","#f7f7f7","#f7f7f7","#fddbc7", "#f4a582","#d6604d", "#b2182b", "#67001f"],
            # levels=[-0.016, -0.008, -0.004, -0.001, -0.0001, -0.0000001, 0, 0.0000001, 0.001, 0.004, 0.008, 0.016, 0.024],
            # colors=["#67001f","#b2182b", "#d6604d","#f4a582","#fddbc7","#f7f7f7","#f7f7f7","#d1e5f0", "#92c5de","#4393c3", "#2166ac", "#053061"],
            vmin = vmin,
            vmax = vmax,
            extend ="both",
        )
        cb = plt.colorbar(pc, ax=ax, orientation='vertical')
        cb.set_label(VIZ_OPT[var_name]["map_unit"])
        plt.title(title, fontsize=24)
        plt.savefig(os.path.join("../fig/", self.model_name, f"mk-{var_name}-{sy}-{ey}.png"))

    def plot_mk_ss(self, sy, ey, var_name, cmap="RdBu_r"):

        ss_ds = self.var_objs[var_name].seasonal_ds
        ss_ds = ss_ds.sel(time=ss_ds.time.dt.year.isin([i for i in range(sy, ey+1)]))

        for i, month in enumerate([12, 3, 6, 9]):
            xarr = ss_ds.sel(time=ss_ds.time.dt.month.isin([month]))
            x = xr.DataArray(
                np.arange(len(xarr[DIM_TIME])) + 1,
                dims=DIM_TIME,
                coords={DIM_TIME: xarr[DIM_TIME]},
            )
            s = mk.kendall_correlation(xarr, x, DIM_TIME)

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
            my_cmap = matplotlib.cm.get_cmap(cmap)

            data = s
            title = f"Seasonal trends of {var_name} from {sy}-{ey} using the Mann-Kendall method"
            data.plot.pcolormesh(
                ax=ax,
                cmap=my_cmap,
                cbar_kwargs={"label": VIZ_OPT[var_name]["map_unit"]},
            )
            plt.title(title, fontsize=18)
    
    def plot_regional_annual_trend(self, mode="annual"):
        var_names = self.var_objs.keys()
        # l_roi = list(ROI_DICT.keys())
        l_roi = LIST_REGION

        for v_name in var_names:
            # years = self.var_objs[v_name].years      

            f, ax = plt.subplots(figsize=(9, 7), layout="constrained")
            colors = ROI_COLORS
            for roi in l_roi:
                if mode == "annual":
                    arr = self.var_objs[v_name].regional_rate[roi]
                elif mode == "anomaly":
                    arr = self.var_objs[v_name].regional_rate_anml[roi]
                
                arr.plot(
                        ax=ax,
                        color=colors[roi],
                        marker="o",
                        lw=1.5,
                        ms=4,
                        markerfacecolor=colors[roi],
                        markeredgecolor=colors[roi],
                        label=roi,
                    )
                ax.legend(ncol=1, bbox_to_anchor=(0.5, 0.5, 0.7, 0.1), borderaxespad=0.0, loc="center right")
                ax.set_xlabel("Year")
                ax.set_ylabel(VIZ_OPT[v_name]["line_bar_unit"])
                ax.set_title(
                    f"{v_name} - Regional Variation"
                )
                # plt.ylim([2.2, 2.7]) # modified to customize the figure
                # plt.savefig(os.path.join("../fig/", self.model_name, f"{v_name}-reg-{mode}.png"))

    def plot_2var_global_annual_trend(self, sy, ey, cv="emiisop", mode="annual"):
        var_names = self.var_objs.keys()
        colors = {
            "emiisop": "green", "emibvoc": "#ff796c", "emioa": "#4d4d4d",
            "pr": "darkblue", "rsds": "#ff7f00", "tas": "#e41a1c", 
            "emiotherbvocs":"darkgreen", "gpp":"#762a83", "npp":"#9970ab", "chepsoa":"#878787", "emipoa":"#000000" }

        for v_name in var_names:
            cv_ds = self.var_objs[cv]
            ev_ds = self.var_objs[v_name]

            if mode == "annual":
                x1, y1 = cv_ds.global_rate.sel(year=slice(sy, ey)).year, cv_ds.global_rate.sel(year=slice(sy, ey))
                x2, y2 = cv_ds.global_rate.sel(year=slice(sy, ey)).year, ev_ds.global_rate.sel(year=slice(sy, ey))
            elif mode == "anomaly":
                x1, y1 = cv_ds.global_rate.sel(year=slice(sy, ey)).year, cv_ds.global_rate_anml.sel(year=slice(sy, ey))
                x2, y2 = cv_ds.global_rate.sel(year=slice(sy, ey)).year, ev_ds.global_rate_anml.sel(year=slice(sy, ey))

            slope, intercept, r, p, std_err = stats.linregress(y1, y2)

            fig, ax1 = plt.subplots(figsize=(10, 5), layout="constrained")
            axbox = ax1.get_position()
            ax1.plot(x1, y1, label=cv, linewidth=1.5, marker="o", ms=4, color=colors[cv], markerfacecolor="white", markeredgecolor=colors[cv])
            ax2=ax1.twinx()
            ax2.plot(x2, y2, label=v_name, linewidth=1.5, marker="o", ms=4, color=colors[v_name], markerfacecolor="white", markeredgecolor=colors[v_name])
            
            ax1.set_xlabel("Year")
            ax1.set_ylabel(VIZ_OPT[cv]["line_bar_unit"], color=colors[cv], fontweight='bold')
            ax2.set_ylabel(VIZ_OPT[v_name]["line_bar_unit"], color=colors[v_name], fontweight='bold')  
            # plt.ylim([250, 650])
            fig.suptitle(f"Annual Global Trend of {cv} - {v_name}")
            fig.legend(
                loc="center",
                ncol=2,
                bbox_to_anchor=[axbox.x0 + 0.5 * axbox.width, axbox.y0 - 0.15],
            )
            fig.text(0.875, 0.95, f"r = {np.round(r, decimals=3)}", fontsize=9)
            plt.savefig(os.path.join("../fig/", self.model_name, f"glob-{cv}-{v_name}-{mode}.png"))
    
    def plot_2var_regional_annual_trend(self, sy, ey, roi= "SA", cv="emiisop", mode="annual"):
        var_names = self.var_objs.keys()
        # l_roi = list(ROI_DICT.keys())
        l_roi = LIST_REGION
        colors = {
            "emiisop": "green", "emibvoc": "#ff796c", "emioa": "#4d4d4d",
            "pr": "darkblue", "rsds": "#ff7f00", "tas": "#e41a1c", 
            "emiotherbvocs":"darkgreen", "gpp":"#762a83", "npp":"#9970ab", "chepsoa":"#878787", "emipoa":"#000000" }

        for roi in l_roi:
            for v_name in var_names:
                cv_ds = self.var_objs[cv]
                ev_ds = self.var_objs[v_name]
                if mode == "annual":
                    x1, y1 = cv_ds.regional_rate[roi].sel(year=slice(sy, ey)).year, cv_ds.regional_rate[roi].sel(year=slice(sy, ey))
                    x2, y2 = ev_ds.regional_rate[roi].sel(year=slice(sy, ey)).year, ev_ds.regional_rate[roi].sel(year=slice(sy, ey))
                elif mode == "anomaly":
                    x1, y1 = cv_ds.regional_rate_anml[roi].sel(year=slice(sy, ey)).year, cv_ds.regional_rate_anml[roi].sel(year=slice(sy, ey))
                    x2, y2 = ev_ds.regional_rate_anml[roi].sel(year=slice(sy, ey)).year, ev_ds.regional_rate_anml[roi].sel(year=slice(sy, ey))

                slope, intercept, r, p, std_err = stats.linregress(y1, y2)

                fig, ax1 = plt.subplots(figsize=(10, 5), layout="constrained")
                axbox = ax1.get_position()
                ax1.plot(x1, y1, label=cv, linewidth=1.5, marker="o", ms=4, color=colors[cv], markerfacecolor="white", markeredgecolor=colors[cv])
                ax2=ax1.twinx()
                ax2.plot(x2, y2, label=v_name, linewidth=1.5, marker="o", ms=4, color=colors[v_name], markerfacecolor="white", markeredgecolor=colors[v_name])
                
                ax1.set_xlabel("Year")
                ax1.set_ylabel(VIZ_OPT[cv]["line_bar_unit"], color=colors[cv], fontweight='bold')
                ax2.set_ylabel(VIZ_OPT[v_name]["line_bar_unit"], color=colors[v_name], fontweight='bold')  
                # plt.ylim([250, 650])
                fig.suptitle(f"{roi} - Annual Variation of {cv} - {v_name}")
                fig.legend(
                    loc="center",
                    ncol=2,
                    bbox_to_anchor=[axbox.x0 + 0.5 * axbox.width, axbox.y0 - 0.15],
                )
                fig.text(0.875, 0.95, f"r = {np.round(r, decimals=3)}", fontsize=9)
                plt.savefig(os.path.join("../fig/", self.model_name, f"{roi}-{cv}-{v_name}-{mode}.png"))
    
class Var:
    var_obj_dict = {
        "emiisop": Cmip6TimeSum,
        "emibvoc": BVOC,
        "emiotherbvocs": BVOC,
        "gpp": PP,
        "npp": PP,
        "pr": PR,
        "tas": TAS,
        "rsds": RSDS,
        "emioa": EMIOA,
        "chepsoa": CHEPSOA,
        "emipoa": EMIPOA,
    }
    processed_dir = "../data/processed_data"

    def __init__(self, var_name):

        self.var_name = var_name
        self.obj_type = None
        self.multi_models = {}

        self.get_obj_type()
        self.get_multi_models()

    def get_obj_type(self):

        self.obj_type = self.var_obj_dict[self.var_name]

    def get_model_name(self, path):
        var_name = {
            "emiisop": "AERmon",
            "emibvoc": "AERmon",
            "emiotherbvocs": "AERmon",
            "gpp": "Lmon",
            "npp": "Lmon",
            "pr": "Amon",
            "tas": "Amon",
            "rsds": "Amon",
            "emioa": "AERmon",
            "chepsoa": "AERmon",
            "emipoa": "AERmon",
        }
        return (
            path.split("\\")[-1]
            .split(var_name[self.var_name])[-1]
            .split("historical")[0]
            .replace("_", "")
        )

    def get_multi_models(self):

        all_files = glob.glob(os.path.join(VAR_DIR, self.var_name, "*.nc"))
        model_names = sorted(list(set([self.get_model_name(f) for f in all_files])))

        multi_models = {}
        for m_name in model_names:
            if "VISIT" not in m_name:
                print(m_name)
                l_model = []
                for f in all_files:

                    if m_name in f:
                        l_model.append(xr.open_dataset(f))

                multi_models[m_name] = self.obj_type(
                    m_name, xr.concat(l_model, dim=DIM_TIME), self.var_name
                )

        if self.var_name == "emiisop":
            for visit_case in VISIT_DICT_PATH.keys():
                multi_models[visit_case] = CMIP6Visit(visit_case, visit_t2cft(VISIT_DICT_PATH[visit_case], case=visit_case), self.var_name)
        self.multi_models = multi_models

    def plot_regional_map(self):
        # rois = list(ROI_DICT.keys())
        rois = LIST_REGION
        l_m_name = list(self.multi_models.keys())
        ds = self.multi_models[l_m_name[0]]
        for i, r in enumerate(rois):
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
            data = ds.regional_ds[r].isel(year=0)
            data.plot.pcolormesh(
                ax=ax,
                cmap="RdPu",
                cbar_kwargs={"label": VIZ_OPT[self.var_name]["map_unit"]},  
            )
            plt.title(f"{l_m_name[0]} - {r} ", fontsize=18)

    def plot_global_map(self, bg_color="snow", cmap1="YlGnBu", cmap2="Spectral_r"):
        years = [1850, 2014, "mean", "change", "changePD", "PD", "PI"]                        #change to 1901 if including VISIT
        l_m_name = self.multi_models.keys()
        i = 0
        for m_name in l_m_name:
            ds = self.multi_models[m_name]
            print(m_name)
            for y in years:
                i = i + 1
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

                my_cmap1 = matplotlib.cm.get_cmap(cmap1)
                my_cmap1.set_under(bg_color)

                my_cmap2 = matplotlib.cm.get_cmap(cmap2)
                my_cmap2.set_under(bg_color)

                if y == "change":
                    data = ds.annual_per_area_unit.isel(
                        year=-1
                    ) - ds.annual_per_area_unit.isel(year=0)
                    title = f"{m_name} - Change from 1850 - 2014"
                    cmap = my_cmap1
                elif y == "changePD":
                    data = ds.annual_per_area_unit.sel(year=slice(1990, 2014)).mean(
                        "year"
                    ) - ds.annual_per_area_unit.sel(year=slice(1850, 1870)).mean("year")
                    title = f"{m_name} - Change between PD and PI"
                    cmap = my_cmap1
                elif y == "PD":
                    data = ds.annual_per_area_unit.sel(year=slice(1990, 2014)).mean("year") 
                    title = f"{m_name} - {self.var_name} at present day (PD, 1990-2014)"
                    cmap = my_cmap2
                elif y == "PI":
                    data = ds.annual_per_area_unit.sel(year=slice(1850, 1870)).mean("year") 
                    title = f"{m_name} - {self.var_name} at pre-industry (PI, 1850-1870)"
                    cmap = my_cmap2
                elif y == "mean":
                    data = ds.annual_per_area_unit.mean("year")
                    title = f"{m_name} - Mean from 1850 - 2014"
                    cmap = my_cmap2
                else:
                    data = ds.annual_per_area_unit.sel(year=y)
                    title = f"{m_name} - {y}"
                    cmap = my_cmap2
                data.plot.pcolormesh(
                    ax=ax,
                    cmap=cmap,
                    # # levels=21,    #customize for individual variable if needed
                    vmin= 0,
                    vmax= 35,
                    extend='both',
                    cbar_kwargs={"label": VIZ_OPT[self.var_name]["map_unit"]},
                )
                plt.title(title, fontsize=18)
                plt.savefig(
                    os.path.join("../fig/", self.var_name, f"{m_name}-{y}-map.png")
                )

    def plot_global_monthly_map(self, bg_color="snow", cmap="RdPu"):  
        indexs = [1956, 1959, 1962, 1965]
        l_m_name = self.multi_models.keys()
        i = 0
        for m_name in l_m_name:
            ds = self.multi_models[m_name]
            print(m_name)
            for m in indexs:
                i = i + 1
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
                my_cmap = matplotlib.cm.get_cmap(cmap)
                my_cmap.set_under(bg_color)

                data = ds.monthly_per_area_unit.isel(time=m)
                title = f"{m_name} - ({data.time.dt.year.item()}-{calendar.month_name[data.time.dt.month.item()]})"
                data.plot.pcolormesh(
                    ax=ax,
                    cmap=my_cmap,
                    # levels = 17,
                    # vmin = 0.001,
                    # vmax = 4,
                    cbar_kwargs={"label": "gC/m2/month"},
                )
                plt.title(title, fontsize=18)

    def plot_global_annual_trend(self, mode="annual"):
        model_names = self.multi_models.keys()
        colors = {"CESM2-WACCM": "#7fc97f", "GFDL-ESM4": "#1f78b4", "GISS-E2-1-G": "#ff7f00", "NorESM2-LM": "#33a02c", "UKESM1-0-LL": "#984ea3", "VISIT_ORG": "#e41a1c", "VISIT_CASE1": "#e41a1c", "VISIT_CASE2": "#e41a1c", "VISIT_CASE3": "#e41a1c"}
        fig, ax = plt.subplots(figsize=(10, 6.5), layout="constrained")
        axbox = ax.get_position()
        for name in model_names:
            cmip6_obj = self.multi_models[name]
            if mode == "annual":
                x1, y1 = cmip6_obj.years, cmip6_obj.global_rate
                # x2, y2 = cmip6_obj.global_rate["year"].sel(year=slice(1980, 2014)), cmip6_obj.global_rate.sel(year=slice(1980, 2014))
                # res2 = pymk.yue_wang_modification_test(y2, alpha=0.05)
                # trend_line2 = np.arange(len(y2)) * res2.slope + res2.intercept
            elif mode == "y10":
                x1, y1 = cmip6_obj.y10_rate["years"], cmip6_obj.y10_rate["y10_rate"]
                x2, y2 = cmip6_obj.years, cmip6_obj.global_rate
            elif mode == "anomaly":
                x1, y1 = cmip6_obj.years, cmip6_obj.global_rate_anml
                # x2, y2 = cmip6_obj.global_rate["year"].sel(year=slice(1980, 2014)), cmip6_obj.global_rate.sel(year=slice(1980, 2014))
                # res2 = pymk.yue_wang_modification_test(y2, alpha=0.05)
                # trend_line2 = np.arange(len(y2)) * res2.slope + res2.intercept

            ax.plot(x1, y1, label=name, linewidth=1.5, marker="o", ms=4, color=colors[name], markerfacecolor="white", markeredgecolor=colors[name])
            # ax.plot(x2, trend_line2, linewidth=1.5, ls="--", color=colors[name])
            # print(name)
            # print(pymk.yue_wang_modification_test(y2, alpha=0.05))

        ax.set_xlabel("Year")
        ax.set_ylabel(VIZ_OPT[self.var_name]["line_bar_unit"])  
        # plt.ylim([250, 650])
        # ax.set_title(f"Annual Global Trend of {self.var_name}")
        ax.legend(
            loc="center",
            ncol=6,
            bbox_to_anchor=[axbox.x0 + 0.5 * axbox.width, axbox.y0 - 0.25],
        )

    def plot_regional_annual_trend(self, mode="annual"):
        
        model_names = self.multi_models.keys()
        roi_ds = {}
        for roi in LIST_REGION:
            roi_ds[roi] = []
            years = []

            for name in model_names:
                years.append(self.multi_models[name].years)
                if mode == "annual":
                    roi_ds[roi].append(self.multi_models[name].regional_rate[roi])
                elif mode == "anomaly":
                    roi_ds[roi].append(self.multi_models[name].regional_rate_anml[roi])
            Var.plot_annual_trend(years, roi_ds[roi], model_names, self.var_name, roi)

    @staticmethod
    def plot_annual_trend(l_x, l_y, l_name, var_name, roi, unit=""):
        colors = {"CESM2-WACCM": "#7fc97f", "GFDL-ESM4": "#1f78b4", "GISS-E2-1-G": "#ff7f00", "NorESM2-LM": "#33a02c", "UKESM1-0-LL": "#984ea3", "VISIT_ORG": "#e41a1c", "VISIT_CASE1": "#e41a1c", "VISIT_CASE2": "#e41a1c", "VISIT_CASE3": "#e41a1c"}
        fig, ax = plt.subplots(figsize=(10, 6.5), layout="constrained")
        axbox = ax.get_position()
        for x, y, n in zip(l_x, l_y, l_name):
            ax.plot(x, y, label=n, linewidth=1.5, marker="o", ms=4, color=colors[n], markerfacecolor="white", markeredgecolor=colors[n])
        ax.set_xlabel("Year")
        ax.set_ylabel(VIZ_OPT[var_name]["line_bar_unit"])
        ax.set_title(
            f"{roi} - Annual Trend of {var_name}"
        )
        # plt.ylim([-12.5, 7.5])
        ax.legend(
            loc="center",
            ncol=6,
            bbox_to_anchor=[axbox.x0 + 0.5 * axbox.width, axbox.y0 - 0.25],
        )
        plt.savefig(os.path.join("../fig/", var_name, f"{roi}-a-anml.png"))


    def plot_global_seasonal_trend(self, mode="abs"):
        model_names = self.multi_models.keys()
        for name in model_names:
            global_ss_rate = self.multi_models[name].global_seasonal_rate
            global_ss_rate_anml = self.multi_models[name].global_seasonal_rate_anml

            if mode == "abs":
                rate = global_ss_rate
            elif mode == "anomaly":
                rate = global_ss_rate_anml

            Var.plot_seasonal_trend(rate, name, self.var_name)

    @staticmethod
    def plot_seasonal_trend(l_rate, l_name, var_name):
        colors = {3: "#5aae61", 6: "#1b7837", 9: "#bababa", 12: "#762a83"}
        seasons = {3: "Spring", 6: "Summer", 9: "Fall", 12: "Winter"}

        f, ax = plt.subplots(figsize=(12, 7), layout="constrained")
        axbox = ax.get_position()
        for month, arr in l_rate.groupby("time.month"):
            arr.plot(
                ax=ax,
                color="#bababa",
                marker="o",
                lw=1.75,
                ms=5.5,
                markerfacecolor=colors[month],
                markeredgecolor=colors[month],
                label=seasons[month],
            )
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.set_xlabel("Year")
        ax.set_ylabel(VIZ_OPT[var_name]["line_bar_unit"])
        ax.set_title(
            f"{l_name} - Seasonal Change in {var_name}"
        )
        # plt.ylim([2.2, 2.7]) # modified ylim if needed
        plt.savefig(os.path.join("../fig/", var_name, f"{l_name}-ss-trend.png"))

    def plot_stacked_bar(self, year):
        # l_roi = list(ROI_DICT.keys())
        l_roi = LIST_REGION

        model_names = list(self.multi_models.keys())

        l_y = []
        for roi in l_roi:
            if year == "mean":
                l_y.append(
                    np.array(
                        [
                            self.multi_models[name].regional_mean_1850_2014[roi].item()
                            for name in model_names
                        ]
                    )
                )
            else:
                l_y.append(
                    np.array(
                        [
                            self.multi_models[name]
                            .regional_rate[roi]
                            .sel(year=year)
                            .item()
                            for name in model_names
                        ]
                    )
                )
        l_y = np.array(l_y)
        df = pd.DataFrame({roi: val for roi, val in zip(l_roi, l_y)}, index=model_names)
        print(df)
        ax = df.plot.bar(
            stacked=True,
            color=ROI_COLORS,
            title=f"Regional Contribution - {year}",
            rot=45,
        )
        for x, y in enumerate(df.sum(axis=1)):
            ax.annotate(np.round(y, decimals=2), (x, y + 5), ha="center")
        ax.legend(
            l_roi,
            loc="right",
            ncol=1,
            bbox_to_anchor=(0.5, 0.5, 0.7, 0.1),
            borderaxespad=0.0,
        )
        # ax.set_ylim(VIZ_OPT[self.var_name]["bar_ylim"])
        ax.set_ylabel(VIZ_OPT[self.var_name]["line_bar_unit"])
    
    def plot_contribution(self):
        years = [1901, 2014, "mean"]                #modified periods if needed

        for y in years:
            self.plot_stacked_bar(y)

    def plot_hovmoller(self, bg_color="snow", cmap="RdYlBu_r"):
        periods = [                                #modified periods if needed
            (1850, 1864),
            (1975, 1989),
            (2000, 2014),
        ]
        l_m_names = list(self.multi_models.keys())
        my_cmap = matplotlib.cm.get_cmap(cmap)  
        my_cmap.set_under(bg_color) 
        for m_name in l_m_names:
            org_ds = self.multi_models[m_name]
            for p in periods:
                ds = org_ds.monthly_per_area_unit.mean(dim="lon").isel(
                    time=slice(y2p(p)[0], y2p(p)[1])
                )
                ds = ds.sel(lat=np.arange(-90, 90, 5), method="nearest")
                if "VISIT" in m_name:
                    xr.plot.contourf(ds,
                    x = ds.dims[0],
                    y = ds.dims[1],
                    figsize=(16, 5),  
                    cmap=my_cmap,  
                    levels=11, 
                    vmin=0.001,  
                    vmax=1.25,  
                    cbar_kwargs={"label": VIZ_OPT[self.var_name]["holv_unit"]},  
                )
                else: 
                    ds.plot.contourf(
                    figsize=(16, 5),  
                    cmap=my_cmap,  
                    levels=11, 
                    vmin=0.001,  
                    vmax=0.25,  
                    cbar_kwargs={"label": VIZ_OPT[self.var_name]["holv_unit"]},  
                )
                plt.xticks(rotation=45, ha="right")  
                plt.title(f"{m_name} - {p}", fontsize=18)
                plt.savefig(os.path.join("../fig/", self.var_name, f"{m_name}-{p}.png"))


    # def save_2_nc(self):

    #     model_names = list(self.multi_models.keys())

    #     for name in model_names:
    #         self.multi_models[name].monthly_ds.to_netcdf(
    #             os.path.join(
    #                 self.processed_dir, "monthly", f"{name}_{self.var_name}.nc"
    #             )
    #         )
    #         self.multi_models[name].annual_ds.to_netcdf(
    #             os.path.join(self.processed_dir, "annual", f"{name}_{self.var_name}.nc")
    #         )

    def plot_corr(self, cv="emiotherbvocs", times={"annual": "year"}, sy="1850", ey="2014"): 

        model_names = list(self.multi_models.keys())
        i = 0
        cv_multi_model = Var(cv)
        ev_multi_model = Var(self.var_name)
        for t in times.keys():
            for name in model_names:
                i = i + 1
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
                cv_ds = cv_multi_model.multi_models[name]
                ev_ds = ev_multi_model.multi_models[name]
                if t == "monthly":
                    corr = xr.corr(cv_ds.monthly_per_area_unit, ev_ds.monthly_per_area_unit, dim=[times[t]])
                    p = xs.pearson_r_p_value(cv_ds.monthly_per_area_unit, ev_ds.monthly_per_area_unit, dim=[times[t]])                                                    
                    sig = p.where(p < 0.05)                                                                                                         
                    X, Y = np.meshgrid(p.lon, p.lat)                                                                                                
                else:
                    corr = xr.corr(cv_ds.annual_per_area_unit.sel(year=slice(sy, ey)), ev_ds.annual_per_area_unit.sel(year=slice(sy, ey)), dim=[times[t]])                
                    p = xs.pearson_r_p_value(cv_ds.annual_per_area_unit.sel(year=slice(sy, ey)), ev_ds.annual_per_area_unit.sel(year=slice(sy, ey)), dim=[times[t]])      
                    sig = p.where(p < 0.05)                                                                                                         
                    X, Y = np.meshgrid(p.lon, p.lat)                                                                                                
                # corr.plot()
                corr.plot.pcolormesh(
                    ax=ax,
                    # levels=11,
                    vmin=-1,
                    vmax=1,
                    cmap="RdBu_r",
                    cbar_kwargs={"label": "Pearson Corr (r)"},
                )
                ax.pcolor(                                                                                                                          
                    X, Y, sig, hatch='//', alpha=0
                )
                fig_title = f"{t} - {name} - {cv} - {self.var_name} ({sy} - {ey})"
                plt.title(fig_title, fontsize=18)
                plt.savefig(os.path.join("../plot_corr", t, f"{fig_title}.png"))

    def plot_contribution_global_map(self, cv= "chepsoa", ev= "emioa" ,cmap="YlGnBu", bg_color="#ffffe5"):   #for checking
            years = ["mean", "PD", "PI"]
            l_m_name = self.multi_models.keys()
            i = 0
            cv_multi_model = Var(cv)
            ev_multi_model = Var(ev)
            for m_name in l_m_name:
                for y in years:
                    i = i + 1
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

                    my_cmap = matplotlib.cm.get_cmap(cmap)
                    my_cmap.set_under(bg_color)

                    cv_ds = cv_multi_model.multi_models[m_name]
                    ev_ds = ev_multi_model.multi_models[m_name]

                    if y == "mean":
                        cv_data = cv_ds.annual_per_area_unit.mean("year")
                        ev_data = ev_ds.annual_per_area_unit.mean("year")
                        title = f"{m_name} - Mean contribution of {cv} to {ev} from 1850 - 2014"
                        cmap = my_cmap
                    elif y == "PD":
                        cv_data = cv_ds.annual_per_area_unit.sel(year=slice(1990, 2014)).mean("year") 
                        ev_data = ev_ds.annual_per_area_unit.sel(year=slice(1990, 2014)).mean("year")
                        title = f"{m_name} - Contribution of {cv} to {ev} at present day (PD, 1990-2014)"
                        cmap = my_cmap
                    elif y == "PI":
                        cv_data = cv_ds.annual_per_area_unit.sel(year=slice(1850, 1870)).mean("year")
                        ev_data = ev_ds.annual_per_area_unit.sel(year=slice(1990, 2014)).mean("year")
                        title = f"{m_name} - Contribution of {cv} to {ev} at pre-industry (PI, 1850-1870)"
                        cmap = my_cmap

                    data = (cv_data/ev_data)*100
                    data.plot.pcolormesh(
                        ax=ax,
                        cmap=cmap,
                        levels= 11,
                        vmin= 0,
                        vmax= 100,
                        extend='both',
                        cbar_kwargs={"label": "%"},
                    )
                    plt.title(title, fontsize=24)
                    plt.savefig(
                        os.path.join("../fig/", cv, f"{m_name}-{y}-{cv}-frac-map.png")
                    )
class Land:
    def __init__(self, model_name="UKESM1-0-LL", mon_type="Emon") -> None:

        self.model_name = model_name
        self.mon_type = mon_type

        self.org_cell_objs = {}
        self.area_weighted_cell_obj = {}
        self.roi_ltype = {}
        self.roi_area = {}

        self.ds_area = None
        self.ds_sftlf = None

        self.get_ds_area()
        self.extract_merge_land_type()
        self.cal_area_weighted_cell()
        self.clip_2_roi()

    def get_ds_area(self):

        ds_area = [xr.open_dataset(f) for f in AREA_LIST if self.model_name in f][0]
        ds_sftlf = [xr.open_dataset(f) for f in SFLTF_LIST if self.model_name in f][0]

        self.ds_sftlf = ds_sftlf[VAR_SFTLF].reindex_like(
            ds_area, method="nearest", tolerance=0.01
        )
        self.ds_area = self.ds_sftlf * ds_area * 1e-2

    def extract_merge_land_type(self):

        all_nc = glob.glob(
            os.path.join(LAND_DIR, self.model_name, self.mon_type, "*.nc")
        )
        self.land_type = list(
            set([f.split("\\")[-1].split("_")[0] for f in all_nc if self.mon_type in f])
        )
        self.land_type.remove(
            "fracLut"
        ) if "fracLut" in self.land_type else self.land_type
        for ltype in self.land_type:

            ltype_ds = []
            list_nc = [f for f in all_nc if self.mon_type in f and ltype in f]

            for nc in list_nc:
                ltype_ds.append(xr.open_dataset(nc))

            self.org_cell_objs[ltype] = xr.concat(ltype_ds, dim=DIM_TIME)

    def cal_area_weighted_cell(self):

        ds_area = copy.deepcopy(self.ds_area[VAR_AREA])
        for ltype in self.land_type:

            reindex_ltype = self.org_cell_objs[ltype][ltype].reindex_like(
                self.ds_area, method="nearest", tolerance=0.01
            )

            ds = reindex_ltype * ds_area * 1e-2
            ds = ds.rio.write_crs("epsg:4326", inplace=True)
            ds.coords["lon"] = (ds.coords["lon"] + 180) % 360 - 180
            ds = ds.sortby(ds.lon)
            ds = ds.rio.set_spatial_dims("lon", "lat", inplace=True)
            self.area_weighted_cell_obj[ltype] = ds

    def clip_2_roi(self, boundary_dict={}):
        land_types = sorted(list(self.org_cell_objs.keys()))

        for i, roi in enumerate(ROI_DICT.keys()):

            self.roi_ltype[roi] = {}

            ds_area = copy.deepcopy(self.ds_area[VAR_AREA])
            ds_area = ds_area.rio.write_crs("epsg:4326", inplace=True)
            ds_area.coords["lon"] = (ds_area.coords["lon"] + 180) % 360 - 180
            ds_area = ds_area.sortby(ds_area.lon)
            ds_area = ds_area.rio.set_spatial_dims("lon", "lat", inplace=True)
            self.roi_area[roi] = ds_area.rio.clip_box(
                minx=ROI_DICT[roi]["min_lon"],
                miny=ROI_DICT[roi]["min_lat"],
                maxx=ROI_DICT[roi]["max_lon"],
                maxy=ROI_DICT[roi]["max_lat"],
                # crs=self.crs,
            ).sum(["lat", "lon"])
            fig, ax = plt.subplots(figsize=(4.5, 4), layout="constrained")
            ax = plt.subplot(1, 1, 1)

            for ltype in land_types:

                ds = copy.deepcopy(self.area_weighted_cell_obj[ltype])
                ds = ds.rio.set_spatial_dims("lon", "lat", inplace=True)
                self.roi_ltype[roi][ltype] = (
                    ds.rio.clip_box(
                        minx=ROI_DICT[roi]["min_lon"],
                        miny=ROI_DICT[roi]["min_lat"],
                        maxx=ROI_DICT[roi]["max_lon"],
                        maxy=ROI_DICT[roi]["max_lat"],
                        # crs=self.crs,
                    ).sum(["lat", "lon"])
                    / self.roi_area[roi] * 1e2
                )
                self.roi_ltype[roi][ltype].sel(
                    time=self.roi_ltype[roi][ltype].time.dt.month.isin([12])
                ).plot(ax=ax, label=ltype)
            plt.title(roi, fontsize=18)
            axbox = ax.get_position()
            ax.legend(
                fontsize="small",
                loc="upper left",
                ncol=1,
                bbox_to_anchor=(1.0,0.72),
            )
            ax.set_xlabel("Year")
            ax.set_ylabel("(%)")
            plt.savefig(os.path.join("../fig/", self.model_name, f"{roi}-luc.png"))

    def plot_mk(self, sy, ey, ltype, cmap="RdBu_r"):
        ds = self.area_weighted_cell_obj[ltype]
        annual_ds = ds.sel(time=ds.time.dt.month.isin([12]))
        annual_ds = annual_ds.sel(
            time=annual_ds.time.dt.year.isin([i for i in range(sy, ey + 1)])
        )

        x = xr.DataArray(
            np.arange(len(annual_ds["time"])) + 1,
            dims="time",
            coords={"time": annual_ds["time"]},
        )
        s = mk.kendall_correlation(annual_ds, x, "time")

        fig = plt.figure(1, figsize=(30, 13))
        ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())
        ax.coastlines()
        ax.gridlines(
            xlocs=range(-180, 180, 40),
            ylocs=range(-80, 81, 20),
            draw_labels=True,
            linewidth=1,
            edgecolor="dimgrey",
        )
        my_cmap = matplotlib.cm.get_cmap(cmap)

        data = s
        title = f"Annual Trends of {ltype} from {sy}-{ey} using the Mann-Kendall method"
        data.plot.pcolormesh(
            ax=ax,
            cmap=my_cmap,
            cbar_kwargs={"label": "$m^{2}/year$"},
        )
        plt.title(title, fontsize=18)
        plt.savefig(os.path.join("../fig/", self.model_name, f"mk-{ltype}-{sy}-{ey}.png"))
    
    def plot_global_annual_trend(self):
        land_types = sorted(list(self.org_cell_objs.keys()))
        total_larea = self.ds_area.sum(dim=[DIM_LAT, DIM_LON])["areacella"]
        fig, ax = plt.subplots(figsize=(4.5, 4), layout="constrained")   
        for ltype in land_types:
            ds = self.area_weighted_cell_obj[ltype]
            annual_ds = ds.sel(time=ds.time.dt.month.isin([12]))
            annual_rate = (annual_ds.sum(dim=[DIM_LAT, DIM_LON])) / total_larea * 1e2
            annual_rate.plot(ax = ax, label=ltype, linewidth=1.5) 
            plt.title("Annual changes of global land-use types")
            ax.set_xlabel("Year")
            ax.set_ylabel("(%)") 
            ax.legend(
                fontsize="small",
                loc="upper left",
                ncol=1,
                bbox_to_anchor=(1.0,0.7),
            )
# %%
