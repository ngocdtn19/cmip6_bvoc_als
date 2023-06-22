import numpy as np
import xarray as xr
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from cartopy.crs import PlateCarree

from xeofs.xarray import EOF


a = Var("emiisop")
clm = (
    a.multi_models["CESM2-WACCM"]
    .monthly_per_area_unit.groupby("time.month")
    .mean(dim="time")
)
anm = a.multi_models["CESM2-WACCM"].monthly_per_area_unit.groupby("time.month") - clm
anmS = anm.resample(time="QS-DEC").mean(skipna=True)


eofs = []
pcs = []
expvar = []

for m in ["Monthly", 12, 3, 6, 9]:
    if m == "Monthly":
        ds = anm
    else:
        ds = anmS.sel(time=anmS.time.dt.month.isin([m]))

    model = EOF(ds, dim=["time"], n_modes=6, norm=False, weights="coslat")
    model.solve()
    eofs.append(model.eofs())
    pcs.append(model.pcs())
    expvar.append(model.explained_variance_ratio() * 100)

    proj = PlateCarree(central_longitude=0)
    fig = plt.figure(figsize=(20, 15))
    gs = GridSpec(3, 2, width_ratios=[1.2, 2])
    ax0 = [fig.add_subplot(gs[i, 0]) for i in range(3)]
    ax1 = [fig.add_subplot(gs[i, 1], projection=proj) for i in range(3)]

    for i, (a0, a1) in enumerate(zip(ax0, ax1)):
        pcs[0].sel(mode=i + 1).plot(ax=a0, color="darkred", lw=1)
        a1.coastlines(color=".5")
        a1.gridlines(
            xlocs=range(-180, 180, 40),
            ylocs=range(-80, 81, 20),
            draw_labels=True,
            linewidth=1,
            edgecolor="dimgrey",
        )
        eofs[0].sel(mode=i + 1).plot.pcolormesh(
            ax=a1,
            vmin=-0.06,
            vmax=0.06,
            cmap="RdBu_r",
            extend="both",
            cbar_kwargs={"label": "[$gC/m^{2}/month$]"},
        )
        a0.set_xlabel("Year")
        a0.set_ylabel("[$gC/m^{2}/month$]")
        a0.set_title(
            f"mode = {i+1} ({np.format_float_positional(expvar[0].sel(mode=i+1).values, precision=2)}%)",
            loc="center",
            weight="bold",
        )
        a1.set_title(f"mode = {i+1}", loc="center", weight="bold")

    plt.tight_layout()
