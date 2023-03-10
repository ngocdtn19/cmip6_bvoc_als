#%%
import glob
import os
import math
import geopandas as gpd
import regionmask
import random


get_colors = lambda n: ["#%06x" % random.randint(0, 0xFFFFFF) for _ in range(n)]


DIM_TIME = "time"
DIM_LAT = "lat"
DIM_LON = "lon"

VAR_AREA = "areacella"
VAR_ISOP = "emiisop"
VAR_SFTLF = "sftlf"

VAR_MONTH_RATE = f"{VAR_ISOP}_month"
VAR_ISOP_AREA = f"{VAR_ISOP}_{VAR_AREA}"

ISOP_2_C = 60 / 68
# POM_2_OC = 1.4                           #in case converting from mass (g) to carbon mass (gC), divide 1.4
DAY_RATE = 60 * 60 * 24

KG_2_G = 1e3
KG_2_TG = 1e-9
KG_2_PG = 1e-12
K_2_C = 273.15

MG_2_G = 1e-6
MG_2_TG = 1e-18
# DEG_2_M2 = 55500**2


LIST_REGION = ["SEA", "EAS"]
REGION = regionmask.defined_regions.giorgi
LIST_REGION = [REGION.regions[i].abbrev for i in REGION.regions.keys()]
LIST_COLOR = get_colors(len(LIST_REGION))




ROI_COLORS = {roi: color for roi, color in zip(LIST_REGION, LIST_COLOR)}

ROI_DICT = {
    "NAm": {"min_lat": 13, "max_lat": 75, "min_lon": -170, "max_lon": -40},
    "SAm": {"min_lat": -60, "max_lat": 13, "min_lon": -90, "max_lon": -35},
    "Eu": {"min_lat": 37, "max_lat": 75, "min_lon": -15, "max_lon": 50},
    "NAf": {"min_lat": 15, "max_lat": 37, "min_lon": -20, "max_lon": 65},
    "EqAf": {"min_lat": -15, "max_lat": 15, "min_lon": -20, "max_lon": 55},
    "Saf": {"min_lat": -35, "max_lat": -15, "min_lon": -20, "max_lon": 55},
    "Rus": {"min_lat": 37, "max_lat": 75, "min_lon": 50, "max_lon": 179},
    "SA": {"min_lat": -10, "max_lat": 37, "min_lon": 65, "max_lon": 170},
    "Aus": {"min_lat": -50, "max_lat": -10, "min_lon": 110, "max_lon": 179},
}

VIZ_OPT = {
    "emiisop": {
        "map_unit": "[$gC/m^{2}/year$]",  # cmap1="RdBu_r", cmap2="RdPu"
        "map_vmin": 0,
        "map_vmax": 40,
        "map_levels": 17,
        "line_bar_unit": "[TgC]",
        "line_ylim": [350, 650],
        "bar_ylim": [0, 670],
        "holv_unit": "[$gC/m^{2}/month$]",
    },
    "emibvoc": {
        "map_unit": "[$g/m^{2}/year$]",  # cmap1="RdBu_r", cmap2="YlGn"
        "map_vmin": 0,
        "map_vmax": 50,
        "map_levels": 17,
        "line_bar_unit": "[Tg]",
        "line_ylim": [400, 1200],
        "bar_ylim": [0, 1200],
    },
    "gpp": {
        "map_unit": "[$gC/m^{2}/year$]",
        "map_vmin": 0.01,
        "map_vmax": 1000,
        "map_levels": 11,
        "line_bar_unit": "[PgC]",
        "line_ylim": [90, 150],
        "bar_ylim": [0, 150],
    },
    "npp": {
        "map_unit": "[$gC/m^{2}/year$]",
        "map_vmin": 0.01,
        "map_vmax": 1000,
        "line_bar_unit": "[PgC]",
        "line_ylim": [30, 150],
        "bar_ylim": [0, 150],
    },
    "pr": {
        "map_unit": "[mm/day]",  # cmap1="RdBu", cmap2="YlGnBu", bg_color="#67001f" for change map
        "map_vmin": 0,
        "map_vmax": 14,
        "map_levels": 17,
        "line_bar_unit": "[mm/day]",
        "line_ylim": [30, 150],
    },
    "rsds": {
        "map_unit": "[$W/m^{2}$]",  # cmap1="RdBu_r", cmap2="YlOrRd", bg_color="#053061" for change map
        "map_vmin": 0,
        "map_vmax": 0,
        "map_levels": 0,
        "line_bar_unit": "[$W/m^{2}$]",
        "line_ylim": 0,
        "bar_ylim": 0,
    },
    "tas": {
        "map_unit": "[$^{\circ}C$]",  # cmap1="RdBu_r", cmap2="Spectral_r", bg_color="#5e4fa2" for abs map, bg_color="#053061" for change map
        "map_vmin": -30,
        "map_vmax": 40,
        "map_levels": 13,
        "line_bar_unit": "[$^{\circ}C$]",
        "line_ylim": 0,
    },
    "emioa": {
        "map_unit": "[$g/m^{2}/year$]",  # cmap1="RdBu_r", cmap2="OrRd"
        "map_vmin": 0,  # bg_color="#053061", vmin=-0.15, vmax=0.15 for change map
        "map_vmax": 0.3,
        "map_levels": 17,
        "line_bar_unit": "[Tg]",
        "line_ylim": [0, 0],
        "bar_ylim": [0, 225],
        # "holv_unit": "[$gC/m^{2}/month$]",
    },
    "chepsoa": {
        "map_unit": "[$g/m^{2}/year$]",  # cmap1="RdBu_r", cmap2="OrRd"
        "map_vmin": 0,
        "map_vmax": 0.2,
        "map_levels": 17,
        "line_bar_unit": "[Tg]",
        "line_ylim": [0, 0],
        "bar_ylim": [0, 0],
        # "holv_unit": "[$gC/m^{2}/month$]",
    },
    "emiotherbvocs": {
        "map_unit": "[$g/m^{2}/year$]",
        "map_vmin": 0,
        "map_vmax": 0,
        "map_levels": 0,
        "line_bar_unit": "[Tg]",
        "line_ylim": [0, 0],
        "bar_ylim": [0, 0],
        "holv_unit": "[$gC/m^{2}/month$]",
    },
    "emipoa": {
        "map_unit": "[$g/m^{2}/year$]",  # cmap1="RdBu_r", cmap2="OrRd"
        "map_vmin": 0,
        "map_vmax": 0.2,
        "map_levels": 17,
        "line_bar_unit": "[Tg]",
        "line_ylim": [0, 0],
        "bar_ylim": [0, 0],
        "holv_unit": "[$gC/m^{2}/year$]",
    },
}

COLOR_STACK_BAR = [
    "darkblue",
    "blue",
    "deepskyblue",
    "aqua",
    "mediumspringgreen",
    "greenyellow",
    "yellow",
    "orange",
    "red",
]

VISIT_DICT_PATH = {
    "VISIT_ORG": "../data/VISIT/visit_20160105_BVOCisprn.nc",
    "VISIT_CASE1": "../data/VISIT/emiisop_AERmon_VISIT-org_historical_r1i1p1f1_gn_185001-201412.nc",
    "VISIT_CASE2": "../data/VISIT/emiisop_AERmon_VISIT-EFtropic_historical_r1i1p1f1_gn_185001-201412.nc",
    "VISIT_CASE3": "../data/VISIT/emiisop_AERmon_VISIT-CO2inhi_historical_r1i1p1f1_gn_185001-201412.nc",
}

TOPDOWN_DIR = "../data/topdown"
TOPDOWN_VAR = "EMworldC5H8"
LIST_TOPDOWN_ANNUAL_FILES = glob.glob(os.path.join(TOPDOWN_DIR, "*.nc"))
CONCAT_TOPDOWN_PATH = "../data/topdown/concat_topdown.nc"


VISIT_LAT_FILE = "../data/visit_latlon/visit_lat.npy"
VISIT_LONG_FILE = "../data/visit_latlon/visit_long.npy"

WORLD_SHP = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))

# %%
