import glob
import os
from const import *


DATA_SERVER = f"/mnt/dg3/ngoc/cmip6_bvoc_als/data/"
DATA_LOCAL = "../data/"
RES_DIR = f"{DATA_SERVER}/1x125deg/processed_data/annual_per_area_unit/"

DATA_DIR = DATA_LOCAL
if os.path.exists(DATA_SERVER):
    DATA_DIR = DATA_SERVER

VAR_DIR = os.path.join(DATA_DIR, "var")
AXL_DIR = os.path.join(DATA_DIR, "axl")
LIST_ATTR = [attr.split("\\")[-1] for attr in glob.glob(os.path.join(VAR_DIR, "*"))]

ISOP_LIST = glob.glob(os.path.join(VAR_DIR, "emiisop", "*.nc"))
BVOC_LIST = glob.glob(os.path.join(VAR_DIR, "emibvoc", "*.nc"))

AREA_LIST = glob.glob(os.path.join(AXL_DIR, VAR_AREA, "*.nc"))
SFLTF_LIST = glob.glob(os.path.join(AXL_DIR, VAR_SFTLF, "*.nc"))

LAND_DIR = os.path.join(DATA_DIR, "land")

TOPDOWN_DIR = os.path.join(DATA_DIR, "topdown")

LIST_TOPDOWN_ANNUAL_FILES = os.path.join(TOPDOWN_DIR, "*.nc")
CONCAT_TOPDOWN_PATH = os.path.join(TOPDOWN_DIR, "concat_topdown.nc")


VISIT_LAT_FILE = os.path.join(DATA_DIR, "visit_latlon", "visit_lat.npy")
VISIT_LONG_FILE = os.path.join(DATA_DIR, "visit_latlon", "visit_long.npy")
