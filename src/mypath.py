import glob
import os
from const import *

DATA_DIR = "/mnt/dg3/ngoc/cmip6_bvoc_als/data"
VAR_DIR = os.path.join(DATA_DIR, "var")
AXL_DIR = os.path.join(DATA_DIR, "axl")
LIST_ATTR = [attr.split("\\")[-1] for attr in glob.glob(os.path.join(VAR_DIR, "*"))]

ISOP_LIST = glob.glob(os.path.join(VAR_DIR, "emiisop", "*.nc"))
BVOC_LIST = glob.glob(os.path.join(VAR_DIR, "emibvoc", "*.nc"))

AREA_LIST = glob.glob(os.path.join(AXL_DIR, VAR_AREA, "*.nc"))
SFLTF_LIST = glob.glob(os.path.join(AXL_DIR, VAR_SFTLF, "*.nc"))

LAND_DIR = os.path.join(DATA_DIR, "land")




