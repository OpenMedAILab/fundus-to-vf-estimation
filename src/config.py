"""Central path configuration for the GRAPE reproduction project.

Every path derives from this file's location. `config.py` lives in `src/`, so
ROOT is its parent directory (the repo root); all data / results / figures /
reports folders hang off ROOT. Data that is NOT shipped in the repo — the GRAPE
dataset, the external-validation images, the StyleGAN output, an optional CJK
font — is looked up via environment variables, each with a sensible default
under ROOT. Override any of them before running, e.g.:

    export GRAPE_ROOT="/data/GRAPE Dataset/GRAPE Dataset"
    export EXTERNAL_ROOT="/data/external"
    export GEN_DIR="/data/generated"
    export SIMHEI_TTF="/usr/share/fonts/simhei.ttf"
"""
import os

# This file is src/config.py -> ROOT is the repo root (parent of src/).
SRC  = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)


def _env(name, default):
    return os.environ.get(name, default)


# ---- External inputs (NOT shipped in the repo) ----
# GRAPE dataset root: holds CFPs/, ROI images/, Annotated Images/, json/, and the xlsx.
GRAPE_ROOT    = _env("GRAPE_ROOT",    os.path.join(ROOT, "GRAPE Dataset", "GRAPE Dataset"))
# External-validation root: holds CFP/ and VF/.
EXTERNAL_ROOT = _env("EXTERNAL_ROOT", os.path.join(ROOT, "external"))
# Flat folder of external CFP images (used only by the domain-shift FID/KID script).
EXT_CFP_IMGS  = _env("EXT_CFP_IMGS",  os.path.join(ROOT, "ext_cfp_imgs"))
# StyleGAN / generator output used for FID/KID and augmentation.
GEN_DIR       = _env("GEN_DIR",       os.path.join(ROOT, "generated"))
# Optional CJK font for matplotlib figures (falls back to default if absent).
SIMHEI_TTF    = _env("SIMHEI_TTF",    os.path.join(ROOT, "simhei.ttf"))

# ---- Derived GRAPE paths ----
GRAPE_CFP  = os.path.join(GRAPE_ROOT, "CFPs")
GRAPE_XLSX = os.path.join(GRAPE_ROOT, "VF and clinical information.xlsx")

# ---- Derived external paths ----
EXT_CFP = os.path.join(EXTERNAL_ROOT, "CFP")
EXT_VF  = os.path.join(EXTERNAL_ROOT, "VF")

# ---- In-repo data (dataset split indices, VF layout, augmentation images) ----
DATA      = os.path.join(ROOT, "data")
SYNTHETIC = os.path.join(DATA, "synthetic")
VF_LAYOUT = os.path.join(DATA, "vf_layout.npy")

# ---- In-repo outputs ----
RESULTS    = os.path.join(ROOT, "results")     # metrics / analysis JSON
CKPT       = os.path.join(RESULTS, "ckpt")     # per-run best.pth (ignored) + result.json
LOGS       = os.path.join(RESULTS, "logs")     # training logs
WANNAN_OUT = os.path.join(RESULTS, "wannan_extval")
FIGURES    = os.path.join(ROOT, "figures")     # paper figures (png)
SUPP_FIG   = os.path.join(FIGURES, "supp_fig")
REPORTS    = os.path.join(ROOT, "reports")     # generated markdown reports


def setup_cjk_font():
    """Register a CJK font for matplotlib if SIMHEI_TTF exists (else no-op)."""
    import matplotlib
    if os.path.exists(SIMHEI_TTF):
        import matplotlib.font_manager as fm
        fm.fontManager.addfont(SIMHEI_TTF)
        matplotlib.rcParams["font.sans-serif"] = ["SimHei"]
    matplotlib.rcParams["axes.unicode_minus"] = False
