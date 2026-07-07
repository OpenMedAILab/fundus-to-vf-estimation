"""Central path configuration for the GRAPE reproduction project.

Every path derives from this file's own location (ROOT), so the repo runs
from wherever it's cloned. Data that is NOT shipped in the repo — the GRAPE
dataset, the external-validation images, the StyleGAN output, an optional CJK
font — is looked up via environment variables, each with a sensible default
under ROOT. Override any of them before running, e.g.:

    export GRAPE_ROOT="/data/GRAPE Dataset/GRAPE Dataset"
    export EXTERNAL_ROOT="/data/external"
    export GEN_DIR="/data/generated"
    export SIMHEI_TTF="/usr/share/fonts/simhei.ttf"
"""
import os

# Repo root = the directory containing this file.
ROOT = os.path.dirname(os.path.abspath(__file__))


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

# ---- In-repo outputs ----
DATA      = os.path.join(ROOT, "data")
CKPT      = os.path.join(ROOT, "ckpt")
SYNTHETIC = os.path.join(ROOT, "synthetic")


def setup_cjk_font():
    """Register a CJK font for matplotlib if SIMHEI_TTF exists (else no-op)."""
    import matplotlib
    if os.path.exists(SIMHEI_TTF):
        import matplotlib.font_manager as fm
        fm.fontManager.addfont(SIMHEI_TTF)
        matplotlib.rcParams["font.sans-serif"] = ["SimHei"]
    matplotlib.rcParams["axes.unicode_minus"] = False
