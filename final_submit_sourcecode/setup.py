import sys
from cx_Freeze import setup, Executable

# Dependencies are automatically detected, but it might need fine tuning.
build_exe_options = {
    "packages": ["os"],
    "excludes": ["tkinter"],
    "includes": [
        "torch",
        "torch._VF",
        "torch.distributions",
        "torch.distributions.constraints",
        "scipy",
        "scipy.sparse.csgraph._validation",
        "scipy.ndimage._ni_support",
    ],
}

# GUI applications require a different base on Windows (the default is for a
# console application).
base = "Console"

exe = Executable("LME_SAGI_Stage2.py", base=base)

setup(
    name="lme_Stage2",
    version="0.1",
    options={"build_exe": build_exe_options},
    description="LME Stabilo 2020 challenge submission.",
    executables=[exe],
)
