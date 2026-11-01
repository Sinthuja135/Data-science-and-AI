"""
Shared plot style.
"""
import matplotlib as mpl

# The project's fixed colour palette -- named constants so every figure-drawing script references e.g.
# S.TEAL instead of retyping the hex code, and the palette can be changed in one place.
NAVY, TEAL, AMBER, INK = "#1F3864", "#0F7C7C", "#F2A900", "#22303F"

def apply():
    """Set matplotlib's global style for the rest of the process. Call this once near the top of a script,
    before any figure is created"""
    mpl.rcParams.update({
        "figure.dpi": 130, "savefig.dpi": 200, "savefig.bbox": "tight",   # crisp on-screen, tight-cropped when saved
        "font.family": "DejaVu Sans", "font.size": 9,                    # small, readable, no missing-font fallback
        "axes.edgecolor": "#C9D1DA", "axes.labelcolor": INK, "axes.titlecolor": INK,
        "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.titlelocation": "left",  # left-aligned bold title
        "axes.spines.top": False, "axes.spines.right": False,            # no boxed-in look -- open top/right
        "axes.grid": True, "grid.color": "#E6EAEE", "grid.linewidth": 0.7, "axes.axisbelow": True,  # grid behind data
        "xtick.color": INK, "ytick.color": INK, "legend.frameon": False,  # no legend border
    })
