"""
    Shared third-party imports for this package
    Import from here instead of importing these libraries directly:
    from .common import np, pd, plt
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

__all__ = ["warnings", "np", "pd", "plt"]
