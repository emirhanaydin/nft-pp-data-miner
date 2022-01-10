import math

import numpy
import numpy as np
from numpy.typing import ArrayLike


def mean_filter(x: ArrayLike) -> ArrayLike:
    avg = numpy.nanmean(x)
    return np.array([avg if math.isnan(v) else v for v in x])
