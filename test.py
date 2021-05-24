import sys
import pandas as pd
import numpy as np



a = np.array([3, 5, 2, 1, 4, ])
x = np.partition(a, 3.5)

print(x)