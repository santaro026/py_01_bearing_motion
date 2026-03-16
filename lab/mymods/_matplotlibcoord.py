"""
Created on Sun Dec 07 11:04:25 2025
@author: santaro



"""

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransform


def printf(var):
    name = None
    for k, v in globals().items():
    # for k, v in locals().items():
        if id(v) == id(var):
            name = k
    print(f"name: {name}\ntype: {type(var)}\nvalue: {var}")



fig, ax = plt.subplots(figsize=(8, 8), dpi=100)

ax.set(xlim=(0, 1), ylim=(0, 4), xticks=np.arange(0, 1.1, 0.5), yticks=np.arange(0, 4.1, 0.5))
ax.grid()

x, y = 0.5, 0.5

ax.text(x, y, "0505 transAxes", transform=ax.transAxes)
ax.text(x, y, "0505 transData", transform=ax.transData)



a = ax.transAxes.transform((x, y))
printf(a)

b = ax.transAxes.inverted().transform((a[0], a[1]))
printf(b)


x_inch, y_inch = 1, 1

c = mtransform.ScaledTranslation(x_inch, y_inch, fig.dpi_scale_trans)
printf(c)

d = mtransform.ScaledTranslation(x_inch, y_inch, fig.dpi_scale_trans).transform((0, 0))
printf(d)


ax.text(0, 0, "test c", transform=ax.transAxes+c)
# ax.text(0, 0, "test d", transform=ax.transAxes+d)



plt.show()


