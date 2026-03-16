"""
Created on Sat Oct 11 13:44:18 2025
@author: honda-shin



"""
#%%
from sympy import *
from sympy import solve
# from sympy.abc import x, y, z
from sympy import init_printing
init_printing()

x, y, z = symbols('x y z')
a, b, r, theta, alpha = symbols('a b r theta alpha')

eq1 = y**2/a**2 + z**2/b**2 - r**2
eq2 = z - y*tan(alpha-theta)
res = solve([eq1, eq2], [y, z], dict=False)

M = Matrix([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
p1 = Matrix([res[1][0], res[1][1]])

res2 = M * p1

simplify(res2)
res2


#%%

a = 20
s = "{:.2f}"

# s.format(a)

print('test')
print(f"{a:.2f}")

#%%

from enum import Enum, auto
import json

class Shape(Enum):
    SQUARE = "square"
    RECTANGLE = "rectangle"
    LANDSCAPE = "landscape"

json_data = '{"shape": "rectangle", "size": "10"}'
data = json.loads(json_data)

# shape = Shape(data["shape"])
# print(shape)
# print(shape.name)
# print(shape.value)


# a = Shape.SQUARE
# print(a)


