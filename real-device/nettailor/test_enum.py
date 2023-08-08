from enum import Enum
class Color(Enum):
    red = 1
    green = 2
    blue = 3

c = Color.red
b = Color.blue

print(c == Color.red)
print(c == Color.green)