from pathlib import Path
print("__file__:", __file__)
print("Path(__file__).resolve():", Path(__file__).resolve())
print("parents[0]:", Path(__file__).resolve().parents[0])
print("parents[1]:", Path(__file__).resolve().parents[1])
print("parents[2]:", Path(__file__).resolve().parents[2])