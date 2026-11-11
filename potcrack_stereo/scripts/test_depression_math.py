#!/usr/bin/env python3
import math
import numpy as np

# Test ray intersection depth for a 5 cm depression below road
fx, fy = 1050.0, 1050.0
cx, cy = 960.0, 540.0
h_cam = 1.25
pitch_rad = math.radians(8.0)
ny = -math.cos(pitch_rad)
nz = -math.sin(pitch_rad)
norm_vec = np.array([0.0, ny, nz])
d = -float(norm_vec[1] * h_cam)

# Pothole center at Z = 4.5m, X = 0.1m
# Point on road plane:
u = 983 # pixel at X=0.1, Z=4.5
# ray:
# road Y at Z=4.5:
y_road = -(norm_vec[2]*4.5 + d) / norm_vec[1]
v = int(fy * (y_road / 4.5) + cy)

rx = (u - cx) / fx
ry = (v - cy) / fy
rz = 1.0
ray = np.array([rx, ry, rz])
denom = np.dot(norm_vec, ray)

Z_road = -d / denom
P_road = Z_road * ray
print(f"Road point at (u={u}, v={v}): P_road = {P_road}")
print(f"Plane residual at P_road: {np.dot(norm_vec, P_road) + d}")

# Now suppose the cavity floor is D_perp = 5.0 cm = 0.05 m below the road plane.
# The cavity floor is on plane: norm_vec . P + (d + D_perp) = 0?
# Wait! Let's check the sign of D_perp!
# norm_vec = [0, -0.99, -0.14] points UP (Y is down).
# Below the road means moving in direction -norm_vec (downward, Y increases).
# Let P_below = P_road - 0.05 * norm_vec.
P_below = P_road - 0.05 * norm_vec
print(f"P_below = {P_below}")
print(f"norm_vec . P_below + d = {np.dot(norm_vec, P_below) + d}")
# Notice: norm_vec . (P_road - 0.05*norm_vec) + d = -0.05 * 1.0 = -0.05!
# In PotholeGeometryEngine:
# depression = -(a*X + b*Y + c*Z + d)
# So depression = -(-0.05) = +0.05 m = +5.0 cm!
# Exactly +5.0 cm!

# Now what is the ray intersection with the cavity floor?
# The ray from camera is P = Z_cavity * ray.
# We want P to lie on the cavity floor plane: norm_vec . P + d = -D_perp
# norm_vec . (Z_cavity * ray) + d = -D_perp
# Z_cavity * (norm_vec . ray) = -d - D_perp
# Z_cavity = (-d - D_perp) / denom = Z_road - D_perp / denom
# Since denom is negative, -D_perp / denom is POSITIVE!
D_perp = 0.05
Z_cavity = (-d - D_perp) / denom
P_cavity_ray = Z_cavity * ray
depr_measured = -(np.dot(norm_vec, P_cavity_ray) + d)
print(f"\nAlong ray:")
print(f"Z_road = {Z_road:.4f} m")
print(f"Z_cavity = {Z_cavity:.4f} m (delta Z = {(Z_cavity - Z_road)*100:.2f} cm)")
print(f"Depression measured = {depr_measured*100:.2f} cm")
