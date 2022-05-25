# imports rotation sweep data file and computes offset

# to do: propagate and calculate new errors on field values
# create general script to take in a file and perform coord & field transformation

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

file = "/Users/alexandraklipfel/Desktop/senior_thesis/prelim_mapper_arm_data/sweep_rotation_out_2-8.txt"
dat = np.loadtxt(file)
print(type(dat))
print(dat.shape)
length = dat.shape[0]
print(length)

# set reference values
T0 = 200
R0 = 0 #don't know
V0 = 0 #don't know

# grab columns
#position
T = dat[:, 0] #1st column
R = dat[:, 1] #2nd column
V = dat[:, 2] #3rd column
#field
BxProbe = dat[:, 3] #4th column
BxProbeErr = dat[:, 4] #5th column
ByProbe = dat[:, 5] #6th column
ByProbeErr = dat[:, 6] #7th column
BzProbe = dat[:, 7] #4th column
BzProbeErr = dat[:, 8] #5th column

rvals = (22.7 / 30000) * (T - T0)
phivals = (np.pi / 8000) * (R - R0)
zvals = (38.61 / 50000) * (V - V0)

# new data arrays in lab frame
#position
x = np.zeros(length)
y =np.zeros(length)
z = np.zeros(length)
#field
Bx = np.zeros(length)
BxErr = np.zeros(length)
By = np.zeros(length)
ByErr = np.zeros(length)
Bz = np.zeros(length)
BzErr = np.zeros(length)

for i in range(length):
    # transform to probe position to cartesian coords
    x[i] = rvals[i] * np.cos(phivals[i])
    y[i] = rvals[i] * np.sin(phivals[i])
    z[i] = zvals[i]
    # transform B field with a rotation in probe xz plane
    Bx[i] = BzProbe[i] * np.cos(phivals[i]) + BxProbe[i] * np.sin(phivals[i])
    By[i] = BzProbe[i] * np.sin(phivals[i]) - BxProbe[i] * np.cos(phivals[i])
    Bz[i] = -ByProbe[i]

# plot the x, y, z components vs. phi
plt.scatter(phivals, Bx)
plt.savefig("Bxscatter.png")
plt.show()

plt.scatter(phivals, By)
plt.savefig("Byscatter.png")
plt.show()

plt.scatter(phivals, Bz)
plt.savefig("Bzscatter.png")
plt.show()

plt.quiver(x, y, BzProbe, BxProbe)
plt.savefig("vf_probe")
plt.show()

plt.quiver(x, y, Bx, By)
plt.savefig("vf_lab")
plt.show()

# to compute offsets: fit sinusiod to phi vs. BxProbe and phi vs. PzProbe data. The vertical shift means a constant offset.

# function to fit
def curve(x, A, B, C, D):
    f = A * np.sin(B * x + C) + D
    return f
    
popt, pcov = curve_fit(curve, phivals, BxProbe)
print(popt)

plt.plot(phivals, curve(phivals, *popt))
plt.scatter(phivals, BxProbe, c="black")
plt.savefig("fitted_BxProbe.png")
plt.show()

popt2, pcov2 = curve_fit(curve, phivals, BzProbe)
print(popt2)

plt.plot(phivals, curve(phivals, *popt2))
plt.scatter(phivals, BzProbe, c="black")
plt.savefig("fitted_BzProbe.png")
plt.show()

# resulting offsets:
#x: -0.0255
#z: 0.252

print(length)


