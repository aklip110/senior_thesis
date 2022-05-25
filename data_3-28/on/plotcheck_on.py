import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
#plt.rcParams["figure.figsize"] = 6,4

file = "/Users/alexandraklipfel/Desktop/senior_thesis/data_3-29/on/output_b0_on.txt"
dat = np.loadtxt(file)
print("Data Shape: ", dat.shape)
length = dat.shape[0]
print("Number of Data Points: ", length)

# grab columns
#position
T = dat[:, 0] #1st column
R = dat[:, 1] #2nd column
V = dat[:, 2] #3rd column
#field
BxProbe = dat[:, 3] #4th column
#BxProbe = BxProbe - offsetX
BxProbeErr = dat[:, 4] #5th column
ByProbe = dat[:, 5] #6th column
ByProbeErr = dat[:, 6] #7th column
BzProbe = dat[:, 7] #4th column
#BzProbe = BzProbe - offset_Z
BzProbeErr = dat[:, 8] #5th column

y = []
bx = []
by = []

for i in range(length):
    if (dat[i, 2] == -5000) and (dat[i, 1] == 3900):
        y = y + [dat[i,0]]
        bx = bx + [dat[i,3]]
        by = by + [dat[i,7]]
        
plt.scatter(y, bx, color="red")

plt.xlabel("y")
plt.ylabel("Bx")
plt.title("Bx vs y: on")
plt.savefig("Bx_y-check.png", dpi=1000)
plt.show()

plt.scatter(y, by, color="red")

plt.xlabel("y")
plt.ylabel("By")
plt.title("By vs y: on")
plt.savefig("By_y-check.png", dpi=1000)
plt.show()

######################

x = []
bx = []
by = []

for i in range(length):
    if (dat[i, 2] == -5000) and (dat[i, 1] == -200):
        x = x + [dat[i,0]]
        bx = bx + [dat[i,7]]
        by = by + [-dat[i,3]]
        
plt.scatter(x, bx, color="red")

plt.xlabel("x")
plt.ylabel("Bx")
plt.title("Bx vs x: on")
plt.savefig("Bx_x-check.png", dpi=1000)
plt.show()

plt.scatter(x, by, color="red")

plt.xlabel("x")
plt.ylabel("By")
plt.title("By vs x: on")
plt.savefig("By_x-check.png", dpi=1000)
plt.show()
