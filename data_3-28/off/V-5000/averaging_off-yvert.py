#NOTE: variable names don't match agreed-upon lab coordinate names. bz==vertical, bx==along B0, by==orthogonal. In lab coords, z and y are swapped so that y is vertical. The plot titles and file names reflect this.

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
#plt.rcParams["figure.figsize"] = 6,4

file = "/Users/alexandraklipfel/Desktop/senior_thesis/data_3-29/off/output_b0_off.txt"
dat = np.loadtxt(file)
print("Data Shape: ", dat.shape)
length = dat.shape[0]
print("Number of Data Points: ", length)

T = dat[:, 0] #1st column
R = dat[:, 1] #2nd column
V = dat[:, 2] #3rd column

print(np.unique(T))
print(np.unique(R))
print(np.unique(V))
numT = len(np.unique(T))
numR = len(np.unique(R))
numV = len(np.unique(V))
print(numT)
print(numR)
print(numV)

numPoints = numT * numR * numV
print(numPoints)

avgd_dat = np.zeros(9)

for t in np.unique(T):
    for r in np.unique(R):
        for v in np.unique(V):
            vals_to_avg = np.zeros(9)
            for i in range(length):
                if (dat[i, 0] == t) and (dat[i, 1] == r) and (dat[i, 2] == v):
                    vals_to_avg = np.vstack([vals_to_avg, dat[i,0:9]])
            vals_to_avg = vals_to_avg[1:, :]
            print(vals_to_avg)
            averaged = np.mean(vals_to_avg, axis=0)
            avgd_dat = np.vstack([avgd_dat, averaged])
            
avgd_dat = avgd_dat[1:, :]
print(avgd_dat)

np.savetxt("averaged_off.txt", avgd_dat, delimiter=" ")

# make y plots---these are actually labeled "z" because i'm switching to the lab coordinates that Umit's using
###############################################
# y axis data:
y = []
bx = []
by = []
bz = []
for i in range(length):
    if (dat[i, 2] == -5000) and (dat[i, 1] == 3900) :
        y += [dat[i,0]]
        bx += [dat[i,3]]
        by += [dat[i,7]]
        bz += [-dat[i, 5]]
    if (dat[i, 2] == -5000) and (dat[i, 1] == -4300) :
        y += [-dat[i,0]]
        bx += [-dat[i,3]]
        by += [-dat[i,7]]
        bz += [-dat[i, 5]]
   
y_avgd = []
bx_avgd = []
by_avgd = []
bz_avgd = []

y_avgdL = []
bx_avgdL = []
by_avgdL = []
bz_avgdL = []
for i in range(len(avgd_dat)):
    if (avgd_dat[i, 2] == -5000) and (avgd_dat[i, 1] == 3900):
        y_avgd += [avgd_dat[i,0]]
        bx_avgd += [avgd_dat[i,3]]
        by_avgd += [avgd_dat[i,7]]
        bz_avgd += [-avgd_dat[i,5]]
    if (avgd_dat[i, 2] == -5000) and (avgd_dat[i, 1] == -4300):
        y_avgdL += [-avgd_dat[i,0]]
        bx_avgdL += [-avgd_dat[i,3]]
        by_avgdL += [-avgd_dat[i,7]]
        bz_avgdL += [-avgd_dat[i,5]]
        
#along B0
plt.scatter(y, bx, color="yellow")
plt.scatter(y_avgd, bx_avgd, color="red")
plt.scatter(y_avgdL, bx_avgdL, color="blue")

plt.xlabel("z")
plt.ylabel("Bx")
plt.title("Bx (along B0) vs z: off")
plt.savefig("Bx_z-avg-off.png", dpi=1000)
plt.show()

#orthogonal
#relabeled!
plt.scatter(y, by, color="yellow")
plt.scatter(y_avgd, by_avgd, color="red")
plt.scatter(y_avgdL, by_avgdL, color="blue")

plt.xlabel("z")
plt.ylabel("Bz")
plt.title("Bz vs z: off")
plt.savefig("Bz_z-avg-off.png", dpi=1000)
plt.show()
    
#vertical
#relaveled!
plt.scatter(y, bz, color="yellow")
plt.scatter(y_avgd, bz_avgd, color="red")
plt.scatter(y_avgdL, bz_avgdL, color="blue")

plt.xlabel("z")
plt.ylabel("By")
plt.title("By (vertical) vs z: off")
plt.savefig("By_z-avg-off.png", dpi=1000)
plt.show()


###############################################
# x axis data:
x = []
bx = []
by = []
bz = []
for i in range(length):
    if (dat[i, 2] == -5000) and (dat[i, 1] == -200) :
        x += [dat[i,0]]
        bx += [dat[i,7]]
        by += [-dat[i,3]]
        bz += [-dat[i, 5]]
    if (dat[i, 2] == -5000) and (dat[i, 1] == 7900) :
        x += [-dat[i,0]]
        bx += [-dat[i,7]]
        by += [dat[i,3]]
        bz += [-dat[i, 5]]
   
x_avgd = []
bx_avgd = []
by_avgd = []
bz_avgd = []

x_avgdL = []
bx_avgdL = []
by_avgdL = []
bz_avgdL = []
for i in range(len(avgd_dat)):
    if (avgd_dat[i, 2] == -5000) and (avgd_dat[i, 1] == -200):
        x_avgd += [avgd_dat[i,0]]
        bx_avgd += [avgd_dat[i,7]]
        by_avgd += [-avgd_dat[i,3]]
        bz_avgd += [-avgd_dat[i,5]]
    if (avgd_dat[i, 2] == -5000) and (avgd_dat[i, 1] == 7900):
        x_avgdL += [-avgd_dat[i,0]]
        bx_avgdL += [-avgd_dat[i,7]]
        by_avgdL += [avgd_dat[i,3]]
        bz_avgdL += [-avgd_dat[i,5]]
        
#along B0
plt.scatter(x, bx, color="yellow")
plt.scatter(x_avgd, bx_avgd, color="red")
plt.scatter(x_avgdL, bx_avgdL, color="blue")

plt.xlabel("x")
plt.ylabel("Bx")
plt.title("Bx (along B0) vs x: off")
plt.savefig("Bx_x-avg-off.png", dpi=1000)
plt.show()

#orthogonal
plt.scatter(x, by, color="yellow")
plt.scatter(x_avgd, by_avgd, color="red")
plt.scatter(x_avgdL, by_avgdL, color="blue")

plt.xlabel("x")
plt.ylabel("Bz")
plt.title("Bz vs x: off")
plt.savefig("Bz_x-avg-off.png", dpi=1000)
plt.show()
    
#vertical
plt.scatter(x, bz, color="yellow")
plt.scatter(x_avgd, bz_avgd, color="red")
plt.scatter(x_avgdL, bz_avgdL, color="blue")

plt.xlabel("x")
plt.ylabel("By")
plt.title("By (vertical) vs x: off")
plt.savefig("By_x-avg-off.png", dpi=1000)
plt.show()
