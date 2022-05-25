#NOTE: here, z is vertical axis
#V=-34500 plots
#for use of the 3/30 dataset

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
#plt.rcParams["figure.figsize"] = 6,4

file = "/Users/alexandraklipfel/Desktop/senior_thesis/data_3-30/data_3-30-noedits.txt"
Alldat = np.loadtxt(file)
print("Data Shape: ", Alldat.shape)
length = Alldat.shape[0]
print("Number of Data Points: ", length)

#based on the raw data time values when it switches back to the top halfway thru
#tHalf = ((12603 - 12530) / 2) + 12530
#print("t half: ", tHalf)

#will be array of off data used throughout
Offdat = np.zeros(13)

#grab off data prior to t1/2
for i in range(length):
    if (Alldat[i, 9] == 0):
        Offdat = np.vstack([Offdat, Alldat[i,:]])

#drop zero row
dat = Offdat[1:, :]
length = dat.shape[0]
print("length: ", length)

T = dat[:, 0] #1st column
R = dat[:, 1] #2nd column
V = dat[:, 2] #3rd column

T0 = 6400

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
            #print(vals_to_avg)
            averaged = np.mean(vals_to_avg, axis=0)
            avgd_dat = np.vstack([avgd_dat, averaged])
            
avgd_dat = avgd_dat[1:, :]
#print(avgd_dat)

np.savetxt("averaged_off-34500.txt", avgd_dat, delimiter=" ")

###############################################
# y axis data:
y = []
bx = []
by = []
bz = []
for i in range(length):
    if (dat[i, 2] == -34500) and (dat[i, 1] == -50) :
        y += [(22.7 / 30000) * (dat[i,0] - T0)]
        bx += [dat[i,3]]
        by += [dat[i,7]]
        bz += [-dat[i, 5]]
    if (dat[i, 2] == -34500) and (dat[i, 1] == 7950) :
        y += [(22.7 / 30000) * (-(dat[i,0] - T0))]
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
    if (avgd_dat[i, 2] == -34500) and (avgd_dat[i, 1] == -50):
        y_avgd += [(22.7 / 30000) * (avgd_dat[i,0] - T0)]
        bx_avgd += [avgd_dat[i,3]]
        by_avgd += [avgd_dat[i,7]]
        bz_avgd += [-avgd_dat[i,5]]
    if (avgd_dat[i, 2] == -34500) and (avgd_dat[i, 1] == 7950):
        y_avgdL += [(22.7 / 30000) * (-(avgd_dat[i,0] - T0))]
        bx_avgdL += [-avgd_dat[i,3]]
        by_avgdL += [-avgd_dat[i,7]]
        bz_avgdL += [-avgd_dat[i,5]]
        
#along B0
plt.scatter(y, bx, color="wheat")
plt.scatter(y_avgd, bx_avgd, color="cyan")
plt.scatter(y_avgdL, bx_avgdL, color="darkblue")

plt.xlabel("y (cm)")
plt.ylabel("Bx (mG)")
plt.title("Bx (along B0) vs y: off")
plt.savefig("Bx_y-avg-off.png", dpi=1000)
plt.show()

#orthogonal
#relabeled!
plt.scatter(y, by, color="wheat")
plt.scatter(y_avgd, by_avgd, color="cyan")
plt.scatter(y_avgdL, by_avgdL, color="darkblue")

plt.xlabel("y (cm)")
plt.ylabel("By (mG)")
plt.title("By vs y: off")
plt.savefig("By_y-avg-off.png", dpi=1000)
plt.show()
    
#vertical
#relaveled!
plt.scatter(y, bz, color="wheat")
plt.scatter(y_avgd, bz_avgd, color="cyan")
plt.scatter(y_avgdL, bz_avgdL, color="darkblue")

plt.xlabel("y (cm)")
plt.ylabel("Bz (mG)")
plt.title("Bz (vertical) vs y: off")
plt.savefig("Bz_y-avg-off.png", dpi=1000)
plt.show()


###############################################
# x axis data:
x = []
bx = []
by = []
bz = []
for i in range(length):
    if (dat[i, 2] == -34500) and (dat[i, 1] == 11950) :
        x += [(22.7 / 30000) * (dat[i,0] - T0)]
        bx += [dat[i,7]]
        by += [-dat[i,3]]
        bz += [-dat[i, 5]]
    if (dat[i, 2] == -34500) and (dat[i, 1] == 3950) :
        x += [(22.7 / 30000) * (-(dat[i,0] - T0))]
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
    if (avgd_dat[i, 2] == -34500) and (avgd_dat[i, 1] == 11950):
        x_avgd += [(22.7 / 30000) * (avgd_dat[i,0] - T0)]
        bx_avgd += [avgd_dat[i,7]]
        by_avgd += [-avgd_dat[i,3]]
        bz_avgd += [-avgd_dat[i,5]]
    if (avgd_dat[i, 2] == -34500) and (avgd_dat[i, 1] == 3950):
        x_avgdL += [(22.7 / 30000) * (-(avgd_dat[i,0] - T0))]
        bx_avgdL += [-avgd_dat[i,7]]
        by_avgdL += [avgd_dat[i,3]]
        bz_avgdL += [-avgd_dat[i,5]]
        
#along B0
plt.scatter(x, bx, color="wheat")
plt.scatter(x_avgd, bx_avgd, color="cyan")
plt.scatter(x_avgdL, bx_avgdL, color="darkblue")

plt.xlabel("x (cm)")
plt.ylabel("Bx (mG)")
plt.title("Bx (along B0) vs x: off")
plt.savefig("Bx_x-avg-off.png", dpi=1000)
plt.show()

#orthogonal
plt.scatter(x, by, color="wheat")
plt.scatter(x_avgd, by_avgd, color="cyan")
plt.scatter(x_avgdL, by_avgdL, color="darkblue")

plt.xlabel("x (cm)")
plt.ylabel("By (mG)")
plt.title("By vs x: off")
plt.savefig("By_x-avg-off.png", dpi=1000)
plt.show()
    
#vertical
plt.scatter(x, bz, color="wheat")
plt.scatter(x_avgd, bz_avgd, color="cyan")
plt.scatter(x_avgdL, bz_avgdL, color="darkblue")

plt.xlabel("x (cm)")
plt.ylabel("Bz (mG)")
plt.title("Bz (vertical) vs x: off")
plt.savefig("Bz_x-avg-off.png", dpi=1000)
plt.show()
