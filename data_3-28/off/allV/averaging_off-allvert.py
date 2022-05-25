#NOTE: here, z is vertical axis

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

T0 = 100

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

np.savetxt("averaged_off--5000.txt", avgd_dat, delimiter=" ")

###############################################
# y axis data:
#V=-5000
y = []
bx = []
by = []
bz = []
for i in range(length):
    if (dat[i, 2] == -5000) and (dat[i, 1] == 3900) :
        y += [(22.7 / 30000) * (dat[i,0] - T0)]
        bx += [dat[i,3]]
        by += [dat[i,7]]
        bz += [-dat[i, 5]]
    if (dat[i, 2] == -5000) and (dat[i, 1] == -4300) :
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
    if (avgd_dat[i, 2] == -5000) and (avgd_dat[i, 1] == 3900):
        y_avgd += [(22.7 / 30000) * (avgd_dat[i,0] - T0)]
        bx_avgd += [avgd_dat[i,3]]
        by_avgd += [avgd_dat[i,7]]
        bz_avgd += [-avgd_dat[i,5]]
    if (avgd_dat[i, 2] == -5000) and (avgd_dat[i, 1] == -4300):
        y_avgdL += [(22.7 / 30000) * (-(avgd_dat[i,0] - T0))]
        bx_avgdL += [-avgd_dat[i,3]]
        by_avgdL += [-avgd_dat[i,7]]
        bz_avgdL += [-avgd_dat[i,5]]
        
###############################################
#V=12000
y12 = []
bx12 = []
by12 = []
bz12 = []
for i in range(length):
    if (dat[i, 2] == 12000) and (dat[i, 1] == 3900) :
        y12 += [(22.7 / 30000) * (dat[i,0] - T0)]
        bx12 += [dat[i,3]]
        by12 += [dat[i,7]]
        bz12 += [-dat[i, 5]]
    if (dat[i, 2] == 12000) and (dat[i, 1] == -4300) :
        y12 += [(22.7 / 30000) * (-(dat[i,0] - T0))]
        bx12 += [-dat[i,3]]
        by12 += [-dat[i,7]]
        bz12 += [-dat[i, 5]]
   
y_avgd12 = []
bx_avgd12 = []
by_avgd12 = []
bz_avgd12 = []

y_avgdL12 = []
bx_avgdL12 = []
by_avgdL12 = []
bz_avgdL12 = []
for i in range(len(avgd_dat)):
    if (avgd_dat[i, 2] == 12000) and (avgd_dat[i, 1] == 3900):
        y_avgd12 += [(22.7 / 30000) * (avgd_dat[i,0] - T0)]
        bx_avgd12 += [avgd_dat[i,3]]
        by_avgd12 += [avgd_dat[i,7]]
        bz_avgd12 += [-avgd_dat[i,5]]
    if (avgd_dat[i, 2] == 12000) and (avgd_dat[i, 1] == -4300):
        y_avgdL12 += [(22.7 / 30000) * (-(avgd_dat[i,0] - T0))]
        bx_avgdL12 += [-avgd_dat[i,3]]
        by_avgdL12 += [-avgd_dat[i,7]]
        bz_avgdL12 += [-avgd_dat[i,5]]
        
###############################################
#V=29000
y29 = []
bx29 = []
by29 = []
bz29 = []
for i in range(length):
    if (dat[i, 2] == 29000) and (dat[i, 1] == 3900) :
        y29 += [(22.7 / 30000) * (dat[i,0] - T0)]
        bx29 += [dat[i,3]]
        by29 += [dat[i,7]]
        bz29 += [-dat[i, 5]]
    if (dat[i, 2] == 29000) and (dat[i, 1] == -4300) :
        y29 += [(22.7 / 30000) * (-(dat[i,0] - T0))]
        bx29 += [-dat[i,3]]
        by29 += [-dat[i,7]]
        bz29 += [-dat[i, 5]]
   
y_avgd29 = []
bx_avgd29 = []
by_avgd29 = []
bz_avgd29 = []

y_avgdL29 = []
bx_avgdL29 = []
by_avgdL29 = []
bz_avgdL29 = []
for i in range(len(avgd_dat)):
    if (avgd_dat[i, 2] == 29000) and (avgd_dat[i, 1] == 3900):
        y_avgd29 += [(22.7 / 30000) * (avgd_dat[i,0] - T0)]
        bx_avgd29 += [avgd_dat[i,3]]
        by_avgd29 += [avgd_dat[i,7]]
        bz_avgd29 += [-avgd_dat[i,5]]
    if (avgd_dat[i, 2] == 29000) and (avgd_dat[i, 1] == -4300):
        y_avgdL29 += [(22.7 / 30000) * (-(avgd_dat[i,0] - T0))]
        bx_avgdL29 += [-avgd_dat[i,3]]
        by_avgdL29 += [-avgd_dat[i,7]]
        bz_avgdL29 += [-avgd_dat[i,5]]
        
#along B0
plt.scatter(y, bx, color="yellow")
plt.scatter(y12, bx12, color="wheat")
plt.scatter(y29, bx29, color="gold")

plt.scatter(y_avgd, bx_avgd, color="red")
plt.scatter(y_avgdL, bx_avgdL, color="maroon")

plt.scatter(y_avgd12, bx_avgd12, color="cyan")
plt.scatter(y_avgdL12, bx_avgdL12, color="darkblue")

plt.scatter(y_avgd29, bx_avgd29, color="yellowgreen")
plt.scatter(y_avgdL29, bx_avgdL29, color="darkgreen")

plt.xlabel("y (cm)")
plt.ylabel("Bx (mG)")
plt.title("Bx (along B0) vs y: off")
plt.savefig("Bx_y-avg-off.png", dpi=1000)
plt.show()

#orthogonal
#relabeled!
plt.scatter(y, by, color="yellow")
plt.scatter(y12, by12, color="wheat")
plt.scatter(y29, by29, color="gold")

plt.scatter(y_avgd, by_avgd, color="red")
plt.scatter(y_avgdL, by_avgdL, color="maroon")

plt.scatter(y_avgd12, by_avgd12, color="cyan")
plt.scatter(y_avgdL12, by_avgdL12, color="darkblue")

plt.scatter(y_avgd29, by_avgd29, color="yellowgreen")
plt.scatter(y_avgdL29, by_avgdL29, color="darkgreen")

plt.xlabel("y (cm)")
plt.ylabel("By (mG)")
plt.title("By vs y: off")
plt.savefig("By_y-avg-off.png", dpi=1000)
plt.show()
    
#vertical
#relaveled!
plt.scatter(y, bz, color="yellow")
plt.scatter(y12, bz12, color="wheat")
plt.scatter(y29, bz29, color="gold")

plt.scatter(y_avgd, bz_avgd, color="red")
plt.scatter(y_avgdL, bz_avgdL, color="maroon")

plt.scatter(y_avgd12, bz_avgd12, color="cyan")
plt.scatter(y_avgdL12, bz_avgdL12, color="darkblue")

plt.scatter(y_avgd29, bz_avgd29, color="yellowgreen")
plt.scatter(y_avgdL29, bz_avgdL29, color="darkgreen")

plt.xlabel("y (cm)")
plt.ylabel("Bz (mG)")
plt.title("Bz (vertical) vs y: off")
plt.savefig("Bz_y-avg-off.png", dpi=1000)
plt.show()


###############################################
# x axis data:
#V=-5000
x = []
bx = []
by = []
bz = []
for i in range(length):
    if (dat[i, 2] == -5000) and (dat[i, 1] == -200) :
        x += [(22.7 / 30000) * (dat[i,0] - T0)]
        bx += [dat[i,7]]
        by += [-dat[i,3]]
        bz += [-dat[i, 5]]
    if (dat[i, 2] == -5000) and (dat[i, 1] == 7900) :
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
    if (avgd_dat[i, 2] == -5000) and (avgd_dat[i, 1] == -200):
        x_avgd += [(22.7 / 30000) * (avgd_dat[i,0] - T0)]
        bx_avgd += [avgd_dat[i,7]]
        by_avgd += [-avgd_dat[i,3]]
        bz_avgd += [-avgd_dat[i,5]]
    if (avgd_dat[i, 2] == -5000) and (avgd_dat[i, 1] == 7900):
        x_avgdL += [(22.7 / 30000) * (-(avgd_dat[i,0] - T0))]
        bx_avgdL += [-avgd_dat[i,7]]
        by_avgdL += [avgd_dat[i,3]]
        bz_avgdL += [-avgd_dat[i,5]]

###############################################
#V=12000
x12 = []
bx12 = []
by12 = []
bz12 = []
for i in range(length):
    if (dat[i, 2] == 12000) and (dat[i, 1] == -200) :
        x12 += [(22.7 / 30000) * (dat[i,0] - T0)]
        bx12 += [dat[i,7]]
        by12 += [-dat[i,3]]
        bz12 += [-dat[i, 5]]
    if (dat[i, 2] == 12000) and (dat[i, 1] == 7900) :
        x12 += [(22.7 / 30000) * (-(dat[i,0] - T0))]
        bx12 += [-dat[i,7]]
        by12 += [dat[i,3]]
        bz12 += [-dat[i, 5]]
   
x_avgd12 = []
bx_avgd12 = []
by_avgd12 = []
bz_avgd12 = []

x_avgdL12 = []
bx_avgdL12 = []
by_avgdL12 = []
bz_avgdL12 = []
for i in range(len(avgd_dat)):
    if (avgd_dat[i, 2] == 12000) and (avgd_dat[i, 1] == -200):
        x_avgd12 += [(22.7 / 30000) * (avgd_dat[i,0] - T0)]
        bx_avgd12 += [avgd_dat[i,7]]
        by_avgd12 += [-avgd_dat[i,3]]
        bz_avgd12 += [-avgd_dat[i,5]]
    if (avgd_dat[i, 2] == 12000) and (avgd_dat[i, 1] == 7900):
        x_avgdL12 += [(22.7 / 30000) * (-(avgd_dat[i,0] - T0))]
        bx_avgdL12 += [-avgd_dat[i,7]]
        by_avgdL12 += [avgd_dat[i,3]]
        bz_avgdL12 += [-avgd_dat[i,5]]
        
###############################################
#V=29000
x29 = []
bx29 = []
by29 = []
bz29 = []
for i in range(length):
    if (dat[i, 2] == 29000) and (dat[i, 1] == -200) :
        x29 += [(22.7 / 30000) * (dat[i,0] - T0)]
        bx29 += [dat[i,7]]
        by29 += [-dat[i,3]]
        bz29 += [-dat[i, 5]]
    if (dat[i, 2] == 29000) and (dat[i, 1] == 7900) :
        x29 += [(22.7 / 30000) * (-(dat[i,0] - T0))]
        bx29 += [-dat[i,7]]
        by29 += [dat[i,3]]
        bz29 += [-dat[i, 5]]
   
x_avgd29 = []
bx_avgd29 = []
by_avgd29 = []
bz_avgd29 = []

x_avgdL29 = []
bx_avgdL29 = []
by_avgdL29 = []
bz_avgdL29 = []
for i in range(len(avgd_dat)):
    if (avgd_dat[i, 2] == 29000) and (avgd_dat[i, 1] == -200):
        x_avgd29 += [(22.7 / 30000) * (avgd_dat[i,0] - T0)]
        bx_avgd29 += [avgd_dat[i,7]]
        by_avgd29 += [-avgd_dat[i,3]]
        bz_avgd29 += [-avgd_dat[i,5]]
    if (avgd_dat[i, 2] == 29000) and (avgd_dat[i, 1] == 7900):
        x_avgdL29 += [(22.7 / 30000) * (-(avgd_dat[i,0] - T0))]
        bx_avgdL29 += [-avgd_dat[i,7]]
        by_avgdL29 += [avgd_dat[i,3]]
        bz_avgdL29 += [-avgd_dat[i,5]]
        

#along B0
plt.scatter(x, bx, color="yellow")
plt.scatter(x12, bx12, color="wheat")
plt.scatter(x29, bx29, color="gold")

plt.scatter(x_avgd, bx_avgd, color="red")
plt.scatter(x_avgdL, bx_avgdL, color="maroon")

plt.scatter(x_avgd12, bx_avgd12, color="cyan")
plt.scatter(x_avgdL12, bx_avgdL12, color="darkblue")

plt.scatter(x_avgd29, bx_avgd29, color="yellowgreen")
plt.scatter(x_avgdL29, bx_avgdL29, color="darkgreen")

plt.xlabel("x (cm)")
plt.ylabel("Bx (mG)")
plt.title("Bx (along B0) vs x: off")
plt.savefig("Bx_x-avg-off.png", dpi=1000)
plt.show()

#orthogonal
plt.scatter(x, by, color="yellow")
plt.scatter(x12, by12, color="wheat")
plt.scatter(x29, by29, color="gold")

plt.scatter(x_avgd, by_avgd, color="red")
plt.scatter(x_avgdL, by_avgdL, color="maroon")

plt.scatter(x_avgd12, by_avgd12, color="cyan")
plt.scatter(x_avgdL12, by_avgdL12, color="darkblue")

plt.scatter(x_avgd29, by_avgd29, color="yellowgreen")
plt.scatter(x_avgdL29, by_avgdL29, color="darkgreen")

plt.xlabel("x (cm)")
plt.ylabel("By (mG)")
plt.title("By vs x: off")
plt.savefig("By_x-avg-off.png", dpi=1000)
plt.show()
    
#vertical
plt.scatter(x, bz, color="yellow")
plt.scatter(x12, bz12, color="wheat")
plt.scatter(x29, bz29, color="gold")

plt.scatter(x_avgd, bz_avgd, color="red")
plt.scatter(x_avgdL, bz_avgdL, color="maroon")

plt.scatter(x_avgd12, bz_avgd12, color="cyan")
plt.scatter(x_avgdL12, bz_avgdL12, color="darkblue")

plt.scatter(x_avgd29, bz_avgd29, color="yellowgreen")
plt.scatter(x_avgdL29, bz_avgdL29, color="darkgreen")

plt.xlabel("x (cm)")
plt.ylabel("Bz (mG)")
plt.title("Bz (vertical) vs x: off")
plt.savefig("Bz_x-avg-off.png", dpi=1000)
plt.show()
