#z is vertical
#for use with the 4-03 dataset
#this just averages the off data

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

#import data ------------------------------
file = "/Users/alexandraklipfel/Desktop/senior_thesis/data_4-03/data_4-03-3D_scan.txt"
Alldat = np.loadtxt(file)
print("Data Shape: ", Alldat.shape)
length = Alldat.shape[0]

#import z data
zfile = "/Users/alexandraklipfel/Desktop/senior_thesis/data_4-03/data_4-03-vertical_sweep.txt"
zdat = np.loadtxt(zfile)
Zlength = zdat.shape[0]

#grab desired off data --------------------
Offdat = np.zeros(13)
ZOffdat = np.zeros(13)

for i in range(length):
    if (Alldat[i, 9] == 0): #could add vertical filter if wanted
        Offdat = np.vstack([Offdat, Alldat[i,:]])
for i in range(Zlength):
    if (zdat[i, 9] == 0): #could add vertical filter if wanted
        ZOffdat = np.vstack([ZOffdat, zdat[i,:]])

#drop zero row
dat = Offdat[1:, :]
Zdat = ZOffdat[1:, :]
length = dat.shape[0]
print("length: ", length)
Zlength = Zdat.shape[0]
print("Z length: ", Zlength)

plusX = 4150
minX = -3850
plusY = 8150
minY = 150

V1 = -72032
V2 = -52032
V3 = -32032

T0 = 4525 #need to verify

def trans_row(row, pX, mX, pY, mY):
    """
    given a row, this function will transform the Bx, By, Bz probe frame values into the lab frame UNDER THE ASSUMPTION that the given plusX==pX etc rotation values (in steps not radians) are exactly the true axes.
    row is a 1x13 dimensional vector of values.
    """
    newRow = np.copy(row)
    R = row[1]
    if R == pX:
        newRow[3] = row[7]
        newRow[5] = -row[3]
    if R == mX:
        newRow[3] = -row[7]
        newRow[5] = row[3]
    if R == pY:
        newRow[3] = row[3]
        newRow[5] = row[7]
    if R == mY:
        newRow[3] = -row[3]
        newRow[5] = -row[7]
    newRow[7] = -row[5]
    return newRow

#unlike the off_t_corr script from the 3-30 data, we don't need to compute and average over intervals since here we have off and on data alternating

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

#data that gets time corrected
avgd_rot_dat = np.zeros(13)
rot_dat = np.zeros(13)
#there is no "non-time-corrected" data in this case

for t in np.unique(T):
    for r in np.unique(R):
        for v in np.unique(V):
            vals_to_avg = np.zeros(13)
            for i in range(length):
                if (dat[i, 0] == t) and (dat[i, 1] == r) and (dat[i, 2] == v):
                    #rotate/transform the row
                    newRow = trans_row(dat[i, :], plusX, minX, plusY, minY)
                    #add rotated row to vals_to_avg and rot_dat (no time-correction)
                    vals_to_avg = np.vstack([vals_to_avg, newRow])
                    rot_dat = np.vstack([rot_dat, newRow])
            #compute average of nonzero rows
            vals_to_avg = vals_to_avg[1:, :]
            #print(vals_to_avg_tcorr)
            avgd_rot_dat = np.vstack([avgd_rot_dat, np.mean(vals_to_avg, axis=0)])
            
avgd_rot_dat = avgd_rot_dat[1:, :]
rot_dat = rot_dat[1:, :]
#print(avgd_rot_tcorr_dat)

np.savetxt("averaged_off_rot.txt", avgd_rot_dat, delimiter=" ")
np.savetxt("off_rot.txt", rot_dat, delimiter=" ")

#average Z data
Z = Zdat[:, 2] #3rd column
numZ = np.unique(Z)
print("numZ: ", numZ)

avgd_rot_Zdat = np.zeros(13)
rot_Zdat = np.zeros(13)

for zval in numZ:
    #print("zval: ", zval)
    zvals_to_avg = np.zeros(13)
    for i in range(Zlength):
        #print("dataval: ", Zdat[i, 2])
        if Zdat[i, 2] == zval:
            print("YES")
            #rotate/transform the row
            newZRow = trans_row(Zdat[i, :], plusX, minX, plusY, minY)
            zvals_to_avg = np.vstack([zvals_to_avg, newZRow])
            rot_Zdat = np.vstack([rot_Zdat, newZRow])
    zvals_to_avg = zvals_to_avg[1:, :]
    avgd_rot_Zdat = np.vstack([avgd_rot_Zdat, np.mean(zvals_to_avg, axis=0)])
        
avgd_rot_Zdat = avgd_rot_Zdat[1:, :]
rot_Zdat = rot_Zdat[1:, :]
print("length of avg_zdat: ", avgd_rot_Zdat.shape[0])
print("length of zdat: ", rot_Zdat.shape[0])

np.savetxt("averaged_off_rot-Z.txt", avgd_rot_Zdat, delimiter=" ")
np.savetxt("off_rot-Z.txt", rot_Zdat, delimiter=" ")
      
#--------------------------------------------------------------
#z plots
z = (15.2 / 50000) * (rot_Zdat[:, 2] - V2)
print(z)
zavgd = (15.2 / 50000) * (avgd_rot_Zdat[:, 2] - V2)

#Bx vs z
plt.scatter(z, rot_Zdat[:, 3], color="yellow")
plt.scatter(zavgd, avgd_rot_Zdat[:, 3], color="maroon")
plt.plot(zavgd, avgd_rot_Zdat[:, 3], color="maroon")
      
plt.xlabel("z (cm)")
plt.ylabel("Bx (mG)")
plt.title("Bx (along B0) vs z: off")
plt.savefig("Bx_z-off_V1.png", dpi=500)
plt.show()
      
#-------------------------------------------------------------

x = []
bx = []
by = []
bz = []
for i in range(length):
    if (rot_dat[i, 2] == V1) and (rot_dat[i, 1] == plusX) :
        x += [(22.7 / 30000) * (rot_dat[i,0] - T0)]
        bx += [rot_dat[i, 3]]
        by += [rot_dat[i, 5]]
        bz += [rot_dat[i, 7]]
    if (rot_dat[i, 2] == V1) and (rot_dat[i, 1] == minX) :
        x += [(22.7 / 30000) * (-(rot_dat[i,0] - T0))]
        bx += [rot_dat[i, 3]]
        by += [rot_dat[i, 5]]
        bz += [rot_dat[i, 7]]
        
x_avgd = []
bx_avgd = []
by_avgd = []
bz_avgd = []

x_avgdL = []
bx_avgdL = []
by_avgdL = []
bz_avgdL = []
for i in range(len(avgd_rot_dat)):
    if (avgd_rot_dat[i, 2] == V1) and (avgd_rot_dat[i, 1] == plusX):
        x_avgd += [(22.7 / 30000) * (avgd_rot_dat[i,0] - T0)]
        bx_avgd += [avgd_rot_dat[i,3]]
        by_avgd += [avgd_rot_dat[i,5]]
        bz_avgd += [avgd_rot_dat[i,7]]
    if (avgd_rot_dat[i, 2] == V1) and (avgd_rot_dat[i, 1] == minX):
        x_avgdL += [(22.7 / 30000) * (-(avgd_rot_dat[i,0] - T0))]
        bx_avgdL += [avgd_rot_dat[i,3]]
        by_avgdL += [avgd_rot_dat[i,5]]
        bz_avgdL += [avgd_rot_dat[i,7]]
        
#x plots
plt.scatter(x, bx, color="yellow")
plt.scatter(x_avgd, bx_avgd, color="gray")
plt.scatter(x_avgdL, bx_avgdL, color="black")
plt.plot(x_avgd, bx_avgd, color="black")
plt.plot(x_avgdL, bx_avgdL, color="black")

plt.xlabel("x (cm)")
plt.ylabel("Bx (mG)")
plt.title("Bx (along B0) vs x: off")
plt.savefig("Bx_x-off_V1.png", dpi=1000)
plt.show()

#----------------------------------------------------
y = []
bx = []
by = []
bz = []
for i in range(length):
    if (rot_dat[i, 2] == V1) and (rot_dat[i, 1] == plusY) :
        y += [(22.7 / 30000) * (rot_dat[i,0] - T0)]
        bx += [rot_dat[i, 3]]
        by += [rot_dat[i, 5]]
        bz += [rot_dat[i, 7]]
    if (rot_dat[i, 2] == V1) and (rot_dat[i, 1] == minY) :
        y += [(22.7 / 30000) * (-(rot_dat[i,0] - T0))]
        bx += [rot_dat[i, 3]]
        by += [rot_dat[i, 5]]
        bz += [rot_dat[i, 7]]
        
y_avgd = []
bx_avgd = []
by_avgd = []
bz_avgd = []

y_avgdL = []
bx_avgdL = []
by_avgdL = []
bz_avgdL = []
for i in range(len(avgd_rot_dat)):
    if (avgd_rot_dat[i, 2] == V1) and (avgd_rot_dat[i, 1] == plusY):
        y_avgd += [(22.7 / 30000) * (avgd_rot_dat[i,0] - T0)]
        bx_avgd += [avgd_rot_dat[i,3]]
        by_avgd += [avgd_rot_dat[i,5]]
        bz_avgd += [avgd_rot_dat[i,7]]
    if (avgd_rot_dat[i, 2] == V1) and (avgd_rot_dat[i, 1] == minY):
        y_avgdL += [(22.7 / 30000) * (-(avgd_rot_dat[i,0] - T0))]
        bx_avgdL += [avgd_rot_dat[i,3]]
        by_avgdL += [avgd_rot_dat[i,5]]
        bz_avgdL += [avgd_rot_dat[i,7]]

#y plots
plt.scatter(y, bx, color="yellow")
plt.scatter(y_avgd, bx_avgd, color="gray")
plt.scatter(y_avgdL, bx_avgdL, color="black")
plt.plot(y_avgd, bx_avgd, color="black")
plt.plot(y_avgdL, bx_avgdL, color="black")

plt.xlabel("y (cm)")
plt.ylabel("Bx (mG)")
plt.title("Bx (along B0) vs y: off")
plt.savefig("Bx_y-off_V1.png", dpi=1000)
plt.show()
