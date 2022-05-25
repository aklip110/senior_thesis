#z is vertical
#for use with the 4-10_20-25 dataset
#Goal: grab the on data. rotate each row (both the standard rotations and then the corrections (future)). save rotated data
#no averaging.
# trolley centering value not known--will have to estimate

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

print("                                               ")
print("-----------------------------------------------")
print("On Data")
#import data ------------------------------
file = "/Users/alexandraklipfel/Desktop/senior_thesis/data_4-10_20-25/data_4-10_20-25-3D_scan.txt"
Alldat = np.loadtxt(file)
print("Data Shape: ", Alldat.shape)
length = Alldat.shape[0]

#import z data
zfile = "/Users/alexandraklipfel/Desktop/senior_thesis/data_4-10_20-25/data_4-10_20-25-vertical_sweep.txt"
zdat = np.loadtxt(zfile)
Zlength = zdat.shape[0]

#grab desired on data --------------------
Offdat = np.zeros(13)
ZOffdat = np.zeros(13)

for i in range(length):
    if (Alldat[i, 9] == 1): #could add vertical filter if wanted
        Offdat = np.vstack([Offdat, Alldat[i,:]])
for i in range(Zlength):
    if (zdat[i, 9] == 1): #could add vertical filter if wanted
        ZOffdat = np.vstack([ZOffdat, zdat[i,:]])

#drop zero row
dat = Offdat[1:, :]
Zdat = ZOffdat[1:, :]
length = dat.shape[0]
Zlength = Zdat.shape[0]
print("length: ", length)
print("Z length: ", Zlength)

plusX = 4160
minX = -3840
plusY = 8160
minY = 160

V1 = -52032

T0 = -100 #need to verify

def trans_row(row, pX, mX, pY, mY):
    """
    given a row, this function will transform the Bx, By, Bz probe frame values into the lab frame UNDER THE ASSUMPTION that the given plusX==pX etc rotation values (in steps not radians) are exactly the true axes.
    row is a 1x13 dimensional vector of values.
    currently not implementing rotations to fix the misalignments.
    thus, the initial measurement errors are not transformed here.
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

#unlike the on_t_corr script from the 3-30 data, we don't need to compute and average over intervals since here we have on and on data alternating

T = dat[:, 0] #1st column
R = dat[:, 1] #2nd column
V = dat[:, 2] #3rd column

print("T values: ", np.unique(T))
print("R values: ",np.unique(R))
print("V values: ",np.unique(V))
numT = len(np.unique(T))
numR = len(np.unique(R))
numV = len(np.unique(V))

numPoints = numT * numR * numV
print("total pts in xy sweep: ", numPoints)

#data that gets time corrected
#avgd_rot_dat = np.zeros(13)
rot_dat = np.zeros(13)
#there is no "non-time-corrected" data in this case

for t in np.unique(T):
    for r in np.unique(R):
        for v in np.unique(V):
            #vals_to_avg = np.zeros(13)
            for i in range(length):
                if (dat[i, 0] == t) and (dat[i, 1] == r) and (dat[i, 2] == v):
                    #rotate/transform the row
                    newRow = trans_row(dat[i, :], plusX, minX, plusY, minY)
                    #add rotated row to vals_to_avg and rot_dat (no time-correction)
                    #vals_to_avg = np.vstack([vals_to_avg, newRow])
                    rot_dat = np.vstack([rot_dat, newRow])
            #compute average of nonzero rows
            #vals_to_avg = vals_to_avg[1:, :]
            #print(vals_to_avg_tcorr)
            #avgd_rot_dat = np.vstack([avgd_rot_dat, np.mean(vals_to_avg, axis=0)])
            
#avgd_rot_dat = avgd_rot_dat[1:, :]
rot_dat = rot_dat[1:, :]
#print(avgd_rot_tcorr_dat)
print("length of rotdat: ", rot_dat.shape[0])
#np.savetxt("averaged_on_rot.txt", avgd_rot_dat, delimiter=" ")
np.savetxt("on_rot.txt", rot_dat, delimiter=" ")

#average Z data-------------------------------------------------
Z = Zdat[:, 2] #3rd column
numZ = np.unique(Z)
print("Z values: ", numZ)

#avgd_rot_Zdat = np.zeros(13)
rot_Zdat = np.zeros(13)

for zval in numZ:
    #print("zval: ", zval)
    #zvals_to_avg = np.zeros(13)
    for i in range(Zlength):
        #print("dataval: ", Zdat[i, 2])
        if Zdat[i, 2] == zval:
            #print("YES")
            #rotate/transform the row
            newZRow = trans_row(Zdat[i, :], plusX, minX, plusY, minY)
            #zvals_to_avg = np.vstack([zvals_to_avg, newZRow])
            rot_Zdat = np.vstack([rot_Zdat, newZRow])
    #zvals_to_avg = zvals_to_avg[1:, :]
    #avgd_rot_Zdat = np.vstack([avgd_rot_Zdat, np.mean(zvals_to_avg, axis=0)])
        
#avgd_rot_Zdat = avgd_rot_Zdat[1:, :]
rot_Zdat = rot_Zdat[1:, :]
#print("length of avg_zdat: ", avgd_rot_Zdat.shape[0])
print("length of zdat: ", rot_Zdat.shape[0])

#np.savetxt("averaged_on_rot-Z.txt", avgd_rot_Zdat, delimiter=" ")
np.savetxt("on_rot-Z.txt", rot_Zdat, delimiter=" ")

print("-----------------------------------------------")
print("                                               ")

#--------------------------------------------------------------
