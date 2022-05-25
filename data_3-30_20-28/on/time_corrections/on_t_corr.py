#time corrections to on data
#NOTE: here, z is vertical axis
#V=-34500 plots
#for use of the 3/30 dataset

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

#import data ------------------------------
file = "/Users/alexandraklipfel/Desktop/senior_thesis/data_3-30/data_3-30-noedits.txt"
Alldat = np.loadtxt(file)
print("Data Shape: ", Alldat.shape)
length = Alldat.shape[0]

#grab desired on data --------------------
Offdat = np.zeros(13)

for i in range(length):
    if (Alldat[i, 9] == 1): #could add vertical filter if wanted
        Offdat = np.vstack([Offdat, Alldat[i,:]])

#drop zero row
dat = Offdat[1:, :]
length = dat.shape[0]
print("length: ", length)

plusX = 11950
minX = 3950
plusY = -50
minY = 7950

def trans_row(row, pX, mX, pY, mY):
    """
    given a row, this function will transform the Bx, By, Bz probe frame values into the lab frame UNDER THE ASSUMPTION that the given pX etc rotation values (in steps not radians) are exactly the true axes.
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
    

#perfom averaging over interval -----------
#set interval length for averaging
interval = 300 #seconds
totTime = dat[-1, 12] - dat[0, 12] #total time in seconds
print("Total time elapsed: ", totTime)
ppInterval = int(np.floor(length / (totTime / interval))) #points per interval
print("points per interval: ", ppInterval)
if length % ppInterval == 0:
    numIntervals = int(length / ppInterval)
else:
    numIntervals = int(np.floor(length / ppInterval) + 1)
print("number of intervals: ", numIntervals)
intervalVals = np.zeros(numIntervals)
timeVals = np.zeros(numIntervals)
#set values
for i in range(numIntervals - 1):
    intervalVals[i] = np.mean(dat[(i * ppInterval):((i+1) * ppInterval), 10])
    timeVals[i] = ((dat[(i+1) * ppInterval - 1, 12] - dat[i * ppInterval, 12]) / 2) + dat[i * ppInterval, 12]
#set last value
intervalVals[-1] = np.mean(dat[((numIntervals - 1) * ppInterval):, 10])
timeVals[-1] = ((dat[-1, 12] - dat[(numIntervals - 1) * ppInterval, 12]) / 2) + dat[(numIntervals - 1) * ppInterval, 12]
#create "deviation" array
deviation = intervalVals - intervalVals[0]
#print(intervalVals)
print(timeVals)

#plot all data and overlay the averaged data
plt.scatter(dat[:, 12] / 60, dat[:, 10])
plt.scatter(timeVals / 60, intervalVals, color="red")
plt.plot(timeVals / 60, intervalVals, color="red")
plt.scatter
plt.xlabel("time (minutes)")
plt.ylabel("background (mG)")
plt.title("Change in single-axis probe reading over time: on")
plt.tight_layout()
plt.savefig("SA_on_avg--" + str(interval) + ".png", dpi=1000)
plt.show()
#plot deviation vs. timevals
plt.scatter(timeVals / 60, deviation, color="red")
plt.plot(timeVals / 60, deviation, color="red")
plt.scatter
plt.xlabel("time (minutes)")
plt.ylabel("background deviation (mG)")
plt.title("Change in single-axis probe deviation over time: on")
plt.tight_layout()
plt.savefig("SA_on_deviation--" + str(interval) + ".png", dpi=1000)
plt.show()

#now take code from the averaging_on-34500.py script
#have to add a section to the part which actually does the averaging. we want to add/subtract the time-dep background from each point BEFORE the averaging
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

#data that gets time corrected
avgd_rot_tcorr_dat = np.zeros(13)
rot_tcorr_dat = np.zeros(13)
#data that does not get time corrected--for comparison
avgd_rot_dat = np.zeros(13)
rot_dat = np.zeros(13)

for t in np.unique(T):
    for r in np.unique(R):
        for v in np.unique(V):
            vals_to_avg_tcorr = np.zeros(13)
            vals_to_avg = np.zeros(13)
            for i in range(length):
                if (dat[i, 0] == t) and (dat[i, 1] == r) and (dat[i, 2] == v):
                    #check time
                    time = dat[i, 12] - dat[0, 12] #time since start, not absolute time
                    #determine with interval it corresponds to
                    intNumber = int(np.floor(time / interval))
                    #rotate/transform the row
                    newRow = trans_row(dat[i, :], plusX, minX, plusY, minY)
                    #add rotated row to vals_to_avg and rot_dat (no time-correction)
                    vals_to_avg = np.vstack([vals_to_avg, newRow])
                    rot_dat = np.vstack([rot_dat, newRow])
                    #subtract/add the value of that interval from the deviation array
                    newRow[3] += deviation[intNumber]
                    #THEN add this row to the "to be averaged" array
                    vals_to_avg_tcorr = np.vstack([vals_to_avg_tcorr, newRow])
                    #ALSO, add this row to new array of rotated, time-corrected data (note that its order will be different from the original dat array)
                    rot_tcorr_dat = np.vstack([rot_tcorr_dat, newRow])
            #compute average of nonzero rows
            vals_to_avg_tcorr = vals_to_avg_tcorr[1:, :]
            vals_to_avg = vals_to_avg[1:, :]
            #print(vals_to_avg_tcorr)
            avgd_rot_tcorr_dat = np.vstack([avgd_rot_tcorr_dat, np.mean(vals_to_avg_tcorr, axis=0)])
            avgd_rot_dat = np.vstack([avgd_rot_dat, np.mean(vals_to_avg, axis=0)])
            
avgd_rot_tcorr_dat = avgd_rot_tcorr_dat[1:, :]
rot_tcorr_dat = rot_tcorr_dat[1:, :]
avgd_rot_dat = avgd_rot_dat[1:, :]
rot_dat = rot_dat[1:, :]
#print(avgd_rot_tcorr_dat)

np.savetxt("averaged_on_rot_timecorr-"+ str(interval)+ ".txt", avgd_rot_tcorr_dat, delimiter=" ")
np.savetxt("averaged_on_rot-"+ str(interval)+ ".txt", avgd_rot_dat, delimiter=" ")

#now make plots
#first plot the un-time corrected data to compare to my existing plots
# x axis data:

#might finish this later...
#def plot_V(regular, averaged, V, axis, plus, minus):
    #"""
    #this function makes Bx, By, and Bz along "axis"=='x' or 'y'.
    #regular is the un-averaged data.
    #averaged is the averaged data.
    #V is the vertical value in
    #"""

x = []
bx = []
by = []
bz = []
for i in range(length):
    if (rot_dat[i, 2] == -34500) and (rot_dat[i, 1] == plusX) :
        x += [(22.7 / 30000) * (rot_dat[i,0] - T0)]
        bx += [rot_dat[i, 3]]
        by += [rot_dat[i, 5]]
        bz += [rot_dat[i, 7]]
    if (rot_dat[i, 2] == -34500) and (rot_dat[i, 1] == minX) :
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
    if (avgd_rot_dat[i, 2] == -34500) and (avgd_rot_dat[i, 1] == plusX):
        x_avgd += [(22.7 / 30000) * (avgd_rot_dat[i,0] - T0)]
        bx_avgd += [avgd_rot_dat[i,3]]
        by_avgd += [avgd_rot_dat[i,5]]
        bz_avgd += [avgd_rot_dat[i,7]]
    if (avgd_rot_dat[i, 2] == -34500) and (avgd_rot_dat[i, 1] == minX):
        x_avgdL += [(22.7 / 30000) * (-(avgd_rot_dat[i,0] - T0))]
        bx_avgdL += [avgd_rot_dat[i,3]]
        by_avgdL += [avgd_rot_dat[i,5]]
        bz_avgdL += [avgd_rot_dat[i,7]]
        
#time corrected data
xt = []
bxt = []
byt = []
bzt = []
for i in range(length):
    if (rot_tcorr_dat[i, 2] == -34500) and (rot_tcorr_dat[i, 1] == plusX) :
        xt += [(22.7 / 30000) * (rot_tcorr_dat[i,0] - T0)]
        bxt += [rot_tcorr_dat[i, 3]]
        byt += [rot_tcorr_dat[i, 5]]
        bzt += [rot_tcorr_dat[i, 7]]
    if (rot_tcorr_dat[i, 2] == -34500) and (rot_tcorr_dat[i, 1] == minX) :
        xt += [(22.7 / 30000) * (-(rot_tcorr_dat[i,0] - T0))]
        bxt += [rot_tcorr_dat[i, 3]]
        byt += [rot_tcorr_dat[i, 5]]
        bzt += [rot_tcorr_dat[i, 7]]

x_avgdt = []
bx_avgdt = []
by_avgdt = []
bz_avgdt = []

x_avgdLt = []
bx_avgdLt = []
by_avgdLt = []
bz_avgdLt = []
for i in range(len(avgd_rot_tcorr_dat)):
    if (avgd_rot_tcorr_dat[i, 2] == -34500) and (avgd_rot_tcorr_dat[i, 1] == plusX):
        x_avgdt += [(22.7 / 30000) * (avgd_rot_tcorr_dat[i,0] - T0)]
        bx_avgdt += [avgd_rot_tcorr_dat[i,3]]
        by_avgdt += [avgd_rot_tcorr_dat[i,5]]
        bz_avgdt += [avgd_rot_tcorr_dat[i,7]]
    if (avgd_rot_tcorr_dat[i, 2] == -34500) and (avgd_rot_tcorr_dat[i, 1] == minX):
        x_avgdLt += [(22.7 / 30000) * (-(avgd_rot_tcorr_dat[i,0] - T0))]
        bx_avgdLt += [avgd_rot_tcorr_dat[i,3]]
        by_avgdLt += [avgd_rot_tcorr_dat[i,5]]
        bz_avgdLt += [avgd_rot_tcorr_dat[i,7]]

#along B0
plt.scatter(x, bx, color="yellow")
plt.scatter(xt, bxt, color="wheat")

plt.scatter(x_avgdt, bx_avgdt, color="cyan")
plt.scatter(x_avgdLt, bx_avgdLt, color="darkblue")

plt.scatter(x_avgd, bx_avgd, color="gray")
plt.scatter(x_avgdL, bx_avgdL, color="black")

plt.xlabel("x (cm)")
plt.ylabel("Bx (mG)")
plt.title("Bx (along B0) vs x: on, time corrected")
plt.savefig("Bx_x-avg-on-tcorr-compare-"+ str(interval)+ ".png", dpi=1000)
plt.show()

#########
#y data
y = []
bx = []
by = []
bz = []
for i in range(length):
    if (rot_dat[i, 2] == -34500) and (rot_dat[i, 1] == plusY) :
        y += [(22.7 / 30000) * (rot_dat[i,0] - T0)]
        bx += [rot_dat[i, 3]]
        by += [rot_dat[i, 5]]
        bz += [rot_dat[i, 7]]
    if (rot_dat[i, 2] == -34500) and (rot_dat[i, 1] == minY) :
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
    if (avgd_rot_dat[i, 2] == -34500) and (avgd_rot_dat[i, 1] == plusY):
        y_avgd += [(22.7 / 30000) * (avgd_rot_dat[i,0] - T0)]
        bx_avgd += [avgd_rot_dat[i,3]]
        by_avgd += [avgd_rot_dat[i,5]]
        bz_avgd += [avgd_rot_dat[i,7]]
    if (avgd_rot_dat[i, 2] == -34500) and (avgd_rot_dat[i, 1] == minY):
        y_avgdL += [(22.7 / 30000) * (-(avgd_rot_dat[i,0] - T0))]
        bx_avgdL += [avgd_rot_dat[i,3]]
        by_avgdL += [avgd_rot_dat[i,5]]
        bz_avgdL += [avgd_rot_dat[i,7]]

#time corrected data
yt = []
bxt = []
byt = []
bzt = []
for i in range(length):
    if (rot_tcorr_dat[i, 2] == -34500) and (rot_tcorr_dat[i, 1] == plusY) :
        yt += [(22.7 / 30000) * (rot_tcorr_dat[i,0] - T0)]
        bxt += [rot_tcorr_dat[i, 3]]
        byt += [rot_tcorr_dat[i, 5]]
        bzt += [rot_tcorr_dat[i, 7]]
    if (rot_tcorr_dat[i, 2] == -34500) and (rot_tcorr_dat[i, 1] == minY) :
        yt += [(22.7 / 30000) * (-(rot_tcorr_dat[i,0] - T0))]
        bxt += [rot_tcorr_dat[i, 3]]
        byt += [rot_tcorr_dat[i, 5]]
        bzt += [rot_tcorr_dat[i, 7]]

y_avgdt = []
bx_avgdt = []
by_avgdt = []
bz_avgdt = []

y_avgdLt = []
bx_avgdLt = []
by_avgdLt = []
bz_avgdLt = []
for i in range(len(avgd_rot_tcorr_dat)):
    if (avgd_rot_tcorr_dat[i, 2] == -34500) and (avgd_rot_tcorr_dat[i, 1] == plusY):
        y_avgdt += [(22.7 / 30000) * (avgd_rot_tcorr_dat[i,0] - T0)]
        bx_avgdt += [avgd_rot_tcorr_dat[i,3]]
        by_avgdt += [avgd_rot_tcorr_dat[i,5]]
        bz_avgdt += [avgd_rot_tcorr_dat[i,7]]
    if (avgd_rot_tcorr_dat[i, 2] == -34500) and (avgd_rot_tcorr_dat[i, 1] == minY):
        y_avgdLt += [(22.7 / 30000) * (-(avgd_rot_tcorr_dat[i,0] - T0))]
        bx_avgdLt += [avgd_rot_tcorr_dat[i,3]]
        by_avgdLt += [avgd_rot_tcorr_dat[i,5]]
        bz_avgdLt += [avgd_rot_tcorr_dat[i,7]]

#along y axis
plt.scatter(y, bx, color="yellow")
plt.scatter(yt, bxt, color="wheat")

plt.scatter(y_avgdt, bx_avgdt, color="cyan")
plt.scatter(y_avgdLt, bx_avgdLt, color="darkblue")

plt.scatter(y_avgd, bx_avgd, color="gray")
plt.scatter(y_avgdL, bx_avgdL, color="black")

plt.xlabel("y (cm)")
plt.ylabel("Bx (mG)")
plt.title("Bx (along B0) vs y: on, time corrected")
plt.savefig("Bx_y-avg-on-tcorr-compare-"+ str(interval)+ ".png", dpi=1000)
plt.show()
