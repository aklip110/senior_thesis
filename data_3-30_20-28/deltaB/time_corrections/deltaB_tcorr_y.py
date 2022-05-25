#NOTE: z is vertical
#input files are the time-corrected and rotated (transformed) data made with three different interval sizes

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
#plt.rcParams["figure.figsize"] = 6,4

fileOff60 = "/Users/alexandraklipfel/Desktop/senior_thesis/data_3-30/off/time_corrections/averaged_off_rot_timecorr-60.txt"
datOff60 = np.loadtxt(fileOff60)

fileOn60 = "/Users/alexandraklipfel/Desktop/senior_thesis/data_3-30/on/time_corrections/averaged_on_rot_timecorr-60.txt"
datOn60 = np.loadtxt(fileOn60)

#not time corrected (for comparison)
fileOff60_noT = "/Users/alexandraklipfel/Desktop/senior_thesis/data_3-30/off/time_corrections/averaged_off_rot-60.txt"
datOff60_noT = np.loadtxt(fileOff60_noT)
fileOn60_noT = "/Users/alexandraklipfel/Desktop/senior_thesis/data_3-30/on/time_corrections/averaged_on_rot-60.txt"
datOn60_noT = np.loadtxt(fileOn60_noT)

fileOff30 = "/Users/alexandraklipfel/Desktop/senior_thesis/data_3-30/off/time_corrections/averaged_off_rot_timecorr-30.txt"
datOff30 = np.loadtxt(fileOff30)

fileOn30 = "/Users/alexandraklipfel/Desktop/senior_thesis/data_3-30/on/time_corrections/averaged_on_rot_timecorr-30.txt"
datOn30 = np.loadtxt(fileOn30)

fileOff300 = "/Users/alexandraklipfel/Desktop/senior_thesis/data_3-30/off/time_corrections/averaged_off_rot_timecorr-300.txt"
datOff300 = np.loadtxt(fileOff300)

fileOn300 = "/Users/alexandraklipfel/Desktop/senior_thesis/data_3-30/on/time_corrections/averaged_on_rot_timecorr-300.txt"
datOn300 = np.loadtxt(fileOn300)

T0 = 6400

plusX = 11950
minX = 3950
plusY = -50
minY = 7950

#print(datOn60[:, 5])
#print(datOff60[:, 5])

#subtract the off field values from the on field values
#TO DO: propagate errors

#final data:

# y axis values, 60, no time corrections
y_avgdt = []
bx_avgdt = []
by_avgdt = []
bz_avgdt = []

y_avgdLt = []
bx_avgdLt = []
by_avgdLt = []
bz_avgdLt = []

for i in range(len(datOn60_noT)):
    if (datOn60_noT[i, 2] == -34500) and (datOn60_noT[i, 1] == plusY):
        y_avgdt += [(22.7 / 30000) * (datOn60_noT[i,0] - T0)]
        bx_avgdt += [datOn60_noT[i,3] - datOff60_noT[i,3]]
        by_avgdt += [datOn60_noT[i,5] - datOff60_noT[i,5]]
        bz_avgdt += [datOn60_noT[i,7] - datOff60_noT[i,7]]

    if (datOn60_noT[i, 2] == -34500) and (datOn60_noT[i, 1] == minY):
        y_avgdLt += [(22.7 / 30000) * (-(datOn60_noT[i,0] - T0))]
        bx_avgdLt += [datOn60_noT[i,3] - datOff60_noT[i,3]]
        by_avgdLt += [datOn60_noT[i,5] - datOff60_noT[i,5]]
        bz_avgdLt += [datOn60_noT[i,7] - datOff60_noT[i,7]]
        
#mags
B_avgdt = np.sqrt(np.square(bx_avgdt) + np.square(by_avgdt) + np.square(bz_avgdt))
B_avgdLt = np.sqrt(np.square(bx_avgdLt) + np.square(by_avgdLt) + np.square(bz_avgdLt))
        
# y axis values, 60
#V=34500
y_avgdt60 = []
bx_avgdt60 = []
by_avgdt60 = []
bz_avgdt60 = []

y_avgdLt60 = []
bx_avgdLt60 = []
by_avgdLt60 = []
bz_avgdLt60 = []

for i in range(len(datOn60)):
    if (datOn60[i, 2] == -34500) and (datOn60[i, 1] == plusY):
        y_avgdt60 += [(22.7 / 30000) * (datOn60[i,0] - T0)]
        bx_avgdt60 += [datOn60[i,3] - datOff60[i,3]]
        by_avgdt60 += [datOn60[i,5] - datOff60[i,5]]
        bz_avgdt60 += [datOn60[i,7] - datOff60[i,7]]

    if (datOn60[i, 2] == -34500) and (datOn60[i, 1] == minY):
        y_avgdLt60 += [(22.7 / 30000) * (-(datOn60[i,0] - T0))]
        bx_avgdLt60 += [datOn60[i,3] - datOff60[i,3]]
        by_avgdLt60 += [datOn60[i,5] - datOff60[i,5]]
        bz_avgdLt60 += [datOn60[i,7] - datOff60[i,7]]
        
#mags
B_avgdt60 = np.sqrt(np.square(bx_avgdt60) + np.square(by_avgdt60) + np.square(bz_avgdt60))
B_avgdLt60 = np.sqrt(np.square(bx_avgdLt60) + np.square(by_avgdLt60) + np.square(bz_avgdLt60))
        
#V=-51500
y_avgdt60_515 = []
bx_avgdt60_515 = []
by_avgdt60_515 = []
bz_avgdt60_515 = []

y_avgdLt60_515 = []
bx_avgdLt60_515 = []
by_avgdLt60_515 = []
bz_avgdLt60_515 = []

for i in range(len(datOn60)):
    if (datOn60[i, 2] == -51500) and (datOn60[i, 1] == plusY):
        y_avgdt60_515 += [(22.7 / 30000) * (datOn60[i,0] - T0)]
        bx_avgdt60_515 += [datOn60[i,3] - datOff60[i,3]]
        by_avgdt60_515 += [datOn60[i,5] - datOff60[i,5]]
        bz_avgdt60_515 += [datOn60[i,7] - datOff60[i,7]]

    if (datOn60[i, 2] == -51500) and (datOn60[i, 1] == minY):
        y_avgdLt60_515 += [(22.7 / 30000) * (-(datOn60[i,0] - T0))]
        bx_avgdLt60_515 += [datOn60[i,3] - datOff60[i,3]]
        by_avgdLt60_515 += [datOn60[i,5] - datOff60[i,5]]
        bz_avgdLt60_515 += [datOn60[i,7] - datOff60[i,7]]

#mags
B_avgdt60_515 = np.sqrt(np.square(bx_avgdt60_515) + np.square(by_avgdt60_515) + np.square(bz_avgdt60_515))
B_avgdLt60_515 = np.sqrt(np.square(bx_avgdLt60_515) + np.square(by_avgdLt60_515) + np.square(bz_avgdLt60_515))
        
#V=-17500
y_avgdt60_175 = []
bx_avgdt60_175 = []
by_avgdt60_175 = []
bz_avgdt60_175 = []

y_avgdLt60_175 = []
bx_avgdLt60_175 = []
by_avgdLt60_175 = []
bz_avgdLt60_175 = []

for i in range(len(datOn60)):
    if (datOn60[i, 2] == -17500) and (datOn60[i, 1] == plusY):
        y_avgdt60_175 += [(22.7 / 30000) * (datOn60[i,0] - T0)]
        bx_avgdt60_175 += [datOn60[i,3] - datOff60[i,3]]
        by_avgdt60_175 += [datOn60[i,5] - datOff60[i,5]]
        bz_avgdt60_175 += [datOn60[i,7] - datOff60[i,7]]
        

    if (datOn60[i, 2] == -17500) and (datOn60[i, 1] == minY):
        y_avgdLt60_175 += [(22.7 / 30000) * (-(datOn60[i,0] - T0))]
        bx_avgdLt60_175 += [datOn60[i,3] - datOff60[i,3]]
        by_avgdLt60_175 += [datOn60[i,5] - datOff60[i,5]]
        bz_avgdLt60_175 += [datOn60[i,7] - datOff60[i,7]]
        
#mags
B_avgdt60_175 = np.sqrt(np.square(bx_avgdt60_175) + np.square(by_avgdt60_175) + np.square(bz_avgdt60_175))
B_avgdLt60_175 = np.sqrt(np.square(bx_avgdLt60_175) + np.square(by_avgdLt60_175) + np.square(bz_avgdLt60_175))

        
# y axis values, 30
y_avgdt30 = []
bx_avgdt30 = []

y_avgdLt30 = []
bx_avgdLt30 = []

for i in range(len(datOn30)):
    if (datOn30[i, 2] == -34500) and (datOn30[i, 1] == plusY):
        y_avgdt30 += [(22.7 / 30000) * (datOn30[i,0] - T0)]
        bx_avgdt30 += [datOn30[i,3] - datOff30[i,3]]
    if (datOn30[i, 2] == -34500) and (datOn30[i, 1] == minY):
        y_avgdLt30 += [(22.7 / 30000) * (-(datOn30[i,0] - T0))]
        bx_avgdLt30 += [datOn30[i,3] - datOff30[i,3]]
        
# y axis values, 300
y_avgdt300 = []
bx_avgdt300 = []

y_avgdLt300 = []
bx_avgdLt300 = []

for i in range(len(datOn300)):
    if (datOn300[i, 2] == -34500) and (datOn300[i, 1] == plusY):
        y_avgdt300 += [(22.7 / 30000) * (datOn300[i,0] - T0)]
        bx_avgdt300 += [datOn300[i,3] - datOff300[i,3]]
    if (datOn300[i, 2] == -34500) and (datOn300[i, 1] == minY):
        y_avgdLt300 += [(22.7 / 30000) * (-(datOn300[i,0] - T0))]
        bx_avgdLt300 += [datOn300[i,3] - datOff300[i,3]]
        
#along B0--------------------------------------------------------
plt.plot(y_avgdt30, bx_avgdt30, color="purple")
plt.plot(y_avgdLt30, bx_avgdLt30, color="purple", label="30s interval")
plt.scatter(y_avgdt30, bx_avgdt30, color="orchid")
plt.scatter(y_avgdLt30, bx_avgdLt30, color="purple")

plt.plot(y_avgdt60, bx_avgdt60, color="blue")
plt.plot(y_avgdLt60, bx_avgdLt60, color="blue", label="60s interval")
plt.scatter(y_avgdt60, bx_avgdt60, color="cyan")
plt.scatter(y_avgdLt60, bx_avgdLt60, color="blue")

plt.plot(y_avgdt300, bx_avgdt300, color="orange")
plt.plot(y_avgdLt300, bx_avgdLt300, color="orange", label="300s interval")
plt.scatter(y_avgdt300, bx_avgdt300, color="gold")
plt.scatter(y_avgdLt300, bx_avgdLt300, color="orange")

plt.xlabel("y (cm)")
plt.ylabel("Bx (mG)")
plt.title("delta Bx (along B0) vs y: time corrected, different intervals")
plt.legend(loc="upper center")
plt.savefig("Bx_y-deltaB_tcorr-ints.png", dpi=500)
plt.show()

#plot time corr (60) and no time corr-----------------------------
plt.plot(y_avgdt60, bx_avgdt60, color="blue")
plt.plot(y_avgdLt60, bx_avgdLt60, color="blue", label="time-corrected")
plt.scatter(y_avgdt60, bx_avgdt60, color="cyan")
plt.scatter(y_avgdLt60, bx_avgdLt60, color="blue")

plt.plot(y_avgdt, bx_avgdt, color="black")
plt.plot(y_avgdLt, bx_avgdLt, color="black", label="not time-corrected")
plt.scatter(y_avgdt, bx_avgdt, color="gray")
plt.scatter(y_avgdLt, bx_avgdLt, color="black")

plt.xlabel("y (cm)")
plt.ylabel("Bx (mG)")
plt.title("delta Bx (along B0) vs y: time corrected and not")
plt.legend(loc="upper center")
plt.savefig("Bx_y-deltaB_tcorr.png", dpi=500)
plt.show()

#time corrected at different heights-------------------------------
plt.plot(y_avgdt60_515, bx_avgdt60_515, color="maroon")
plt.plot(y_avgdLt60_515, bx_avgdLt60_515, color="maroon", label="V = +3.9cm")
plt.scatter(y_avgdt60_515, bx_avgdt60_515, color="red")
plt.scatter(y_avgdLt60_515, bx_avgdLt60_515, color="maroon")

plt.plot(y_avgdt60, bx_avgdt60, color="blue")
plt.plot(y_avgdLt60, bx_avgdLt60, color="blue", label="V = 0cm")
plt.scatter(y_avgdt60, bx_avgdt60, color="cyan")
plt.scatter(y_avgdLt60, bx_avgdLt60, color="blue")

plt.plot(y_avgdt60_175, bx_avgdt60_175, color="darkgreen")
plt.plot(y_avgdLt60_175, bx_avgdLt60_175, color="darkgreen",label="V = -3.9cm")
plt.scatter(y_avgdt60_175, bx_avgdt60_175, color="yellowgreen")
plt.scatter(y_avgdLt60_175, bx_avgdLt60_175, color="darkgreen")

plt.xlabel("y (cm)")
plt.ylabel("Bx (mG)")
plt.title("delta Bx (along B0) vs y: time corrected, different heights")
plt.legend(loc="upper center")
plt.savefig("Bx_y-deltaB_tcorr-heights.png", dpi=500)
plt.show()

#magnitude plots--------------------------------------------------
#1) compare time-corr with non-time-corr-----------------
plt.plot(y_avgdt60, B_avgdt60, color="blue")
plt.plot(y_avgdLt60, B_avgdLt60, color="blue", label="time-corrected")
plt.scatter(y_avgdt60, B_avgdt60, color="cyan")
plt.scatter(y_avgdLt60, B_avgdLt60, color="blue")

plt.plot(y_avgdt, B_avgdt, color="black")
plt.plot(y_avgdLt, B_avgdLt, color="black", label="not time-corrected")
plt.scatter(y_avgdt, B_avgdt, color="gray")
plt.scatter(y_avgdLt, B_avgdLt, color="black")

plt.xlabel("y (cm)")
plt.ylabel("|B| (mG)")
plt.title("|B| vs y: time corrected and not")
plt.legend(loc="upper center")
plt.savefig("B_y-deltaB_tcorr.png", dpi=500)
plt.show()

#2) different heights-----------------------------------
plt.plot(y_avgdt60_515, B_avgdt60_515, color="maroon")
plt.plot(y_avgdLt60_515, B_avgdLt60_515, color="maroon", label="V = +3.9cm")
plt.scatter(y_avgdt60_515, B_avgdt60_515, color="red")
plt.scatter(y_avgdLt60_515, B_avgdLt60_515, color="maroon")

plt.plot(y_avgdt60, B_avgdt60, color="blue")
plt.plot(y_avgdLt60, B_avgdLt60, color="blue", label="V = 0cm")
plt.scatter(y_avgdt60, B_avgdt60, color="cyan")
plt.scatter(y_avgdLt60, B_avgdLt60, color="blue")

plt.plot(y_avgdt60_175, B_avgdt60_175, color="darkgreen")
plt.plot(y_avgdLt60_175, B_avgdLt60_175, color="darkgreen", label="V = -3.9cm")
plt.scatter(y_avgdt60_175, B_avgdt60_175, color="yellowgreen")
plt.scatter(y_avgdLt60_175, B_avgdLt60_175, color="darkgreen")

plt.xlabel("y (cm)")
plt.ylabel("|B| (mG)")
plt.title("|B| vs y: time corrected, different heights")
plt.legend(loc="upper center")
plt.savefig("B_y-deltaB_tcorr_heights.png", dpi=500)
plt.show()

