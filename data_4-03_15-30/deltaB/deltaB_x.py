#we compute deltaB and make plots of Bx, By, Bx, |B| vs x
#for the 4/03 data

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

#import off data
fileOff = "/Users/alexandraklipfel/Desktop/senior_thesis/data_4-03/off/averaged_off_rot.txt"
datOff = np.loadtxt(fileOff)

#import on data
fileOn = "/Users/alexandraklipfel/Desktop/senior_thesis/data_4-03/on/averaged_on_rot.txt"
datOn = np.loadtxt(fileOn)

#Again, there are no intervals, so everything is much simpler

plusX = 4150
minX = -3850
plusY = 8150
minY = 150

V1 = -72032
V2 = -52032
V3 = -32032

T0 = 4525 #need to verify

#plots vs. x
#V= -72032
x_1 = []
bx_1 = []
by_1 = []
bz_1 = []

x_1L = []
bx_1L = []
by_1L = []
bz_1L = []

for i in range(len(datOn)):
    if (datOn[i, 2] == V1) and (datOn[i, 1] == plusX):
        x_1 += [(22.7 / 30000) * (datOn[i,0] - T0)]
        bx_1 += [datOn[i,3] - datOff[i,3]]
        by_1 += [datOn[i,5] - datOff[i,5]]
        bz_1 += [datOn[i,7] - datOff[i,7]]

    if (datOn[i, 2] == V1) and (datOn[i, 1] == minX):
        x_1L += [(22.7 / 30000) * (-(datOn[i,0] - T0))]
        bx_1L += [datOn[i,3] - datOff[i,3]]
        by_1L += [datOn[i,5] - datOff[i,5]]
        bz_1L += [datOn[i,7] - datOff[i,7]]
        
#mags
B_1 = np.sqrt(np.square(bx_1) + np.square(by_1) + np.square(bz_1))
B_1L = np.sqrt(np.square(bx_1L) + np.square(by_1L) + np.square(bz_1L))

#V= -52032
x_2 = []
bx_2 = []
by_2 = []
bz_2 = []

x_2L = []
bx_2L = []
by_2L = []
bz_2L = []

for i in range(len(datOn)):
    if (datOn[i, 2] == V2) and (datOn[i, 1] == plusX):
        x_2 += [(22.7 / 30000) * (datOn[i,0] - T0)]
        bx_2 += [datOn[i,3] - datOff[i,3]]
        by_2 += [datOn[i,5] - datOff[i,5]]
        bz_2 += [datOn[i,7] - datOff[i,7]]

    if (datOn[i, 2] == V2) and (datOn[i, 1] == minX):
        x_2L += [(22.7 / 30000) * (-(datOn[i,0] - T0))]
        bx_2L += [datOn[i,3] - datOff[i,3]]
        by_2L += [datOn[i,5] - datOff[i,5]]
        bz_2L += [datOn[i,7] - datOff[i,7]]
        
#mags
B_2 = np.sqrt(np.square(bx_2) + np.square(by_2) + np.square(bz_2))
B_2L = np.sqrt(np.square(bx_2L) + np.square(by_2L) + np.square(bz_2L))

#V= -32032
x_3 = []
bx_3 = []
by_3 = []
bz_3 = []

x_3L = []
bx_3L = []
by_3L = []
bz_3L = []

for i in range(len(datOn)):
    if (datOn[i, 2] == V3) and (datOn[i, 1] == plusX):
        x_3 += [(22.7 / 30000) * (datOn[i,0] - T0)]
        bx_3 += [datOn[i,3] - datOff[i,3]]
        by_3 += [datOn[i,5] - datOff[i,5]]
        bz_3 += [datOn[i,7] - datOff[i,7]]

    if (datOn[i, 2] == V3) and (datOn[i, 1] == minX):
        x_3L += [(22.7 / 30000) * (-(datOn[i,0] - T0))]
        bx_3L += [datOn[i,3] - datOff[i,3]]
        by_3L += [datOn[i,5] - datOff[i,5]]
        bz_3L += [datOn[i,7] - datOff[i,7]]
        
#mags
B_3 = np.sqrt(np.square(bx_3) + np.square(by_3) + np.square(bz_3))
B_3L = np.sqrt(np.square(bx_3L) + np.square(by_3L) + np.square(bz_3L))

#time corrected at different heights-------------------------------
#Bx
plt.plot(x_1, bx_1, color="maroon")
plt.plot(x_1L, bx_1L, color="maroon", label="V = "+ str(V1))
plt.scatter(x_1, bx_1, color="red")
plt.scatter(x_1L, bx_1L, color="maroon")

plt.plot(x_2, bx_2, color="blue")
plt.plot(x_2L, bx_2L, color="blue", label="V = "+ str(V2))
plt.scatter(x_2, bx_2, color="cyan")
plt.scatter(x_2L, bx_2L, color="blue")

plt.plot(x_3, bx_3, color="darkgreen")
plt.plot(x_3L, bx_3L, color="darkgreen",label="V = "+ str(V3))
plt.scatter(x_3, bx_3, color="yellowgreen")
plt.scatter(x_3L, bx_3L, color="darkgreen")

plt.xlabel("x (cm)")
plt.ylabel("Bx (mG)")
plt.title("delta Bx (along B0) vs x: time corrected, different heights")
plt.legend(loc="upper center")
plt.savefig("Bx_x-deltaB_tcorr-heights.png", dpi=500)
plt.show()

#By---------------------------------------------------------
plt.plot(x_1, by_1, color="maroon")
plt.plot(x_1L, by_1L, color="maroon", label="V = "+ str(V1))
plt.scatter(x_1, by_1, color="red")
plt.scatter(x_1L, by_1L, color="maroon")

plt.plot(x_2, by_2, color="blue")
plt.plot(x_2L, by_2L, color="blue", label="V = "+ str(V2))
plt.scatter(x_2, by_2, color="cyan")
plt.scatter(x_2L, by_2L, color="blue")

plt.plot(x_3, by_3, color="darkgreen")
plt.plot(x_3L, by_3L, color="darkgreen",label="V = "+ str(V3))
plt.scatter(x_3, by_3, color="yellowgreen")
plt.scatter(x_3L, by_3L, color="darkgreen")

plt.xlabel("x (cm)")
plt.ylabel("By (mG)")
plt.title("delta By vs x: time corrected, different heights")
plt.legend(loc="upper left")
plt.savefig("By_x-deltaB_tcorr-heights.png", dpi=500)
plt.show()

#Bz---------------------------------------------------------
plt.plot(x_1, bz_1, color="maroon")
plt.plot(x_1L, bz_1L, color="maroon", label="V = "+ str(V1))
plt.scatter(x_1, bz_1, color="red")
plt.scatter(x_1L, bz_1L, color="maroon")

plt.plot(x_2, bz_2, color="blue")
plt.plot(x_2L, bz_2L, color="blue", label="V = "+ str(V2))
plt.scatter(x_2, bz_2, color="cyan")
plt.scatter(x_2L, bz_2L, color="blue")

plt.plot(x_3, bz_3, color="darkgreen")
plt.plot(x_3L, bz_3L, color="darkgreen",label="V = "+ str(V3))
plt.scatter(x_3, bz_3, color="yellowgreen")
plt.scatter(x_3L, bz_3L, color="darkgreen")

plt.xlabel("x (cm)")
plt.ylabel("Bz (mG)")
plt.title("delta Bz (vertical) vs x: time corrected, different heights")
plt.legend(loc="upper left")
plt.savefig("Bz_x-deltaB_tcorr-heights.png", dpi=500)
plt.show()

#|B|----------------------------------------------
plt.plot(x_1, B_1, color="maroon")
plt.plot(x_1L, B_1L, color="maroon", label="V = "+ str(V1))
plt.scatter(x_1, B_1, color="red")
plt.scatter(x_1L, B_1L, color="maroon")

plt.plot(x_2, B_2, color="blue")
plt.plot(x_2L, B_2L, color="blue", label="V = "+ str(V2))
plt.scatter(x_2, B_2, color="cyan")
plt.scatter(x_2L, B_2L, color="blue")

plt.plot(x_3, B_3, color="darkgreen")
plt.plot(x_3L, B_3L, color="darkgreen",label="V = "+ str(V3))
plt.scatter(x_3, B_3, color="yellowgreen")
plt.scatter(x_3L, B_3L, color="darkgreen")

plt.xlabel("x (cm)")
plt.ylabel("|B| (mG)")
plt.title("delta |B| (vertical) vs x: time corrected, different heights")
plt.legend(loc="upper left")
plt.savefig("B_x-deltaB_tcorr-heights.png", dpi=500)
plt.show()
