#rescale comsol plots and overlay 3axis probe data

import numpy as np
import matplotlib.pyplot as plt

#import comsol data
BxX_file = "/Users/alexandraklipfel/Desktop/senior_thesis/comsol_dat/B017fieldprofile_Metglas_noPB_Bx(x).txt"
BxX_CS = np.loadtxt(BxX_file)

ByX_file = "/Users/alexandraklipfel/Desktop/senior_thesis/comsol_dat/B017fieldprofile_Metglas_noPB_By(x).txt"
ByX_CS = np.loadtxt(ByX_file)

BzX_file = "/Users/alexandraklipfel/Desktop/senior_thesis/comsol_dat/B017fieldprofile_Metglas_noPB_Bz(x).txt"
BzX_CS = np.loadtxt(BzX_file)

#import off data
fileOff = "/Users/alexandraklipfel/Desktop/senior_thesis/data_4-03/off/averaged_off_rot.txt"
datOff = np.loadtxt(fileOff)

#import on data
fileOn = "/Users/alexandraklipfel/Desktop/senior_thesis/data_4-03/on/averaged_on_rot.txt"
datOn = np.loadtxt(fileOn)

#values
plusX = 4150
minX = -3850
plusY = 8150
minY = 150

V1 = -72032
V2 = -52032
V3 = -32032

T0 = 4525

#get BxX_probe data at middle V2
x_1 = []
bx_1 = []
by_1 = []
bz_1 = []

x_1L = []
bx_1L = []
by_1L = []
bz_1L = []

for i in range(len(datOn)):
    if (datOn[i, 2] == V2) and (datOn[i, 1] == plusX):
        x_1 += [(22.7 / 30000) * (datOn[i,0] - T0)]
        bx_1 += [datOn[i,3] - datOff[i,3]]
        by_1 += [datOn[i,5] - datOff[i,5]]
        bz_1 += [datOn[i,7] - datOff[i,7]]
        
    if (datOn[i, 2] == V2) and (datOn[i, 1] == minX):
        x_1L += [(22.7 / 30000) * (-(datOn[i,0] - T0))]
        bx_1L += [datOn[i,3] - datOff[i,3]]
        by_1L += [datOn[i,5] - datOff[i,5]]
        bz_1L += [datOn[i,7] - datOff[i,7]]
        
#need to find value to rescale by
#should be around 4, but isnt quite right
scaleFactor = (((bx_1[1] - bx_1[0])/2) + bx_1[0]) / min(BxX_CS[:, 1])

#we only plor the "righthand side" data
plt.scatter(BxX_CS[0:10, 0]*100, BxX_CS[0:10, 1] * scaleFactor, color="darkblue")
plt.plot(BxX_CS[0:10, 0]*100, BxX_CS[0:10, 1] * scaleFactor, color="darkblue", label="comsol data")

plt.scatter(-BxX_CS[0:10, 0]*100, BxX_CS[0:10, 1] * scaleFactor, color="darkblue")
plt.plot(-BxX_CS[0:10, 0]*100, BxX_CS[0:10, 1] * scaleFactor, color="darkblue")

plt.scatter(x_1, bx_1, color="red")
plt.plot(x_1, bx_1, color="maroon", label="3axis probe data")

plt.scatter(x_1L, bx_1L + (bx_1[1] - bx_1L[1]), color="maroon")
plt.plot(x_1L, bx_1L  + (bx_1[1] - bx_1L[1]), color="maroon")

#plt.scatter(x_1L, bx_1L, color="maroon")
#plt.plot(x_1L, bx_1L, color="maroon")

plt.legend(loc="upper left")
plt.title("Bx vs. x")
plt.xlabel("x (cm)")
plt.ylabel("Bx (mG)")
plt.savefig("BxX_compare.png", dpi=500)
plt.show()

#By vs X

plt.scatter(ByX_CS[0:10, 0]*100, ByX_CS[0:10, 1], color="darkblue")
plt.plot(ByX_CS[0:10, 0]*100, ByX_CS[0:10, 1], color="darkblue", label="comsol data")

plt.legend(loc="upper left")
plt.title("By vs. x")
plt.xlabel("x (cm)")
plt.ylabel("By (mG)")
plt.savefig("ByX_compare.png")
plt.show()

#Bz vs X

plt.scatter(BzX_CS[0:10, 0]*100, BzX_CS[0:10, 1], color="darkblue")
plt.plot(BzX_CS[0:10, 0]*100, BzX_CS[0:10, 1], color="darkblue", label="comsol data")

plt.legend(loc="upper left")
plt.title("Bz vs. x")
plt.xlabel("x (cm)")
plt.ylabel("Bz (mG)")
plt.savefig("BzX_compare.png")
plt.show()
