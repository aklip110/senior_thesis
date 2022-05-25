#NOTE: z is vertical

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
#plt.rcParams["figure.figsize"] = 6,4

fileOff = "/Users/alexandraklipfel/Desktop/senior_thesis/data_3-30/off/V-34500/averaged_off-34500.txt"
datOff = np.loadtxt(fileOff)
print("Data Off Shape: ", datOff.shape)
lengthOff = datOff.shape[0]
print("Number of Off Data Points: ", lengthOff)

fileOn = "/Users/alexandraklipfel/Desktop/senior_thesis/data_3-30/on/V-34500/averaged_on-34500.txt"
datOn = np.loadtxt(fileOn)
print("Data On Shape: ", datOn.shape)
lengthOn = datOn.shape[0]
print("Number of On Data Points: ", lengthOn)

T0 = 6400

print(datOn[:, 5])
#print(datOff[:, 5])

#subtract the off field values from the on field values
#TO DO: propagate errors

#final data:
# y axis values
dat = datOn.copy()
y = []
bx = []
by = []
bz = []

yleft = []
bxleft = []
byleft = []
bzleft = []
for i in range(lengthOn):
    #print(datOn[:, 5])
    #print(i)
    if (dat[i, 2] == -34500) and (dat[i, 1] == -50) :
        y += [(22.7 / 30000) * (dat[i,0] - T0)]
        dat[i, 3] = datOn[i, 3] - datOff[i, 3]
        bx += [dat[i, 3]]
        dat[i, 5] = datOn[i, 7] - datOff[i, 7]
        by += [dat[i, 5]]
        dat[i, 7] = -datOn[i, 5] + datOff[i, 5]
        bz += [dat[i, 7]]
        #print(datOn[i, 5])
    if (dat[i, 2] == -34500) and (dat[i, 1] == 7950) :
        yleft += [(22.7 / 30000) * (-(dat[i,0] - T0))]
        dat[i, 3] = -datOn[i, 3] + datOff[i, 3]
        bxleft += [dat[i, 3]]
        dat[i, 5] = -datOn[i, 7] + datOff[i, 7]
        byleft += [dat[i, 5]]
        dat[i, 7] = -datOn[i, 5] + datOff[i, 5]
        bzleft += [dat[i, 7]]
        #print(datOn[i, 5])


# make y plots---these are actually labeled "z" because i'm switching to the lab coordinates that Umit's using
#along B0
plt.scatter(y, bx, color="cyan")
plt.scatter(yleft, bxleft, color="darkblue")

plt.xlabel("y (cm)")
plt.ylabel("Bx (mG)")
plt.title("delta Bx (along B0) vs y")
plt.savefig("Bx_y-deltaB.png", dpi=1000)
plt.show()

#orthogonal
plt.scatter(y, by, color="cyan")
plt.scatter(yleft, byleft, color="darkblue")

plt.xlabel("y (cm)")
plt.ylabel("By (mG)")
plt.title("delta By vs y")
plt.savefig("By_y-deltaB.png", dpi=1000)
plt.show()
        
#vertical
plt.scatter(y, bz, color="cyan")
plt.scatter(yleft, bzleft, color="darkblue")

plt.xlabel("y (cm)")
plt.ylabel("Bz (mG)")
plt.title("delta Bz (vertical) vs y")
plt.savefig("Bz_y-deltaB.png", dpi=1000)
plt.show()

# y axis values
dat = datOn.copy()
x = []
bxX = []
byX = []
bzX = []

xleft = []
bxXleft = []
byXleft = []
bzXleft = []
for i in range(lengthOn):
    #print(datOn[:, 5])
    #print(i)
    if (dat[i, 2] == -34500) and (dat[i, 1] == 11950) :
        x += [(22.7 / 30000) * (dat[i,0] - T0)]
        dat[i, 3] = datOn[i, 7] - datOff[i, 7]
        bxX += [dat[i, 3]]
        dat[i, 5] = -datOn[i, 3] - -datOff[i, 3]
        byX += [dat[i, 5]]
        dat[i, 7] = -datOn[i, 5] + datOff[i, 5]
        bzX += [dat[i, 7]]
        #print(datOn[i, 5])
    if (dat[i, 2] == -34500) and (dat[i, 1] == 3950) :
        xleft += [(22.7 / 30000) * (-(dat[i,0] - T0))]
        dat[i, 3] = -datOn[i, 7] + datOff[i, 7]
        bxXleft += [dat[i, 3]]
        dat[i, 5] = datOn[i, 3] - datOff[i, 3]
        byXleft += [dat[i, 5]]
        dat[i, 7] = -datOn[i, 5] + datOff[i, 5]
        bzXleft += [dat[i, 7]]
        #print(datOn[i, 5])
        
#along B0
plt.scatter(x, bxX, color="cyan")
plt.scatter(xleft, bxXleft, color="darkblue")

plt.xlabel("x (cm)")
plt.ylabel("Bx (mG)")
plt.title("delta Bx (along B0) vs x")
plt.savefig("Bx_x-deltaB.png", dpi=1000)
plt.show()

#orthogonal
plt.scatter(x, byX, color="cyan")
plt.scatter(xleft, byXleft, color="darkblue")

plt.xlabel("x (cm)")
plt.ylabel("By (mG)")
plt.title("delta By vs x")
plt.savefig("By_x-deltaB.png", dpi=1000)
plt.show()
        
#vertical
plt.scatter(x, bzX, color="cyan")
plt.scatter(xleft, bzXleft, color="darkblue")

plt.xlabel("x (cm)")
plt.ylabel("Bz (mG)")
plt.title("delta Bz (vertical) vs x")
plt.savefig("Bz_x-deltaB.png", dpi=1000)
plt.show()

# now the new file should be ready to be plotted--note that the errors are currently meaningless and need to be fixed
np.savetxt("averaged_deltaB.txt", data, delimiter=" ")
