#NOTE: variable names don't match agreed-upon lab coordinate names. bz==vertical, bx==along B0, by==orthogonal. In lab coords, z and y are swapped so that y is vertical. The plot titles and file names reflect this.

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
#plt.rcParams["figure.figsize"] = 6,4

fileOff = "/Users/alexandraklipfel/Desktop/senior_thesis/data_3-29/deltaB/averaged_off.txt"
datOff = np.loadtxt(fileOff)
print("Data Off Shape: ", datOff.shape)
lengthOff = datOff.shape[0]
print("Number of Off Data Points: ", lengthOff)

fileOn = "/Users/alexandraklipfel/Desktop/senior_thesis/data_3-29/deltaB/averaged_on.txt"
datOn = np.loadtxt(fileOn)
print("Data On Shape: ", datOn.shape)
lengthOn = datOn.shape[0]
print("Number of On Data Points: ", lengthOn)


print(datOn[:, 5])
#print(datOff[:, 5])

#first have to transfrom the field values into the lab frame (this was done in the averaging file but only for the plots not saved to the output text files)

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
    if (dat[i, 2] == -5000) and (dat[i, 1] == 3900) :
        y += [dat[i,0]]
        dat[i, 3] = datOn[i, 3] - datOff[i, 3]
        bx += [dat[i, 3]]
        dat[i, 5] = datOn[i, 7] - datOff[i, 7]
        by += [dat[i, 5]]
        dat[i, 7] = -datOn[i, 5] + datOff[i, 5]
        bz += [dat[i, 7]]
        #print(datOn[i, 5])
    if (dat[i, 2] == -5000) and (dat[i, 1] == -4300) :
        yleft += [-dat[i,0]]
        dat[i, 3] = -datOn[i, 3] + datOff[i, 3]
        bxleft += [dat[i, 3]]
        dat[i, 5] = -datOn[i, 7] + datOff[i, 7]
        byleft += [dat[i, 5]]
        dat[i, 7] = -datOn[i, 5] + datOff[i, 5]
        bzleft += [dat[i, 7]]
        #print(datOn[i, 5])


# make y plots---these are actually labeled "z" because i'm switching to the lab coordinates that Umit's using
#along B0
plt.scatter(y, bx, color="red")
plt.scatter(yleft, bxleft, color="blue")

plt.xlabel("z")
plt.ylabel("Bx")
plt.title("delta Bx (along B0) vs z")
plt.savefig("Bx_z-deltaB.png", dpi=1000)
plt.show()

#orthogonal
plt.scatter(y, by, color="red")
plt.scatter(yleft, byleft, color="blue")

plt.xlabel("z")
plt.ylabel("Bz")
plt.title("delta Bz vs z")
plt.savefig("Bz_z-deltaB.png", dpi=1000)
plt.show()
        
#vertical
plt.scatter(y, bz, color="red")
plt.scatter(yleft, bzleft, color="blue")

plt.xlabel("z")
plt.ylabel("By")
plt.title("delta By (vertical) vs z")
plt.savefig("By_z-deltaB.png", dpi=1000)
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
    if (dat[i, 2] == -5000) and (dat[i, 1] == -200) :
        x += [dat[i,0]]
        dat[i, 3] = datOn[i, 7] - datOff[i, 7]
        bxX += [dat[i, 3]]
        dat[i, 5] = -datOn[i, 3] - -datOff[i, 3]
        byX += [dat[i, 5]]
        dat[i, 7] = -datOn[i, 5] + datOff[i, 5]
        bzX += [dat[i, 7]]
        #print(datOn[i, 5])
    if (dat[i, 2] == -5000) and (dat[i, 1] == 7900) :
        xleft += [-dat[i,0]]
        dat[i, 3] = -datOn[i, 7] + datOff[i, 7]
        bxXleft += [dat[i, 3]]
        dat[i, 5] = datOn[i, 3] - datOff[i, 3]
        byXleft += [dat[i, 5]]
        dat[i, 7] = -datOn[i, 5] + datOff[i, 5]
        bzXleft += [dat[i, 7]]
        #print(datOn[i, 5])
        
#along B0
plt.scatter(x, bxX, color="red")
plt.scatter(xleft, bxXleft, color="blue")

plt.xlabel("x")
plt.ylabel("Bx")
plt.title("delta Bx (along B0) vs x")
plt.savefig("Bx_x-deltaB.png", dpi=1000)
plt.show()

#orthogonal
plt.scatter(x, byX, color="red")
plt.scatter(xleft, byXleft, color="blue")

plt.xlabel("x")
plt.ylabel("Bz")
plt.title("delta Bz vs x")
plt.savefig("Bz_x-deltaB.png", dpi=1000)
plt.show()
        
#vertical
plt.scatter(x, bzX, color="red")
plt.scatter(xleft, bzXleft, color="blue")

plt.xlabel("x")
plt.ylabel("By")
plt.title("delta By (vertical) vs x")
plt.savefig("By_x-deltaB.png", dpi=1000)
plt.show()

# now the new file should be ready to be plotted--note that the errors are currently meaningless and need to be fixed
np.savetxt("averaged_deltaB.txt", data, delimiter=" ")
