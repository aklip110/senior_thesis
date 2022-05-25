#we compute deltaB and make plots of Bx, By, Bx, |B| vs x
#for the 4/03 data

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

#import off data
fileOff = "/Users/alexandraklipfel/Desktop/senior_thesis/data_4-14/off/averaged_off_rot-Z.txt"
datOff = np.loadtxt(fileOff)

#import on data
fileOn = "/Users/alexandraklipfel/Desktop/senior_thesis/data_4-14/on/averaged_on_rot-Z.txt"
datOn = np.loadtxt(fileOn)

#Again, there are no intervals, so everything is much simpler

plusX = 4050
minX = -3950
plusY = 8050
minY = 50

V1 = -58654

T0 = -100 #need to verify

#get data
z = (15.2 / 50000) * (datOn[:, 2] - V1)
print(z)
dBx = datOn[:, 3] - datOff[:, 3]
dBy = datOn[:, 5] - datOff[:, 5]
dBz = datOn[:, 7] - datOff[:, 7]

B = np.sqrt(np.square(dBx) + np.square(dBy) + np.square(dBz))

#dBx vs z
plt.scatter(z, dBx, color="maroon")
plt.plot(z, dBx, color="maroon")
      
plt.xlabel("z (cm)")
plt.ylabel("Bx (mG)")
plt.title("delta Bx (along B0) vs z: off")
plt.savefig("Bx_z-deltaB_tcorr.png", dpi=500)
plt.show()

#dBy vs z
plt.scatter(z, dBy, color="maroon")
plt.plot(z, dBy, color="maroon")
      
plt.xlabel("z (cm)")
plt.ylabel("By (mG)")
plt.title("delta  By vs z: off")
plt.savefig("By_z-deltaB_tcorr.png", dpi=500)
plt.show()

#dBz vs z
plt.scatter(z, dBz, color="maroon")
plt.plot(z, dBz, color="maroon")
      
plt.xlabel("z (cm)")
plt.ylabel("Bz (mG)")
plt.title("delta  Bz vs z: off")
plt.savefig("Bz_z-deltaB_tcorr.png", dpi=500)
plt.show()

#dB vs z
plt.scatter(z, B, color="maroon")
plt.plot(z, B, color="maroon")
      
plt.xlabel("z (cm)")
plt.ylabel("|B| (mG)")
plt.title("delta  |B| vs z: off")
plt.savefig("B_z-deltaB_tcorr.png", dpi=500)
plt.show()
