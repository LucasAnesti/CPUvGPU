
#!/usr/bin/env python3


import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import math



gpuFile = open('occ.txt', 'r')
gpuData = gpuFile.read().split('\n')
gpuFile.close()

del gpuData[-1]  #x.remove(len(x)-1)

gpuTimes = []
for i in gpuData:
	 gpuTimes.append( float( i ) )
gpuInputs = []
N = 1;
while N < 31:
	gpuInputs.append(str(N))
	N = N+1
# gpuInputs = [i for i in range(100, 5100, 100)]
plt.plot( gpuInputs,gpuTimes, color="red", label="GPU")


print(gpuInputs)

plt.xlabel("input size 2^N")
plt.ylabel("Time(sec)")
plt.legend()
plt.title("Occupancy")

plt.show()








