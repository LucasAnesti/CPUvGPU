
#!/usr/bin/env python3


import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import math



staticFile = open('static_times', 'r')
staticData = staticFile.read().split('\n')
staticFile.close()

del staticData[-1]  #x.remove(len(x)-1)

staticTimes = []
for i in staticData:
	 staticTimes.append( float( i ) )
staticInputs = [i for i in range(32, 1025, 32)]
plt.plot( staticInputs,staticTimes, color="red", label="Blocks = 1, threads = (32,1024)")




dynamicFile = open('dynamic_times', 'r')
dynamicData = dynamicFile.read().split('\n')
dynamicFile.close()

del dynamicData[-1]  #x.remove(len(x)-1)

dynamicTimes = []
for i in dynamicData:
	 dynamicTimes.append( float( i ) )
dynamicInputs = []
i = 32
while i < 1025:
	dynamicInputs.append(i)
	i *= 2
plt.plot( dynamicInputs,dynamicTimes, color="blue", label="Blocks * threads = 1024")




plt.xlabel("Dimension")
plt.ylabel("Time(msec)")
plt.legend()
plt.title("Vector addition: Increasing thread/block count")

plt.show()








