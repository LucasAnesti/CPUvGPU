
#!/usr/bin/env python3


import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import math



gflopFile = open('perf.txt', 'r')
gflopData = gflopFile.read().split('\n')
gflopFile.close()

del gflopData[-1]  #x.remove(len(x)-1)

gflopTimes = []
for i in gflopData:
	 gflopTimes.append( float( i ) )
gflopInputs = [i for i in range(32, 32*25, 32)]
plt.plot( gflopInputs,gflopTimes, color="red", label="Gflop/s")




timesFile = open('times.txt', 'r')
timesData = timesFile.read().split('\n')
timesFile.close()

del timesData[-1]  #x.remove(len(x)-1)

times = []
for i in timesData:
	 times.append( 100*float( i ) )
timesInputs = gflopInputs
plt.plot( timesInputs,times, color="blue", label="time")




msecFile = open('msec.txt', 'r')
msecData = msecFile.read().split('\n')
msecFile.close()

del msecData[-1]  #x.remove(len(x)-1)

msecTimes = []
for i in msecData:
	 msecTimes.append( 100*float( i ) )
msecInputs = gflopInputs
plt.plot( msecInputs,msecTimes, color="orange", label="msec")






plt.xlabel("N x N dimension")
plt.ylabel("Time(x100 ms)")
plt.legend()
plt.title("Matrix Multiplication Gflop/s")

plt.savefig("myfigure.png")

plt.show()











