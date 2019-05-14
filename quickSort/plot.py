
#!/usr/bin/env python3


import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import math



gpuFile = open('GPUrun.txt', 'r')
gpuData = gpuFile.read().split('\n')
gpuFile.close()

del gpuData[-1]  #x.remove(len(x)-1)

gpuTimes = []
for i in gpuData:
	 gpuTimes.append( float( i ) )
gpuInputs = []
N = 32;
while N < 1000000:
	gpuInputs.append(N)
	N = N*2
# gpuInputs = [i for i in range(100, 5100, 100)]
plt.plot( gpuInputs,gpuTimes, color="red", label="GPU")



#expCurve = np.polyfit(gpuInputs, np.log(gpuTimes), 1)
#curveOut = [ math.exp(expCurve[1]) * math.exp(expCurve[0]*x)  for x in gpuInputs ]
#expLabel = "e^ (" + str( round(expCurve[0],5) ) + "*x " + str( round(expCurve[1],2) )+")"
# plt.plot( gpuInputs, curveOut , color="orange", label=expLabel)




cpuFile = open('CPUrun.txt', 'r')
cpuData = cpuFile.read().split('\n')
cpuFile.close()

del cpuData[-1]  #x.remove(len(x)-1)

cpuTimes = []
for i in cpuData:
	 cpuTimes.append( float( i ) )
cpuInputs = [i for i in range(10000, 1060000, 50000)]
plt.plot( cpuInputs,cpuTimes, color="blue", label="CPU")

#expCurve = np.polyfit(cpuInputs, np.log(cpuTimes), 1)
#curveOut = [ math.exp(expCurve[1]) * math.exp(expCurve[0]*x)  for x in cpuInputs ]
#expLabel = "e^ (" + str( round(expCurve[0],5) ) + "*x " + str( round(expCurve[1],2) )+")"
# plt.plot( cpuInputs, curveOut , color="purple", label=expLabel)



'''
if True:
    coeffs = np.polyfit( inputs, times, 2 )
    plt.plot(inputs, \
			[ coeffs[2] + coeffs[1]*n + coeffs[0]*n**2 for n in inputs], \
			color="green", label="Nice Quadratic")
    print("nice polynomial is", coeffs[2] ,"+", coeffs[1],"*x + " \
			, coeffs[0] , "x^2")

>>> x = numpy.array([1, 7, 20, 50, 79])
>>> y = numpy.array([10, 19, 30, 35, 51])
>>> numpy.polyfit(numpy.log(x), y, 1)
array([ 8.46295607,  6.61867463])
# y ≈ 8.46 log(x) + 6.62

>>> x = numpy.array([10, 19, 30, 35, 51])
>>> y = numpy.array([1, 7, 20, 50, 79])
>>> numpy.polyfit(x, numpy.log(y), 1)
array([ 0.10502711, -0.40116352])
#    y ≈ exp(-0.401) * exp(0.105 * x) = 0.670 * exp(0.105 * x)
# (^ biased towards small values)
>>> numpy.polyfit(x, numpy.log(y), 1, w=numpy.sqrt(y))
array([ 0.06009446,  1.41648096])
#    y ≈ exp(1.42) * exp(0.0601 * x) = 4.12 * exp(0.0601 * x)
# (^ not so biased)

'''

plt.xlabel("input size N")
plt.ylabel("Time(sec)")
plt.legend()
plt.title("Quick Sort")

plt.show()








