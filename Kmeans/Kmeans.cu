
#include <ctime> 
#include <cstdlib> 
#include <iostream>
#include <string>
#include <cmath>
#include <vector>


class Point
{
public:
	float x = 0;
	float y = 0;
	int group = 1;
};


__device__
float getDistance(const Point& p1, const Point& p2)
{
	float s = sqrt(   pow( (p1.x - p2.x), 2) 
				    + pow( (p1.y - p2.y), 2) 
				  );
	return s;
}



// a CUDA kernel that sets all the entries in an array to a specified value
__global__
void setFalse(bool*& membershipChanged, int dataSize)
{
	int index =  blockIdx.x * blockDim.x + threadIdx.x;
	if( index < dataSize )
	{
		membershipChanged[index] = false;
	}
}


// a CUDA kernel that determines the group a point belongs to and
// updates the "moved" status of each point if membership changes from 
// one group to another
__global__
void findGroup(Point*& data, int dataSize, 
	 		   Point* dev_centers, bool*& moved) 
{
	int p =  blockIdx.x * blockDim.x + threadIdx.x;

	//for( int p = 0; p < dataSize; ++p)
	if(p < dataSize)
	{
		float d1 = getDistance(dev_centers[0], data[p]);
		float d2 = getDistance(dev_centers[1], data[p]);
		int oldGroup = data[p].group;

		if (d1 < d2) data[p].group = 1;
		else data[p].group = 2;

		if( data[p].group != oldGroup )
		{
			moved[p] = true;
		}	
	}
}


// a CUDA kernel that traverses the "moved" status for each point
// and sets the dev_pointMoved variable to false if one is found
__global__
void findMoved(bool* moved, int dataSize, bool* dev_pointMoved)
{
	int index = 0;
	while( index < dataSize && ! dev_pointMoved[0] )
	{
		if(moved[index] == true){ dev_pointMoved[0] = true ; }
		index++;
	}
}



// if at least one point has moved during the last iteration of the 
// kmeans algorithm, then go through each point in the data set and
// update their groups
__global__
void updateGroup(Point*& data, int dataSize, float* sums, int* counts) 
{

	int p =  blockIdx.x * blockDim.x + threadIdx.x;

	if( p < dataSize )
	{
		if( data[p].group == 1)
		{
			sums[0] += data[p].x; sums[1] += data[p].y;
			counts[0]++;
		}
		else
		{
			sums[2] += data[p].x; sums[3] += data[p].y;
			counts[1]++;
		}
	}
}







int main(int argc, char* argv[])
{

	if( argc < 2 )
	{
		std::cout << "Usage:  ./a.out <data points> \n";
		exit(1);
	}

	unsigned seed = time(0);
	srand(seed);


	
	// std::vector<Point> data;
	// deliberate partitioning into clusters
	const int dataSize = atoi(argv[1]);
	const int groupSize = dataSize/2;
	const int min1 = 0, max1 = groupSize;
	const int min2 = max1+1, max2 = dataSize;

	

	// these are the centers we expect to get at the end
	Point expected1, expected2;
	float sumX = 0, sumY = 0;

	Point* data;
	cudaMallocManaged( &data, dataSize * sizeof(Point) );
	bool* moved;
	cudaMallocManaged( &moved, dataSize * sizeof(bool) );



	// Number of threads in target GPU is 1024, detrmine blocks
	int blockSize = 1024;
	int blockNum = (dataSize + blockSize - 1) / blockSize;

	Point* dataTemp  = new Point[dataSize];
	for(int i = 0; i < groupSize; ++i)
	{
		Point p;
		p.x = min1 + rand() % (max1 - min1);
		sumX += p.x;
		p.y = min1 + rand() % (max1 - min1);
		sumY += p.y;
		dataTemp[i]=p;
	}
	expected1.x = sumX/groupSize;
	expected1.y = sumY/groupSize;


	sumX = 0, sumY = 0;
	for(int i = 0; i < groupSize; ++i)
	{
		Point p;
		p.x = min2 + rand() % (max2 - min2);
		sumX += p.x;
		p.y = min2 + rand() % (max2 - min2);
		sumY += p.y;
		dataTemp[i + groupSize]=p;
	}
	expected2.x = sumX/groupSize;
	expected2.y = sumY/groupSize;
	

	cudaMemcpy(data,dataTemp, dataSize * sizeof( Point ), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
	/*
	for( int i = 0; i < dataSize; ++i)
	{
		std::cout << dataTemp[i].x << ", " << dataTemp[i].y << ".  " <<dataTemp[i].group << "\n";
	} 
	
	std::cout << "-----------------------\n";
	Point* dataCopy = new Point[dataSize];
	cudaMemcpy(dataCopy, data, dataSize * sizeof( Point ), cudaMemcpyDeviceToHost);
	for( int i = 0; i < dataSize; ++i)
	{
		std::cout << dataCopy[i].x << ", " << dataCopy[i].y << ".  " <<dataCopy[i].group << "\n";
	} 
	*/
//-----------------------------------------------------------

	// Random points over the whole domain and range, declared at main memory
	Point* centers = new Point[2]; 
	centers[0].x = min1 + rand() % (max2-min1);
	centers[0].y = min1 + rand() % (max2-min1);
	centers[1].x = min1 + rand() % (max2-min1);
	centers[1].y = min1 + rand() % (max2-min1);

	// Centroids declared in GPU memory 
	Point* dev_centers;
	cudaMallocManaged(&dev_centers, 2 * sizeof(Point));
	// Transfer data from main to gpu mem
	cudaMemcpy(dev_centers, centers, 2 * sizeof( Point ), cudaMemcpyHostToDevice);



	// sums and counts are used to update the centers and the groups for
	// each point
	float* sums = new float[4];
	for(int s = 0; s < 4; ++s) sums[s] = 0;
	float* dev_sums;
	cudaMallocManaged(&dev_sums, 4 * sizeof(float));

	int* counts = new int[2];
	counts[0] = 1; counts[1] = 1;
	int* dev_counts;
	cudaMallocManaged(&dev_counts, 2 * sizeof(int));


	bool* pointMoved = new bool[1]; 
	pointMoved[0] = true;

	// gpu version of pointMoved variable
	bool* dev_pointMoved;
	cudaMallocManaged(&dev_pointMoved, sizeof(bool));

	while( pointMoved[0] )
	{

		std::cout << "Center1 = (" << centers[0].x << ", " 
							   	   << centers[0].y << ")\n";
		std::cout << "Center2 = (" << centers[1].x << ", " 
							   	   << centers[1].y << ")\n";

		pointMoved[0] = false;
		// set the "moved" status for all the points to false
		setFalse<<<blockNum, blockSize>>>(moved, dataSize);
		cudaDeviceSynchronize();



		// compared to C++: replaced loop for determining membership with a 
		// kernel. Instead, now there is a loop that goes over a bool array.
		// GPU performance should be better.
		findGroup<<<blockNum, blockSize>>>(data, dataSize, dev_centers, moved);
		cudaDeviceSynchronize();

		// copy from pointMoved to dev_pointMoved to use in kernel
		cudaMemcpy(dev_pointMoved, pointMoved, sizeof( bool ), cudaMemcpyHostToDevice);

		findMoved<<<1, 1>>>(moved, dataSize, dev_pointMoved);
		cudaDeviceSynchronize();
		cudaMemcpy(pointMoved, dev_pointMoved, sizeof( bool ), cudaMemcpyDeviceToHost);


		// Code segment to see if the "moved" array isbeing updated properly
		/*bool* movedCopy = new bool[dataSize];
		cudaMemcpy(movedCopy, moved, dataSize*sizeof(bool), cudaMemcpyDeviceToHost);
		for( int i = 0; i < dataSize; ++i)
		{
			std::cout << movedCopy[i] << "\n";
		}
		cudaDeviceSynchronize();*/
	


		std::cout << pointMoved[0] << " \n";
		if( pointMoved[0] )
		{

			cudaMemcpy(dev_sums, sums, 4 * sizeof( float ), cudaMemcpyHostToDevice);
			cudaMemcpy(dev_counts, counts, 2 * sizeof( int ), cudaMemcpyHostToDevice);
	
			// Since a point has moved, update every points group
			updateGroup<<<blockNum, blockSize>>>(data, dataSize, dev_sums, dev_counts);
			cudaDeviceSynchronize();

			cudaMemcpy(sums, dev_sums, 4 * sizeof( float ), cudaMemcpyDeviceToHost);
			cudaMemcpy(counts, dev_counts, 2 * sizeof( int ), cudaMemcpyDeviceToHost);

			centers[0].x = sums[0] / counts[0];
			centers[0].y = sums[1] / counts[0];
			centers[1].x = sums[2] / counts[1];
			centers[1].y = sums[3] / counts[1];

		}

		//std::cin.get();
		cudaMemcpy(centers,dev_centers, 2 * sizeof( Point ), cudaMemcpyDeviceToHost);

	}

	std::cout << "---Comparison---:\n";
	std::cout << "Expected1 = (" << expected1.x << ", " 
								 << expected1.y << ")\n";
	std::cout << "Expected2 = (" << expected2.x << ", " 
								 << expected2.y << ")\n";

	std::cout << "Center1 = (" << centers[0].x << ", " 
							   << centers[0].y << ")\n";
	std::cout << "Center2 = (" << centers[1].x << ", " 
							   << centers[1].y << ")\n";



	cudaFree(&data);
	cudaFree(&moved);
	delete [] dataTemp;
	delete [] pointMoved;
	cudaFree( &dev_pointMoved);

	delete [] sums;
	cudaFree( &dev_sums);
	delete [] counts;
	cudaFree( &dev_counts);

}








