
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
	 		   const Point& center1, const Point& center2, bool*& moved) 
{
	int p =  blockIdx.x * blockDim.x + threadIdx.x;

	//for( int p = 0; p < dataSize; ++p)
	if(p < dataSize)
	{
		float d1 = getDistance(center1, data[p]);
		float d2 = getDistance(center2, data[p]);
		int oldGroup = data[p].group;

		if (d1 < d2) data[p].group = 1;
		else data[p].group = 2;

		if( data[p].group != oldGroup )
		{
			moved[p] = true;
		}	
	}
}

__global__
void findMoved(bool* moved, int dataSize, bool& pointMoved)
{
	int index = 0;
	while( index < dataSize && ! pointMoved )
	{
		if(moved[index] == true){ pointMoved = true ; }
		index++;
	}
}

__global__
void updateGroup(Point*& data, int dataSize, 
	 			float& sumX1, float& sumY1, 
				float& sumX2, float& sumY2, int& count1, int& count2) 
{

/*	int p =  blockIdx.x * blockDim.x + threadIdx.x;

	if( p < dataSize )
	{
		if( data[p].group == 1)
		{
			sumX1 += data[p].x; sumY1 += data[p].y;
			count1++;
		}
		else
		{
			sumX2 += data[p].x; sumY2 += data[p].y;
			count2++;
		}
	}*/
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
	Point* dataTemp  = new Point[dataSize];




	// Number of threads in target GPU is 1024, detrmine blocks
	int blockSize = 1024;
	int blockNum = (dataSize + blockSize - 1) / blockSize;


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
		dataTemp[i + min2]=p;
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

	// 2 random points over the whole domain and range
	Point center1, center2; 
	center1.x = min1 + rand() % (max2-min1);
	center1.y = min1 + rand() % (max2-min1);
	center2.x = min1 + rand() % (max2-min1);
	center2.y = min1 + rand() % (max2-min1);

	bool pointMoved = true;
	while( pointMoved )
	{

		std::cout << "Center1 = (" << center1.x << ", " 
							   	   << center1.y << ")\n";
		std::cout << "Center2 = (" << center2.x << ", " 
							   	   << center2.y << ")\n";

		pointMoved = false;
		// set the "moved" status for all the points to false
		setFalse<<<blockNum, blockSize>>>(moved, dataSize);
		cudaDeviceSynchronize();

		
		/*bool* movedCopy = new bool[dataSize];
		cudaMemcpy(movedCopy, moved, dataSize*sizeof(bool), cudaMemcpyDeviceToHost);
		for( int i = 0; i < dataSize; ++i)
		{
			std::cout << movedCopy[i] << "\n";
		}
		cudaDeviceSynchronize();	*/

		// compared to C++: replaced loop for determining membership with a 
		// kernel. Instead, now there is a loop that goes over a bool array.
		// GPU performance should be better.
		findGroup<<<blockNum, blockSize>>>(data, dataSize, center1, center2, moved);
		cudaDeviceSynchronize();

		findMoved<<<1, 1>>>(moved, dataSize, pointMoved);
		cudaDeviceSynchronize();
	
		std::cout << pointMoved << " \n";
		if( pointMoved )
		{
			float sumX1 = 0, sumY1 = 0, sumX2 = 0, sumY2 = 0;
			int count1 = 1, count2 = 1;
			/*
			for( int p = 0; p < dataSize; ++p )
			{
				if( data[p].group == 1)
				{
					sumX1 += data[p].x; sumY1 += data[p].y;
					count1++;
				}
				else
				{
					sumX2 += data[p].x; sumY2 += data[p].y;
					count2++;
				}
			}
			*/
			//updateGroup<<<blockNum, blockSize>>>(
			//		data, dataSize, sumX1, sumY1, sumX2, sumY2, count1, count2);
			//cudaDeviceSynchronize();
	
			center1.x = sumX1 / count1;
			center1.y = sumY1 / count1;
			center2.x = sumX2 / count2;
			center2.y = sumY2 / count2;
		}

		//std::cin.get();

	}

	std::cout << "---Comparison---:\n";
	std::cout << "Expected1 = (" << expected1.x << ", " 
								 << expected1.y << ")\n";
	std::cout << "Expected2 = (" << expected2.x << ", " 
								 << expected2.y << ")\n";

	std::cout << "Center1 = (" << center1.x << ", " 
							   << center1.y << ")\n";
	std::cout << "Center2 = (" << center2.x << ", " 
							   << center2.y << ")\n";



	//cudaFree(&data);
	//cudaFree(&moved);
	//delete [] dataTemp;

}








