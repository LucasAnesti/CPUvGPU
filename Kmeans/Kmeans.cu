
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
	int group = 0;
};


float getDistance(const Point& p1, const Point& p2)
{
	float s = sqrt(   pow( (p1.x - p2.x), 2) 
				    + pow( (p1.y - p2.y), 2) 
				  );
	return s;
}


/***********
CUDA kernels
***********/

__global__
void generatePoints(Point*& data,int minvalue, int maxvalue, int dataSize)
{

  	int index =  blockIdx.x * blockDim.x + threadIdx.x;

	if( index < dataSize )
	{
		Point p;
		p.x = minvalue + rand() % (maxvalue - minvalue);
		p.y = minvalue + rand() % (maxvalue - minvalue);
		data[index] = p;
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
	Point* data;
	// deliberate partitioning into clusters
	const int dataSize = atoi(argv[1]);
	const int groupSize = dataSize/2;
	const int min1 = 0, max1 = groupSize;
	const int min2 = max1+1, max2 = dataSize;

    cudaMallocManaged( &data, dataSize * sizeof(Point) );


    int blockSize = 1024;
    int numBlocks = (dataSize + blockSize - 1) / blockSize;
    generatePoints<<<numBlocks,blockSize>>>(data, min1, max1, max1-min1);
    cudaDeviceSynchronize();


	// these are the centers we expect to get at the end
	Point expected1, expected2;
	float sumX = 0, sumY = 0;
	for(int i = 0 + min1; i < max1; ++i)
	{
		sumX += data[i].x;
		sumY += data[i].y;
	}
	expected1.x = sumX/groupSize;
	expected1.y = sumY/groupSize;

    generatePoints<<<numBlocks,blockSize>>>(data, min2, max2, max2-min2);
    cudaDeviceSynchronize();
	sumX = 0, sumY = 0;
	for(int i = 0 + min2; i < max2; ++i)
	{
		sumX += data[i].x;
		sumY += data[i].y;
	}
	expected2.x = sumX/groupSize;
	expected2.y = sumY/groupSize;


//-----------------------------------------------------------

	// 2 random points over the whole domain and range
	Point center1, center2; 
	center1.x = min1 + rand() % (max2-min1);
	center1.y = min1 + rand() % (max2-min1);
	center2.x = min1 + rand() % (max2-min1);
	center2.y = min1 + rand() % (max2-min1);

	float d1 = 0, d2 = 0;

	bool pointMoved = true;
	while( pointMoved )
	{

		std::cout << "Center1 = (" << center1.x << ", " 
							   	   << center1.y << ")\n";
		std::cout << "Center2 = (" << center2.x << ", " 
							   	   << center2.y << ")\n";

		pointMoved = false;
		
		for( int p = 0; p < dataSize; ++p) //(auto p : data )
		{
			d1 = getDistance(center1, data[p]);
			d2 = getDistance(center2, data[p]);
			int oldGroup = data[p].group;

			if (d1 < d2) data[p].group = 1;
			else data[p].group = 2;

			if( data[p].group != oldGroup )
			{
				pointMoved = true;
			}	
		}

	
		if( pointMoved )
		{
			float sumX1 = 0, sumY1 = 0, sumX2 = 0, sumY2 = 0;
			float count1 = 1, count2 = 1;
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
	std::cout << "Center1 = (" << center2.x << ", " 
							   << center2.y << ")\n";


}








