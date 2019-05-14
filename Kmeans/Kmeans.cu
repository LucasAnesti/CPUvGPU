
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




int main()
{
	// deliberate partitioning into clusters
	// testing the algorithm
	const int min1 = 0, max1 = 25;
	const int min2 = 25, max2 = 50;
	const int groupSize = 25;

	unsigned seed = time(0);
	srand(seed);


	

	// these are the centers we expect to get at the end
	Point expected1, expected2;
	float sumX = 0, sumY = 0;
	std::vector<Point> data;

	for(int i = 0; i < groupSize; ++i)
	{
		Point p;
		p.x = min1 + rand() % (max1 - min1);
		sumX += p.x;
		p.y = min1 + rand() % (max1 - min1);
		sumY += p.y;
		data.push_back(p);
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
		data.push_back(p);
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
		for( int p = 0; p < data.size(); ++p) //(auto p : data )
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
			for( auto p : data )
			{
				if( p.group == 1)
				{
					sumX1 += p.x; sumY1 += p.y;
					count1++;
				}
				else
				{
					sumX2 += p.x; sumY2 += p.y;
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








