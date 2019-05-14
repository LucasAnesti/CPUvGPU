#include <iostream>
#include <cstdlib>
#include <ctime>
void printArray(int *array, int n)
{
    for (int i = 1; i < n + 1; ++i)
        std::cout << array[i] << ", ";
}

void quickSort(int *array, int low, int high)
{
    int i = low;
    int j = high;
    int pivot = array[(i + j) / 2];
    int temp;

    while (i <= j)
    {
        while (array[i] < pivot)
            i++;
        while (array[j] > pivot)
            j--;
        if (i <= j)
        {
            temp = array[i];
            array[i] = array[j];
            array[j] = temp;
            i++;
            j--;
        }
    }
    if (j > low)
        quickSort(array, low, j);
    if (i < high)
        quickSort(array, i, high);
}
int main(int argc, char *argv[])
{
    if (argc < 1)
	std::cout << "Looking for input n elements." << std::endl;
    int N = atoi(argv[1]);
    int array[N];
    srand(2047);
    std::cout << "Running Quick Sort on " << N << " elements." << std::endl;
    std::cout << "Initializing data:" << std::endl;
   
    for (int i = 1; i < N; i++){
        array[i] = rand() % N;
    }
    quickSort(array, 0, N);

    std::cout << "After Quick Sort :" << std::endl;
    printArray(array, 10);
    return (0);
}
