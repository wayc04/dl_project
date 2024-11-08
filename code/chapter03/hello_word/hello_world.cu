#include <stdio.h>


__global__ void	kernel(void)
{
	
}

int main()
{
	kernel << <1, 1 >> > ();
	printf("hello, world");
	return 0;
}

