### chapter 03

----

#### 1.核函数调用

```c
// hello_world
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

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


```

```
//output
hello, world
```



+ `__global__`: 告诉编译器函数应该编译为在设备上运行，而不是在主机上运行，函数`kernel()`将交给编译设备代码的编译器
+ `<<<num1, num2>>>`将函数标记为Device code，将主机代码发送到一个编译器而将设备代码发送到另一个编译器，告知运行时如何启动设备代码



#### 2.传递参数

```c
// add
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>


__global__ void	add(int a, int b, int *c)
{
	*c = a + b;
}

int main()
{
	int c;
	int* dev_c;
	cudaMalloc((void**)&dev_c, sizeof(int));

	add<<<1, 1 >>> (2, 7, dev_c);

	cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost);

	printf("2 + 7 = %d\n", c);

	cudaFree(dev_c);

	return 0;
}

```

```
//output
2 + 7 = 9
```

+ 当设备执行任何有用的操作时，都需要分配内存

采用`cudaMalloc ( void** devPtr, size_t size )`进行内存分配，告知cuda运行时在设备上分配内存

+ 参数1：指向用于保存新分配内存地址的变量
+ 参数2： 分配内存的大小

采用`cudaFree ( void* devPtr )`释放设备上分配的内存

+ 参数：需要释放的设备内存的指针

采用`cudaMemcpy ( void* dst, const void* src, size_t count, cudaMemcpyKind kind )`访问设备上的内存

+ 参数1：源指针
+ 参数2：目标指针
+ 参数3：内存大小
+ 参数4：指定设备内存究竟是源指针还是目标指针，`cudaMemcpyDeviceToHost`将告知运行时源指针是设备指针而目标指针是主机指针

不能在主机代码中对`cudaMalloc()`返回的指针进行解引用(Dereference)，主机代码不能将其读取或者写入内存。



#### 3.查询设备

需要知道设备拥有多少内存以及具备的功能将非常有用，cuda提供了`cudaGetDeviceCount ( int* count )`

+   参数：指向整数的指针，用于存储设备数量。

在调用之后，可以对每个设备进行迭代获得各个设备的相关信息其返回一个`cudaDeviceProp`类型的结构，包含设备的相关属性。

+ `char name[256]` - 设备名称。

+ `size_t totalGlobalMem` - 全局内存的总字节数。

+ `size_t sharedMemPerBlock` - 每个线程块可用的共享内存大小（以字节为单位）。

+ `int regsPerBlock` - 每个线程块可用的寄存器数量。

+ `int warpSize` - warp 的大小（即并行处理的线程数量，通常为 32）。

+ `size_t memPitch` - 内存间距（用于线性内存的最大宽度）。

+ `int maxThreadsPerBlock` - 每个线程块可支持的最大线程数。

+ `int maxThreadsDim[3]` - 线程块维度的最大值。

+ `int maxGridSize[3]` - 网格的最大维度大小。

+ `int clockRate` - 设备时钟频率（千赫兹）。

+ `size_t totalConstMem` - 常量内存的总字节数。

+ `int major`、`int minor` - 设备的计算能力（主次版本号）。

+ `int multiProcessorCount` - 多处理器的数量。

+ `int maxThreadsPerMultiProcessor` - 每个多处理器支持的最大线程数。

```c
// cuda_info
#include<stdio.h>


int main() {
    cudaDeviceProp prop;

    int count;
    cudaGetDeviceCount(&count);
    for(int i = 0; i < count; i ++ ) {
        cudaGetDeviceProperties(&prop, i);
        printf("  --- General Information for device %d ---\n", i);
        printf("Name: %s\n", prop.name);
        printf("Compute capability: %d.%d\n", prop.major, prop.minor);
        printf("Clock rate: %d\n", prop.clockRate);
        printf("Device copy overlap: ");
        if(prop.deviceOverlap)
            printf("Enabled\n");
        else
            printf("Disabled\n");
        printf("Kernel execution timeout: ");
        if(prop.kernelExecTimeoutEnabled)
            printf("Enabled\n");
        else
            printf("Disabled\n");
        printf("  --- Memory Information for device %d ---\n", i);
        printf("Total global mem: %ld\n", prop.totalGlobalMem);
        printf("Total constant Mem: %ld\n", prop.totalConstMem);
        printf("Max mem pitch: %ld\n", prop.memPitch);
        printf("Texture Alignment: %ld\n", prop.textureAlignment);
        printf("  --- MP Information for device %d ---\n", i);
        printf("Multiprocessor count: %d\n", prop.multiProcessorCount);
        printf("Shared mem per mp: %ld\n", prop.sharedMemPerBlock);
        printf("Registers per mp: %d\n", prop.regsPerBlock);
        printf("Threads in warp: %d\n", prop.warpSize);
        printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
        printf("Max thread dimensions: (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("Max grid dimensions: (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("\n");
    }

}
```

```
// output
  --- General Information for device 0 ---
Name: NVIDIA GeForce RTX 3060 Laptop GPU
Compute capability: 8.6
Clock rate: 1425000
Device copy overlap: Enabled
Kernel execution timeout: Enabled
  --- Memory Information for device 0 ---
Total global mem: 2146959360
Total constant Mem: 65536
Max mem pitch: 2147483647
Texture Alignment: 512
  --- MP Information for device 0 ---
Multiprocessor count: 30
Shared mem per mp: 49152
Registers per mp: 65536
Threads in warp: 32
Max threads per block: 1024
Max thread dimensions: (1024, 1024, 64)
Max grid dimensions: (2147483647, 65535, 65535)
```

`cudaGetDeviceProperties ( cudaDeviceProp* prop, int device )`获取指定 cuda 设备的属性信息，并将其存储在 `cudaDeviceProp` 结构体中。

+ 参数1：`cudaDeviceProp* prop` - 指向 `cudaDeviceProp` 结构体的指针，用于存储设备属性信息。

+ 参数2：`int device` - 设备 ID，表示需要查询属性的设备编号。

#### 4.设备属性的使用

cuda为查找特定要求的设备提供了自动的迭代方式，并且可以设置对应的工作设备

```c
// cuda_info_use
#include "stdio.h"

int main() {
    cudaDeviceProp prop;
    int dev;

    cudaGetDevice(&dev);
    printf("ID of current CUDA device: %d\n", dev);

    memset(&prop, 0, sizeof(cudaDeviceProp));
    prop.major = 1;
    prop.minor = 3;
    cudaChooseDevice(&dev, &prop);
    printf("ID of CUDA device closest to revision 1.3: %d\n", dev);
    cudaSetDevice(dev);
}
```

```
// output
ID of current CUDA device: 0
ID of CUDA device closest to revision 1.3: 0
```

`cudaGetDevice ( int* device )`：获取当前线程在使用的 cuda 设备 ID。

+ 参数：`int* device`:指向整数的指针，用于存储当前线程绑定的设备 ID。成功调用后，该整数包含当前使用的设备 ID

`cudaChooseDevice(int* device, const cudaDeviceProp* prop)`:根据指定的属性，选择与需求最匹配的 CUDA 设备。

+ 参数1：`int* device` - 指向整数的指针，用于存储选定设备的 ID。函数成功执行后，该整数包含符合需求的设备 ID。

+ 参数2： `const cudaDeviceProp* prop` - 指向 `cudaDeviceProp` 结构体的指针，描述对设备的需求属性。

`cudaSetDevice ( int device )`：设置当前线程使用的 CUDA 设备。

+ 参数：`int device` - 设备 ID，表示要选择的 CUDA 设备。



#### 