#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <complex>
#include <cmath>
#include <sys/time.h>

#define NUM_COLORS 143
#define THREADS_PER_BLOCK 960

using namespace cv;
using namespace std;

__const__ char* COLORNAMES[NUM_COLORS] = {
	"INDIANRED", "LIGHTCORAL", "SALMON", "DARKSALMON", "LIGHTSALMON", "CRIMSON", "RED", "FIREBRICK", "DARKRED", "PINK",
"LIGHTPINK","HOTPINK","DEEPPINK","MEDIUMVIOLETRED","PALEVIOLETRED","LIGHTSALMON","CORAL","TOMATO","ORANGERED",
"DARKORANGE","ORANGE","GOLD","YELLOW","LIGHTYELLOW","LEMONCHIFFON","LIGHTGOLDENRODYELLOW","PAPAYAWHIP","MOCCASIN",
"PEACHPUFF","PALEGOLDENROD","KHAKI","DARKKHAKI","LAVENDER","THISTLE","PLUM","VIOLET","ORCHID","FUCHSIA","MAGENTA",
"MEDIUMORCHID","MEDIUMPURPLE","REBECCAPURPLE","BLUEVIOLET","DARKVIOLET","DARKORCHID","DARKMAGENTA","PURPLE","INDIGO",
"SLATEBLUE","DARKSLATEBLUE","MEDIUMSLATEBLUE","GREENYELLOW","CHARTREUSE","LAWNGREEN","LIME","LIMEGREEN","PALEGREEN",
"LIGHTGREEN","MEDIUMSPRINGGREEN","SPRINGGREEN","MEDIUMSEAGREEN","SEAGREEN","FORESTGREEN","GREEN","DARKGREEN","YELLOWGREEN",
"OLIVEDRAB","OLIVE","DARKOLIVEGREEN","MEDIUMAQUAMARINE","DARKSEAGREEN","LIGHTSEAGREEN","DARKCYAN","TEAL","AQUA","CYAN",
"LIGHTCYAN","PALETURQUOISE","AQUAMARINE","TURQUOISE","MEDIUMTURQUOISE","DARKTURQUOISE","CADETBLUE","STEELBLUE","LIGHTSTEELBLUE",
"POWDERBLUE","LIGHTBLUE","SKYBLUE","LIGHTSKYBLUE","DEEPSKYBLUE","DODGERBLUE","CORNFLOWERBLUE","MEDIUMSLATEBLUE","ROYALBLUE",
"BLUE","MEDIUMBLUE","DARKBLUE","NAVY","MIDNIGHTBLUE","CORNSILK","BLANCHEDALMOND","BISQUE","NAVAJOWHITE","WHEAT","BURLYWOOD",
"TAN","ROSYBROWN","SANDYBROWN","GOLDENROD","DARKGOLDENROD","PERU","CHOCOLATE","SADDLEBROWN","SIENNA","BROWN","MAROON","WHITE",
"SNOW","HONEYDEW","MINTCREAM","AZURE","ALICEBLUE","GHOSTWHITE","WHITESMOKE","SEASHELL","BEIGE","OLDLACE","FLORALWHITE","IVORY",
"ANTIQUEWHITE","LINEN","LAVENDERBLUSH","MISTYROSE","GAINSBORO","LIGHTGRAY","SILVER","DARKGRAY","GRAY","DIMGRAY","LIGHTSLATEGRAY",
"SLATEGRAY","DARKSLATEGRAY","BLACK" };

__constant__ uchar RED[NUM_COLORS] =
{ 205,240,250,233,255,220,255,178,139,255,255,255,255,199,219,255,255,255,255,255,255,255,255,255,255,250,255,255,255,238,240,189,230,
216,221,238,218,255,255,186,147,102,138,148,153,139,128,75,106,72,123,173,127,124,0,50,152,144,0,0,60,46,34,0,0,154,107,128,85,102,143,
32,0,0,0,0,224,175,127,64,72,0,95,70,176,176,173,135,135,0,30,100,123,65,0,0,0,0,25,255,255,255,255,245,222,210,188,244,218,184,205,210,
139,160,165,128,255,255,240,245,240,240,248,245,255,245,253,255,255,250,250,255,255,220,211,192,169,128,105,119,112,47,0 };

__constant__ uchar GREEN[NUM_COLORS] =
{ 92,128,128,150,160,20,0,34,0,192,182,105,20,21,112,160,127,99,69,140,165,215,255,255,250,250,239,228,218,232,230,183,230,191,160,130,112,
0,0,85,112,51,43,0,50,0,0,0,90,61,104,255,255,252,255,205,251,238,250,255,179,139,139,128,100,205,142,128,107,205,188,178,139,128,255,255,
255,238,255,224,209,206,158,130,196,224,216,206,206,191,144,149,104,105,0,0,0,0,25,248,235,228,222,222,184,180,143,164,165,134,133,105,69,
82,42,0,255,250,255,255,255,248,248,245,245,245,245,250,255,235,240,240,228,220,211,192,169,128,105,136,128,79,0 };

__constant__ uchar BLUE[NUM_COLORS] =
{ 92,128,114,122,122,60,0,34,0,203,193,180,147,133,147,122,80,71,0,0,0,0,0,224,205,210,213,181,185,170,140,107,250,216,221,238,214,255,255,211,
219,153,226,211,204,139,128,130,205,139,238,47,0,0,0,50,152,144,154,127,113,87,34,0,0,50,35,0,47,170,139,170,139,128,255,255,255,238,212,208,
204,209,160,180,222,230,230,235,250,255,255,237,238,225,255,205,139,128,112,220,205,196,173,179,135,140,143,96,32,11,63,30,19,45,42,0,255,250,
240,250,255,255,255,245,238,220,230,240,240,215,230,245,225,220,211,192,169,128,105,153,144,79,0 };




__global__ void frqColorArrayBuilder(uchar* gFREQINDEXES, uchar* gpixelBArr,uchar* gpixelGArr,uchar* gpixelRArr, int totalPixels) {	
	uint minIndex = 0;
	int freqIndexesSize = blockIdx.x*blockDim.x + threadIdx.x;;
	uchar avg;
	if(freqIndexesSize < totalPixels)
	{		
		//initialize minAvg to the first color
		uchar minAvg = (fabsf(gpixelBArr[freqIndexesSize] - BLUE[0]) + fabsf(gpixelGArr[freqIndexesSize]) + fabsf(gpixelRArr[freqIndexesSize])) / 3;
		
			for (int i = 1; i < NUM_COLORS; i++) //iterate through the array of 143 color codes
			{
				//calculate the avg for the passed pixel values and the current color code (i)
				avg = (fabsf(gpixelBArr[freqIndexesSize] - BLUE[i]) + fabsf(gpixelGArr[freqIndexesSize]- GREEN[i]) + fabsf(gpixelRArr[freqIndexesSize] - RED[i])) / 3;

				//find our lowest avg
				if (avg < minAvg) //if compareNum is less than lowest min (min0)
				{
					minIndex = i; //save the index of the color code from the COLORNAMES array
					minAvg = avg; //save the new minAvg for future comparisons
				}
			}
			gFREQINDEXES[freqIndexesSize] = minIndex; //populate the FREQINDEXES array
	}
}



struct timeval start, end;
void starttime() {
	gettimeofday(&start, 0);
}

void endtime(const char* c) {
   gettimeofday( &end, 0 );
   double elapsed = ( end.tv_sec - start.tv_sec ) * 1000.0 + ( end.tv_usec - start.tv_usec ) / 1000.0;
   printf("%s: %f ms\n", c, elapsed); 
}

int main( int argc, char** argv )
{	//get BGR values from image
    Mat image;
    image = imread("greens.jpg", CV_LOAD_IMAGE_COLOR);   // Read the file
    if(! image.data ) { // Check for invalid input
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }
	int totalPixels = image.total();
	
	uchar* pixelBArr = (uchar*) malloc(totalPixels * sizeof(uchar)); //array holding image's pixel B values
	uchar* pixelGArr = (uchar*) malloc(totalPixels * sizeof(uchar)); //array holding image's pixel G values
	uchar* pixelRArr = (uchar*) malloc(totalPixels * sizeof(uchar));; //array holding image's pixel R values
	int currentBSize = 0;
	int currentGSize = 0;
	int currentRSize = 0;
	int currentIndex = 0;
	int i;
    for (i = 0; i < 3*image.total(); i++)
	{
		if(currentIndex == 0) { //if a B value
			currentIndex++;
			pixelBArr[currentBSize] = (uchar)image.data[i]; //put the pixel's b value into b array
			currentBSize++;
		}
		else if(currentIndex == 1) { //if a G value
			currentIndex++;
			pixelGArr[currentGSize] = (uchar)image.data[i];//put the pixel's g value into g array
			currentGSize++;
		}
		else if(currentIndex == 2) { //if a R value
			currentIndex++;
			pixelRArr[currentRSize] = (uchar)image.data[i];//put the pixel's r value into r array
			currentRSize++;
			currentIndex = 0;
		}
		else{
			printf("currentIndex is not 0 1 or 2");
			currentIndex = 0;
		}
		
	}
	///////////////////////////////////////////////////////////////////
	starttime(); //START THE TIMER
	//Create array of totalPixels size that will hold each pixel's closest HTML color RGB value
	uchar* FREQINDEXES = (uchar*) malloc(totalPixels * sizeof(uint)); //array for each pixel's closest color index	
	uchar* gpu_pixelBArr = (uchar*) malloc(totalPixels * sizeof(uchar));
	uchar* gpu_pixelGArr = (uchar*) malloc(totalPixels * sizeof(uchar));
	uchar* gpu_pixelRArr = (uchar*) malloc(totalPixels * sizeof(uchar));
	
	//allocate GPU memory
	uchar* gpu_FREQINDEXES;
	cudaMalloc(&gpu_FREQINDEXES, totalPixels*sizeof(uchar));
	cudaMalloc(&gpu_pixelBArr, totalPixels*sizeof(uchar));
	cudaMalloc(&gpu_pixelGArr, totalPixels*sizeof(uchar));
	cudaMalloc(&gpu_pixelRArr, totalPixels*sizeof(uchar));
	
	
	//copy cpu to gpu
	cudaMemcpy(gpu_FREQINDEXES , FREQINDEXES, totalPixels*sizeof( uchar ) , cudaMemcpyHostToDevice );
	cudaMemcpy(gpu_pixelBArr , pixelBArr, totalPixels*sizeof(uchar) , cudaMemcpyHostToDevice );
	cudaMemcpy(gpu_pixelGArr , pixelGArr, totalPixels*sizeof(uchar) , cudaMemcpyHostToDevice );
	cudaMemcpy(gpu_pixelRArr , pixelRArr, totalPixels*sizeof(uchar) , cudaMemcpyHostToDevice );
	
	
	//gpu function call
	int numblocks = totalPixels / THREADS_PER_BLOCK;
	if(totalPixels % THREADS_PER_BLOCK != 0) {
		numblocks++;
	}
	frqColorArrayBuilder<<<numblocks , THREADS_PER_BLOCK>>>(gpu_FREQINDEXES, gpu_pixelBArr, gpu_pixelGArr, gpu_pixelRArr, totalPixels);
	
	
	//copy gpu to cpu
	cudaMemcpy(FREQINDEXES , gpu_FREQINDEXES, totalPixels*sizeof( uchar ) , cudaMemcpyDeviceToHost );
	cudaMemcpy(pixelBArr , gpu_pixelBArr, totalPixels*sizeof(uchar) , cudaMemcpyDeviceToHost );
	cudaMemcpy(pixelGArr , gpu_pixelGArr, totalPixels*sizeof(uchar) , cudaMemcpyDeviceToHost );
	cudaMemcpy(pixelRArr , gpu_pixelRArr, totalPixels*sizeof(uchar) , cudaMemcpyDeviceToHost );
	
	
	cudaFree(gpu_FREQINDEXES);
	cudaFree(gpu_pixelBArr);
	cudaFree(gpu_pixelGArr);
	cudaFree(gpu_pixelRArr);
	free(pixelBArr);
	free(pixelGArr);
	free(pixelRArr);
	
	///////////////////////////////////////////////////////////////
	int* COUNTER = (int*) malloc(NUM_COLORS * sizeof(int)); //array acting as a counter for how many times each of the HTML color indexes appears in FREQINDEXES
	//initialize all COUNTER values to 0
	for(int i = 0; i < NUM_COLORS; i++) {
		COUNTER[i] = 0; 
	}
	
	///////////////////////////////////////////////////////////////////////
	//populate COUNTER array to show how many times each color shows up in image
	for (int i = 0;  i < NUM_COLORS; i++) { //COLORNAMES (indexes) 
		for(int j = 0; j < totalPixels; j++) { //FREQINDEXES 
			if (i == FREQINDEXES[j]) {
				COUNTER[i] = COUNTER[i] + 1;
			}
		}
	}

	int max1Index = 300;
	int max2Index = 300;
	int max3Index = 300;
	int max4Index = 300;
	int max5Index = 300;
	int max6Index = 300;
	
	int maxIndex;
	////////////////////////////////////find top 6///////////////////////////////
	for(int i = 1; i < 7; i++) 
	{
		maxIndex = 0;
		//make sure you don't check an index you already assigned
		while(maxIndex == max1Index || maxIndex == max2Index || maxIndex == max3Index || maxIndex == max4Index || maxIndex == max5Index || maxIndex == max6Index)
		{
			maxIndex++;
		}
		for(int j = 1; j < NUM_COLORS; j++) 
		{
			//make sure you don't check an index you already assigned
			if(j == max1Index || j == max2Index || j == max3Index || j == max4Index || j == max5Index || j == max6Index) {
				;
			}
			else if(COUNTER[maxIndex] <= COUNTER[j]) {
				maxIndex = j;
			}
		}
		if(i == 1) {
			max1Index = maxIndex;
		}
		if(i == 2) {
			max2Index = maxIndex;
		}
		if(i == 3) {
			max3Index = maxIndex;
		}
		if(i == 4) {
			max4Index = maxIndex;
		}
		if(i == 5) {
			max5Index = maxIndex;
		}
		if(i == 6) {
			max6Index = maxIndex;
		}
	}

	///////////////////////////////////////////////////////
	//PRINT COLOR PALETTE
	printf("*********************************GPU***********************************\n");
	printf("Now printing the color palette of this image:\n");
	//if image has 6 colors or more:
	if(COUNTER[max1Index] != 0 && COUNTER[max2Index] != 0 && COUNTER[max3Index] != 0 
		&& COUNTER[max4Index] != 0 && COUNTER[max5Index] != 0 && COUNTER[max6Index] != 0) {
		printf("1- color#%d: %s- %d pixels\n", max1Index + 1, COLORNAMES[max1Index], COUNTER[max1Index]);
		printf("2- color#%d: %s- %d pixels\n", max2Index + 1, COLORNAMES[max2Index], COUNTER[max2Index]);
		printf("3- color#%d: %s- %d pixels\n", max3Index + 1, COLORNAMES[max3Index], COUNTER[max3Index]);
		printf("4- color#%d: %s- %d pixels\n", max4Index + 1, COLORNAMES[max4Index], COUNTER[max4Index]);
		printf("5- color#%d: %s- %d pixels\n", max5Index + 1, COLORNAMES[max5Index], COUNTER[max5Index]);
		printf("6- color#%d: %s- %d pixels\n", max6Index + 1, COLORNAMES[max6Index], COUNTER[max6Index]);
	}
	//if image has 5 colors only
	else if(COUNTER[max1Index] != 0 && COUNTER[max2Index] != 0 && COUNTER[max3Index] != 0 
		&& COUNTER[max4Index] != 0 && COUNTER[max5Index] != 0) {
		printf("1- color#%d: %s- %d pixels\n", max1Index + 1, COLORNAMES[max1Index], COUNTER[max1Index]);
		printf("2- color#%d: %s- %d pixels\n", max2Index + 1, COLORNAMES[max2Index], COUNTER[max2Index]);
		printf("3- color#%d: %s- %d pixels\n", max3Index + 1, COLORNAMES[max3Index], COUNTER[max3Index]);
		printf("4- color#%d: %s- %d pixels\n", max4Index + 1, COLORNAMES[max4Index], COUNTER[max4Index]);
		printf("5- color#%d: %s- %d pixels\n", max5Index + 1, COLORNAMES[max5Index], COUNTER[max5Index]);
		printf("image only contains 5 colors\n");
	}	
	//if image has 4 colors only
	else if(COUNTER[max1Index] != 0 && COUNTER[max2Index] != 0 && COUNTER[max3Index] != 0 
		&& COUNTER[max4Index] != 0) {
		printf("1- color#%d: %s- %d pixels\n", max1Index + 1, COLORNAMES[max1Index], COUNTER[max1Index]);
		printf("2- color#%d: %s- %d pixels\n", max2Index + 1, COLORNAMES[max2Index], COUNTER[max2Index]);
		printf("3- color#%d: %s- %d pixels\n", max3Index + 1, COLORNAMES[max3Index], COUNTER[max3Index]);
		printf("4- color#%d: %s- %d pixels\n", max4Index + 1, COLORNAMES[max4Index], COUNTER[max4Index]);
		printf("image only contains 4 colors\n");
	}
	else if(COUNTER[max1Index] != 0 && COUNTER[max2Index] != 0 && COUNTER[max3Index] != 0) { //if image has 3 colors only
		printf("1- color#%d: %s- %d pixels\n", max1Index + 1, COLORNAMES[max1Index], COUNTER[max1Index]);
		printf("2- color#%d: %s- %d pixels\n", max2Index + 1, COLORNAMES[max2Index], COUNTER[max2Index]);
		printf("3- color#%d: %s- %d pixels\n", max3Index + 1, COLORNAMES[max3Index], COUNTER[max3Index]);
		printf("image only contains 3 colors\n");
	}	
	else if(COUNTER[max1Index] != 0 && COUNTER[max2Index] != 0) { //if image has 2 colors only
		printf("1- color#%d: %s- %d pixels\n", max1Index + 1, COLORNAMES[max1Index], COUNTER[max1Index]);
		printf("2- color#%d: %s- %d pixels\n", max2Index + 1, COLORNAMES[max2Index], COUNTER[max2Index]);
		printf("image only contains 2 colors\n");
	}
	else if(COUNTER[max1Index] != 0) { //if image has 1 colors only
		printf("1- color#%d: %s- %d pixels\n", max1Index + 1, COLORNAMES[max1Index], COUNTER[max1Index]);
		printf("image only contains 1 color\n");
	}
	
	free(COUNTER);
	
	endtime("GPU");
    return 0;
}
