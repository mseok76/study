#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

int intcmp(const void *a, const void *b)
{
return (*(int *)a - *(int *)b);
}

void gen_select_sort(void *base, size_t nelem, size_t width, 
    int (*fcmp)(const void *, const void *))
    {
    void *min;
    int min_idx;
    int i, j;
    min = malloc(width);
    for(int y = 0; y<nelem-1; y++)
    {
        min_idx = y;
        memcpy(min, (char *)base+y*width, width);
        for(int x = y+1; x<nelem; x++)
        {
            if(fcmp(min, (char *)base + x*width) > 0)
            {
                memcpy(min, (char *)base + x*width, width);
                min_idx = x;
            }
        }
        memcpy((char *)base + min_idx*width, (char *)base + y*width, width);
        memcpy((char *)base + y*width, min, width);
    }
    free(min);
}

int main(){
    int arr[1000];
    srand((int)time(NULL));
    for(int i =0;i<1000;i++){
        arr[i] = rand()%1000;
    }
    gen_select_sort(arr,1000,sizeof(int),intcmp);
    for(int i =0;i<1000;i++){
        printf("%d",arr[i]);
        printf("\n");
    }
}