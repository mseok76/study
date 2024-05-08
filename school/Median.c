#include <stdio.h>
#include <stdlib.h>

void read_file(int *input, int height, int width, char filename[])
{
    int x, y;
    FILE *fp;
    fp = fopen(filename, "r");
    for(y = 0; y<height; y++){
        for(x = 0; x<width; x++){
            fscanf(fp, "%d", &input[y*width+x]);
        }
        fscanf(fp, "\n");
    }
    fclose(fp);
}

void write_file(int *output, int height, int width, char filename[])
{
    int x, y;
    FILE *fp;
    FILE *fp2;
    fp = fopen(filename, "w");
    fp2 = fopen("img.png","wb");
    for(y = 0; y<height; y++){
        for(x = 0; x<width; x++){
            fprintf(fp, "%d\t", output[y*width+x]);
            fprintf(fp, "%d\t", output[y*width+x]);
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
}

void insert_sort(int *a, int N)
{
    int i, j, t;
    for(i = 1; i<N; i++)
    {
        t = a[i];
        j = i;
        while(j > 0 && a[j-1] > t)
        {
            a[j] = a[j-1];
            j--;
        }
        a[j] = t;
    }
}

void median_filtering(int* input, int* output, int height, int width){
    int kernel_size = 5;
    int count;
    int* sortarr = (int*)malloc(sizeof(int)*kernel_size*kernel_size);
    for(int i =0;i<height;i++){
        for(int j =0;j<width;j++){
            count = 0;
            for(int y = i-2;y<=i+2;y++){
                for(int x=j-2;x<=j+2;x++){
                    if(x < 0 || x>=width || y<0 || y>= height){
                        continue;
                    }
                    sortarr[count] = input[y*width+x];
                    count++;
                }
            }
            //sort
            insert_sort(sortarr, count);
            output[i*width+j] = sortarr[count/2];
        }
    }
    free(sortarr);
}


int main(){
    int* input;
    int* output;
    int width = 45,height = 44;
    input = (int*)malloc(sizeof(int)*width*height);
    output = (int*)malloc(sizeof(int)*width*height);
    char filename[] = "noisy_data.txt";
    read_file(input,height,width,filename);
    median_filtering(input,output,height,width);
    char outputname[] = "output.txt";
    write_file(output,height,width,outputname);

    free(output);
    free(input);
}
