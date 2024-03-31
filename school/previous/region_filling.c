#include <stdio.h>
#include <stdlib.h>
#define BLK 7
void Fill(int x, int y);

//int* canvas;
int canvas[BLK][BLK] = \
    {{0,0,0,0,0,0,0}\
    ,{0,0,1,1,1,0,0}\
    ,{0,1,0,0,0,1,0}\
    ,{0,1,0,0,0,1,0}\
    ,{0,1,0,0,1,0,0}\
    ,{0,0,1,1,1,0,0}\
    ,{0,0,0,0,0,0,0}};

void main(){
    int n,x,y;
    //printf("BLK = ");
    //scanf("%d",&n);
    //canvas = (int*)calloc(n*n,sizeof(int));
    printf("Seed(x,y) : ");
    scanf("%d %d",&y,&x);

    printf("Original\n");
    for(int i=0;i<BLK;i++){
        for(int j=0;j<BLK;j++){
            printf("%d ",canvas[j][i]);
        }
        printf("\n");
    }

    printf("Filled\n");
    Fill(x,y);
    printf("flag");
    for(int i=0;i<BLK;i++){
        for(int j=0;j<BLK;j++){
            printf("%d ",canvas[j][i]);
        }
        printf("\n");
    }
}

void Fill(int x,int y){
    canvas[x][y] = 1;
    if(x-1 >=0){
        if(canvas[x-1][y] != 1) Fill(x-1,y);
    }
    if(x+1<BLK){
        if(canvas[x+1][y] != 1) Fill(x+1,y);
    }
    if(y-1>=0){
        if(canvas[x][y-1] != 1) Fill(x,y-1);
    }
    if(y+1<BLK){
        if(canvas[x][y+1] != 1) Fill(x,y+1);
    }
}