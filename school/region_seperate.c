#include <stdio.h>
#include <stdlib.h>
//xy 구분 잘하기...
//10026과 유사..
#define BLK 7

int canvas[BLK][BLK] = {
    0,0,0,0,0,0,0,\
    0,0,0,1,0,0,0,\
    0,0,0,1,0,0,0,\
    0,1,1,1,1,0,0,\
    0,0,0,0,0,1,0,\
    0,0,0,0,0,0,0,\
    0,0,0,0,0,0,0};
void sector(int x, int y);
int num=3;

void main(){
    //int x,y;
    //scanf("%d %d",&x,&y);
    printf("---Given Map---\n");
    for(int i =0;i<BLK;i++){
        for(int j=0;j<BLK;j++){
            printf("%d ",canvas[i][j]);
        }
        printf("\n");
    }

    for(int i=0;i<BLK;i++){
        for(int j=0;j<BLK;j++){
            if(canvas[i][j] == 1){
                sector(j,i);
                num++;
            }
        }
    }

    printf("---seperated Map---\n");
    for(int i =0;i<BLK;i++){
        for(int j=0;j<BLK;j++){
            printf("%d ",canvas[i][j]);
        }
        printf("\n");
    }

}

void sector(int x,int y){
    if( x<0 || x==BLK || y<0 || y ==BLK ) return;
    if(canvas[y][x] != 1) return;
    canvas[y][x] = num;
    sector(x-1,y);
    sector(x+1,y);
    sector(x,y-1);
    sector(x,y+1);
}