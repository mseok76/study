#include<stdio.h>
#include<stdlib.h>
#define BLK 20
void middle(int x1, int x2, int y1, int y2);
int** arr;

void main(){
    arr = (int**)calloc(BLK*BLK,sizeof(int));
    for(int i =0;i<BLK;i++){
        arr[i] = (int*)calloc(BLK,sizeof(int));
    }

    int a,b,c,d;
    printf("x1, y1, x2, y2\n");
    scanf("%d %d %d %d",&a,&b,&c,&d);
    middle(a,b,c,d);
    for(int i=0;i<20;i++){
        for(int j=0;j<20;j++){
            printf("%d ", arr[j][i]);
        }
        printf("\n");
    }

    for(int i=0;i<BLK;i++){
        free(arr[i]);
    }
    free(arr);
}

void middle(int x1, int y1, int x2, int y2){
    int x_mid = (x1+x2)/2;
    int y_mid = (y1+y2)/2;

    arr[x_mid][y_mid] = 1;
    if((x2-x1) <= 1 && (y2-y1) <=1) return;
    middle(x1,y1,x_mid,y_mid);
    middle(x_mid,y_mid,x2,y2);
}