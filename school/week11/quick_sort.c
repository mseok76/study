#include <stdio.h>
#include <stdlib.h>

#define MAX_CNT 10

int arr[MAX_CNT];

void quick_sort(int *a, int N)
{  
    int v, t, k;
    int i, j;
    if(N > 1) // termination
    {
        // for(int i =0;i<MAX_CNT;i++){
        //     printf("%5d",arr[i]);
        // }
        k = rand()%N;
        // printf("\trand = %d\n",k);
        v = a[k];

        t = a[k];
        a[k] = a[N-1];
        a[N-1] = t;

        i = -1;
        j = N;
        while(1){
            while(a[++i] < v);
            while(a[--j] > v);
            if(i >= j) break;
            t = a[i];
            a[i] = a[j];
            a[j] = t;
        }
        t = a[i];
        a[i] = a[N-1];
        a[N-1] = t;
        quick_sort(a, i);
        quick_sort(a+i+1, N-i-1);
    }

}

//non-recursion version
// void quick_sort1(int *a, int N)
// {
// int v, t;
// int i, j;
// int l, r;
// init_stack();
// l = 0;
// r = N-1;
// push(r);
// push(l);
// while(!is_stack_empty()){
// l = pop();
// r = pop();
// if(r-l+1 > 1) // termination
// {
// v = a[r];
// i = l-1;
// j = r;
// while(1){
// while(a[++i] < v);
// while(a[--j] > v);
// if(i >= j) break;
// t = a[i];
// a[i] = a[j];
// a[j] = t;
// }
// t = a[i];
// a[i] = a[r];
// a[r] = t;
// push(r);
// push();
// }
// }
// }




int main(){

    for(int i =0;i<MAX_CNT;i++){
        arr[i] = rand()%(2*MAX_CNT);
        printf("%5d",arr[i]);
    }
    printf("\n");
    quick_sort(arr,MAX_CNT);
    printf("\n");
    for(int i =0;i<MAX_CNT;i++){
        printf("%5d",arr[i]);
    }
}