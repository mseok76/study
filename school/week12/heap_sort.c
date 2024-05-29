#include<stdio.h>
#include<stdlib.h>
#include<time.h>

#define INT_MAX 10000000
#define SIZE 20

void upheap(int *a, int k)
{
    int v;
    v = a[k];
    a[0] = INT_MAX;
    while(a[k/2] <= v){
        a[k] = a[k/2];
        k /= 2;
    }
    a[k] = v;
}

void downheap(int *a, int N, int k)
{
    int i, v;
    v = a[k];
    while(k<=N/2){
        i = k<<1;
        if(i < N && a[i] < a[i+1]) i++;
        if(v >= a[i]) break;
        a[k] = a[i];
        k = i;
    }
    a[k] = v;
}

void insert(int *a, int *N, int v)
{
    a[++(*N)] = v;
    upheap(a, *N);
}

int extract(int *a, int *N)
{
    int v = a[1];
    a[1] = a[(*N)--];
    downheap(a, *N, 1);
    return v;
}

void heap_sort(int *a, int N)
{
    int i;
    int hn = 0; // # of heap nodes
    for(i = 1; i<=N; i++)
        insert(a, &hn, a[i]);
    for(i = hn; i>=1; i--)
        a[i] = extract(a, &hn);
}

void heap_sort_bottomup(int *a, int N)
{
int k, t;
for(k=N/2;k>=0;k--)
    downheap(a, N, k);
while(N>1){
    t = a[1];
    a[1] = a[N]; 
    a[N]= t;
    downheap(a,--N,1);
}
}

int main(){
    int k;
    int data[SIZE];
    int dataheap[SIZE+1];
    int heap[SIZE+1];
    srand(time(NULL));
    for(k=0;k<SIZE;k++){
        dataheap[k+1] = rand()%10000;

    }
    heap_sort_bottomup(dataheap,SIZE);   //->bottom up
    // heap_sort_bottomup(dataheap,heap,SIZE);
    //bottomup -> 원소 2개
    //topdown -> 원소 3개 dataheap heap SIZE
    printf("\n");
    for(k=0;k<SIZE;k++){
        printf("%-5d",dataheap[k+1]);
    }
    printf("\n");
}
