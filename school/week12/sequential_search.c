//need generalize

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MAX_NODE 100
#define FIND 19

int seq_search(int key, int *a, int *num)
{
    int i = 0;
    while(a[i] != key && i < *num) i++;
    return (i < *num ? i : -1);
    }

int seq_insert(int key, int *a, int *num)
{
    a[(*num)++] = key;
    return 1; 
}

int seq_delete(int key, int *a, int *num)
{
    int p, i;
    if(*num > 0)
    {
        if((p = seq_search(key, a, num)) < 0) return -1;
        for(i = p+1; i<*num; i++)
            a[i-1] = a[i];
        (*num)--;
        return p;
    }
    return -1;
}

int seq_f_search(int key, int *a, int *num)
{
    int p, i;
    if((p=seq_search(key, a, num)) < 0)
        return -1;
    for(i = p-1; i>=0; i--)
        a[i+1] = a[i];
    a[0] = key;
    return 0;
}

int main(){
    int arr[MAX_NODE];
    srand((unsigned int)time(NULL));
    for(int i =0;i<MAX_NODE;i++){
        arr[i] = rand()%30;
    }
    int count = MAX_NODE;
    int a;
    a = seq_search(FIND, arr,&count);
    printf("1: %d\n",a);
    seq_insert(FIND, arr,&count);
    a=seq_f_search(FIND,arr,&count);
    printf("2: %d\n",a);
    a=seq_f_search(FIND, arr,&count);
    printf("3: %d\n",a);
    seq_delete(FIND, arr,&count);
    a=seq_f_search(FIND, arr,&count);
    printf("4: %d\n",a);
}