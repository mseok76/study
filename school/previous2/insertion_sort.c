#include <stdio.h>


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
void indirect_insert_sort(int *a, int *index, int N)
{
    int i, j, t;
    for(i = 0; i<N; i++)
    index[i] = i;
    for(i = 1; i<N; i++)
    {
        t = index[i];
        j = i;
        while( j > 0 && a[index[j-1]] > a[t] )
        {
            index[j] = index[j-1];
            j--;
        }
        index[j] = t;
    }
}

int main(){
    int arr[] = {9,4,5,1,3,9,9};
    int leng = sizeof(arr)/sizeof(int);
    int index[10];

    indirect_insert_sort(arr,index,leng);
    for(int i =0;i<leng;i++){
        printf("%d\t",index[i]);
        printf("%d\n",arr[index[i]]);
    }
}