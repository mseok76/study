#include "header/queue.c"

void main(){
    int k;
    init_queue();
    int arr[6] = {3,6,9,1,6,3};
    for(int i=0;i<6;i++){
        put(arr[i]);
    }
    printf("%d\n",tail->prev->key);

    printf("%d\n",get());

    int arr2[5] = {4,8,7,2,0};
    for(int i=0;i<5;i++){
        put(arr2[i]);
    }
    print_queue();

}