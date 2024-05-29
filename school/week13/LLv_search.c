#include<stdio.h>
#include <stdlib.h>
#include <string.h>
// #include <windows.h>

#define LEN 5

typedef struct _node{
    struct _node *next;
}node;
typedef int (*FCMP)(const void *, const void *);
int lv_search(void *key, void *base, int *num, int width, FCMP fcmp);
int lv_insert(void *key, void *base, int *num, int width, FCMP fcmp);
int lv_delete(void *key, void *base, int *num, int width, FCMP fcmp);
int lfv_search(void *key, void *base, int *num, int width, FCMP fcmp);

void *llv_insert(void *key, node *base, size_t *num, size_t width, FCMP fcmp)
{
    node *t;
    t = (node *)malloc(sizeof(node) + width);
    memcpy(t+1,key,width);
    t->next = base->next;
    base->next = t;
    (*num)++;
    return t;
}

void *llv_search(void *key, node *base, size_t *num, size_t width, FCMP fcmp)
{
    node *t;
    t = base->next;
    while(fcmp(t+1, key) != 0 && t != NULL) t = t->next; 
    //while(fcmp(t+1, key) != 0 && t != NULL)
    //  t = t->next; 
    //  (*idx)++
    return (t == NULL ? NULL : t+1);
}

void main(){
    void *p;
    node *t, *head;
    int size = 0, key = 9;
    int data[LEN] = { 3, 1, 9, 7, 5};
    init_list( ); // using the pointer t
    // Construct your linked list
    // Print out your linked list using for loop and printf()
    for(int i =0;i<5;i++){
        p = llv_insert(data+i,t,&size,sizeof(int),strcmp());
        printf("%d",*((int*)p+1));
    }
    
    p = llv_search(9,t,&size,LEN,strcmp());
    printf("%d",*((int*)p+1));
    // Conduct search with key value
    // Print out the relative position of key value to the starting addr. of LL 

    //------get inital word---
    sprintf(filename,"words.txt");
    read_file(word,WORD_NUM,filename);

    for(y=0;y<WORD_NUM;y++){
        initial[y] = *(word[y]);
    }
    }

//--------------
