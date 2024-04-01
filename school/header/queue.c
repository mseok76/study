#include <stdio.h>
#include <stdlib.h>

typedef struct _dnode
{
int key;
struct _dnode *prev;
struct _dnode *next;
}dnode;

// (globally defined)
dnode *head, *tail;

void init_queue()
{
    head = (dnode *)calloc(1, sizeof(dnode));
    tail = (dnode *)calloc(1, sizeof(dnode));
    head->prev = head;
    head->next = tail;
    tail->prev = head;
    tail->next = tail;
}

int put(int k)
{
    dnode *t;
    if((t = (dnode*)malloc(sizeof(dnode))) == NULL) //예외 사항
    {
        printf("Out of memory !\n");
        return -1;
    }
    t->key = k;
    tail->prev->next = t;
    t->prev = tail->prev;
    tail->prev = t;
    t->next = tail;
    return k;
}

int get()
{
    dnode *t;
    int k;
    t = head->next;
    if(t == tail)
    {
        printf("Queue underflow\n");
        return -1;
    }
    k = t->key;
    head->next = t->next;
    t->next->prev = head;
    free(t);
    return k;
}

int queue_empty(){
    dnode *t = head->next;
    return (t == tail); 
}