#include <stdio.h>
#include <stdlib.h>
// #define MAX_NODE 100
//FILE *fp;

typedef struct _node{
    int vertex;
    struct _node *next;
}node;
node *GL[MAX_NODE];

int name2int(char c){
    return c-'A';
}
char int2name(int i){
    return i+'A';
}

void input_adjlist(node *a[], int *V, int *E){
    char vertex[3];
    int i, j;
    node *t;
    printf("\nInput number of node & edge\n");
    scanf("%d %d", V, E);
    for(i = 0; i<*V; i++){
        a[i] = NULL;
    }
    for( j = 0; j<*E; j++){
        printf("\nInput two node consist of edge ->");
        scanf("%s", vertex);
        i = name2int(vertex[0]);
        t = (node *)malloc(sizeof(node));
        t->vertex = name2int(vertex[1]);
        t->next = a[i];
        a[i] = t;
        i = name2int(vertex[1]);
        t = (node *)malloc(sizeof(node));
        t->vertex = name2int(vertex[0]);
        t->next = a[i];
        a[i] = t;
    }
}

void print_adjlist(node *a[], int V){

    for(int i=0;i<V;i++){
        printf("%3c",int2name(i));
        while(a[i] != NULL){    //자기자신도 출력
            printf("%3c",int2name(a[i]->vertex));
            a[i] = a[i]->next;
        }
        printf("\n");
    }
}