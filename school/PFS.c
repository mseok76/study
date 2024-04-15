// #include "header/heap.c"
#include <stdio.h>
#include <stdlib.h>

#define INT_MAX 10
#define MAX_NODE 100
#define UNSEEN (-INT_MAX)

typedef struct node{
    int vertex;
    int weight;
    struct node *next;
}node;

node *G[MAX_NODE]; // for storing the graph
int check[MAX_NODE]; // for storing the weight
int parent[MAX_NODE]; // for preserving the tree structure 
FILE *fp;
int nheap = 0; // # of elements in the heap
int heap[MAX_NODE];

void adjust_heap(int h[], int n);


void upheap(int h[], int k)
{
    int v;
    v = h[k];
    while(check[h[k/2]] <= check[v] &&
        k/2 > 0){
        h[k] = h[k/2];
        k /= 2;
    }
    h[k] = v;
}

void downheap(int h[], int k)
{
    int i, v;
    v = h[k];
    while(k <= nheap/2){
        i = k << 1;
        if(i < nheap && check[h[i]] < check[h[i+1]]){
        i++;
        if(check[v] >= check[h[i]]) break;
        h[k] = h[i];
        k = i;
    }
    h[k] = v;
    }
}

int name2int(char c){
    return c-'A';
}
char int2name(int i){
    return i+'A';
}

int pq_update(int h[], int v, int p)
{
    if(check[v] == UNSEEN){
        h[++nheap] = v;
        check[v] = p; // store the weight
        upheap(h, nheap);
        return 1;
    }
    else{
        if(check[v] < p){
            check[v] = p;
            adjust_heap(h, nheap);
            return 1;
        }
    }
}

void input_adjlist(node *g[], int *V, int *E)
{
    char vertex[3];
    int i, j, w;
    node *t;
    printf("\nInput number of nodes & edges\n");
    fscanf(fp, "%d %d", V, E);
    for(i = 0; i<*V; i++)
        g[i] = NULL;
    for( j = 0; j<*E; j++){
        printf("\nInput two nodes of edge and weight -> ");
        fscanf(fp, "%s %d", vertex, &w);
        
        i = name2int(vertex[0]);
        t = (node*)malloc(sizeof(node));
        t ->weight = name2int(vertex[1]);
        t->weight = w;
        t->next = g[i];
        g[i] = t;

        i = name2int(vertex[1]);
        t = (node*)malloc(sizeof(node));
        t->vertex = name2int(vertex[0]);
        t->weight = w;
        t->next = g[i];
        g[i] = t;
    }
}


int pq_empty(){
    if(nheap == 0) return 1;
    return 0;
}

int pq_extract(int h[]){
    int v = h[1];
    h[1] = h[nheap];
    downheap(h, 1);
    return v;
}

void adjust_heap(int h[], int n){
    int k;
    for(k = n/2; k>=1; k--)
        downheap(h, k);
}


void print_adjlist(node *g[], int V){
    int i;
    node *t;
    for(i = 0; i<V; i++){
        printf("\n%c : ", int2name(i));
        for(t = g[i]; t != NULL; t = t->next){
            printf("-> %c:%d ", int2name(t->vertex),t->weight);
            }
    }
}

void print_heap(int h[]){
    int i;
    printf("\n");
    for(i = 1; i<=nheap; i++)
        printf("%c:%d ", int2name(h[i]), check[h[i]]);
}

void pq_init(){
    nheap = 0;
}

void visit(int i){
    printf("%c",int2name(i));
}

void PFS_adjlist(node *g[], int V)
{
    int i;
    node *t;
    pq_init();
    for(i = 0; i<V; i++){
        check[i] = UNSEEN; // set nodes as unseen
        parent[i] = 0; // initialize a tree
    }
    for(i = 0; i<V; i++){
        if(check[i] == UNSEEN){
            parent[i] = -1; // set the root
            pq_update(heap, i, UNSEEN); 
            while(!pq_empty()){
                print_heap(heap);
                i = pq_extract(heap);
                check[i] = -check[i];
                visit(i);
                for(t=g[i]; t != NULL; t=t->next){
                    if(check[t->vertex] < 0){
                        if(pq_update(heap, t->vertex, -t->weight))
                            parent[t->vertex] = i;
                    }
                }
            }
        }
    }
}



void main()
{
    int V, E;
    fp = fopen("graph.txt", "rt");
    input_adjlist(G, &V, &E);
    printf("\nOriginal graph\n");
    print_adjlist(G, V);
    printf("\nVisit order of Minimum Spanning Tree\n");
    PFS_adjlist(G, V);
    //print_tree(parent, V); 
    //printf("\nMinimum Cost is \n");
    //print_cost(check, V);
    fclose(fp);
}
