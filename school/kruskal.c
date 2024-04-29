#include <stdio.h>
#include <stdlib.h>

#define MAX_NODE 50
#define MAX_EDGE 100
#define ONLY_FIND 0
#define UNION 1
//rufrhk JK AD FJ HI CD IJ AE QG DF AB

typedef struct edge{
    int v1, v2;
    int weight;
}edge;

int parent[MAX_NODE]; // tree structure for representing the set
int height[MAX_NODE]; // store the height of the tree
int cost = 0;
int nheap = 0; // # of elements in the heap
int heap[MAX_NODE];
FILE* fp;
edge Edge[MAX_EDGE];

void find_init(int elem){
    int i;
    for(i = 0; i<elem; i++)
        parent[i] = -1;
}

int pq_empty(){
    if(nheap == 0) return 1;
    return 0;
}

char int2name(int a){
    return a+'A';
}

int name2int(char a){
    return a - 'A';
}

void union_set(int elem, int asso){
    parent[elem] = asso;
}

void upheap(int h[], int k)     //수정 필요
{
    int v;
    v = h[k];
    while(Edge[h[k/2]].weight >= Edge[v].weight && k/2 >0){
        h[k] = h[k/2];
        k /= 2;
    }
    h[k] = v;
}

void downheap(int h[], int k)       //수정 필요
{
    int i, v;
    v = h[k];
    while(k <= nheap/2){
        i = k << 1;
        if(i < nheap && Edge[h[i]].weight > Edge[h[i+1]].weight){
        i++;
        if(Edge[v].weight <= Edge[h[i]].weight) break;
        h[k] = h[i];
        k = i;
    }
    h[k] = v;
    }
}

void visit(int e){
    printf("%c%c ", int2name(Edge[e].v1), int2name(Edge[e].v2));
    cost += Edge[e].weight;
}


int find_set(int elem, int asso, int flag){
    int i = elem, j = asso;
    while(parent[i] >= 0)
        i = parent[i];
    while(parent[j] >= 0)
        j = parent[ j];
    if(flag == UNION && i != j)
        union_set(i, j);
    return (i != j);
}


void pq_init(){
    nheap = 0;
}

void pq_insert(int h[], int v)
{
    h[++nheap] = v;
    upheap(h, nheap);
}

int pq_extract(int h[])
{
    int v = h[1];
    h[1] = h[nheap--];
    downheap(h, 1);
    return v;
}

void input_edge(edge e[], int *V, int *E){
    char vertex[3];
    int i, j, w;
    printf("\nInput number of nodes and edges\n");
    fscanf(fp, "%d %d", V, E);
    for( j = 0; j<*E; j++){
        printf("\nInput two nodes consisting of edge and its weight ->");
        fscanf(fp, "%s %d", vertex, &w);
        vertex[2] = NULL;
        e[j].v1 = name2int(vertex[0]);
        e[j].v2 = name2int(vertex[1]);
        e[j].weight = w;
    }
}

void kruskal(edge e[], int V, int E){
    int n, val = 0;
    find_init(V);
    pq_init();
    for(n = 0; n<E; n++)
        pq_insert(heap, n);
    n = 0;
    while(!pq_empty()){
        val = pq_extract(heap);
        if(find_set(e[val].v1, e[val].v2, UNION)){
            visit(val);
            n++;
        }
        if(n == V-1)
            break;
    }
}

void main()
{
    int V, E;
    fp = fopen("graph.txt", "rt");
    input_edge(Edge, &V, &E);
    printf("\n\nVisited edge of minimum spanning tree\n");
    kruskal(Edge, V, E);
    printf("\n\nMinimum cost is \n %d\n", cost);
    fclose(fp);
}
