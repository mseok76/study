#include <stdio.h>
#include <stdlib.h>
#include "header/graph.h"
// #define MAX_NODE 20
#define INFINITE 10000

int check[MAX_NODE];
int distance[MAX_NODE];
int parent[MAX_NODE];

int V,E;

void input_adjmatrix_Dijkstra(int g[][MAX_NODE], int *V, int *E)
{
    FILE* fp = fopen("graph.txt","r");
	char vertex[3];
	int i, j, k, w;
	printf("\nInput number of nodes and edges\n");
	fscanf(fp, "%d %d", V, E);
	for (i = 0; i < *V; i++)
		for (j = 0; j < *V; j++)
			g[i][j] = 1000;
	for (i = 0; i < *V; i++)
		g[i][i] = 0;

	printf("\nInput two nodes consisting of edge --> ");

	for (k = 0; k < *E; k++) {
		fscanf(fp, "%s %d", vertex, &w);
		i = name2int(vertex[0]);
		j = name2int(vertex[1]);
		g[i][j] = w;
		g[j][i] = w;
	}
}

void dijkstra(int a[][MAX_NODE], int s, int V){
    int x = 0, y, d;
    int i, checked = 0;
    for(x = 0; x < V; x++){
        distance[x] = a[s][x];
        if(x != s) parent[x] = s;
        }
    check[s] = 1;
    checked++;
    print_distance(distance, s, V);
    while(checked < V){
        x = 0;
        while(check[x]) x++;
        for(i = x; i < V; i++)
        if(check[i] == 0 && distance[i] < distance[x]) x = i;
        check[x] = 1;
        checked++;
        for(y = 0; y<V; y++){
            if(x == y || a[x][y] >=INFINITE || check[y]) continue;
            d = distance[x] + a[x][y];
            if(d < distance[y]){
                distance[y] = d;
                parent[y] = x;
            }
        }
        print_distance(distance, x, V);
    }
}

void print_distance(int distance[], int s, int V){
    int x;
    for(int x=0;x<V;x++){
        printf("%3d",distance[x]);
    }
    printf("\n");
}

void print_parent_dijkstra(int parent_dijkstra[], int V){
    int i;
    printf("\n");
    for(i=0;i<V;i++){
        printf("%c: %c\n",int2name(i),int2name(parent_dijkstra[i]));
    }
}

int main(){
    input_adjmatrix_Dijkstra(GM, &V,&E);
    print_distance(distance,0,V);
    printf("\n\n");
    dijkstra(GL,0,V);
    print_parent_dijkstra(parent,V);
}