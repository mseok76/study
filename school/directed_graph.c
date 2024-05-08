#include <stdio.h>
#include <stdlib.h>

#define MAX_NODE 100

int name2int(char c){
    return c-'A';
}
char int2name(int i){
    return i+'A';
}

void input_adjmatrix(int g[][MAX_NODE], int *V, int *E){
    FILE* fp = fopen("graph_input.txt","r");
    char vertex[3];
    int i, j, k;
    printf("\nInput number of nodes & edges\n");
    fscanf(fp, "%d %d", V, E);
    for(i = 0; i<*V; i++)
        for(j = 0; j<*V; j++)
            g[i][ j] = 0;
    for(i = 0; i<*V; i++)
        g[i][i] = 0;
    for(k = 0; k<*E; k++){
        printf("\nInput two nodes consisting of edge ->");
        fscanf(fp, "%s", vertex);
        i = name2int(vertex[0]);
        j = name2int(vertex[1]);
        g[i][ j] = 1;
    }
}

void warshall(int a[][MAX_NODE], int V){    //2다리 건너 연결된 애만 가능
    for(int i =0;i<V;i++){
        for(int j=0;j<V;j++){
            if(a[i][j] == 1){
                for(int k =0;k<V;k++){
                    if(a[j][k] == 1){
                        a[i][k] = 1;
                    }
                }
            }
        }
    }
}

int main(){
    int V,E;
    int G[MAX_NODE][MAX_NODE];
    input_adjmatrix(G,&V,&E);
    warshall(G,V);
    for(int i=0;i<V;i++){
        printf("%5d",G[0][i]);
    }
    printf("\n");
}