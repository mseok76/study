#include <stdio.h>
#include <stdlib.h>

#define MAX_NODE 100
int GM[MAX_NODE][MAX_NODE];
//FILE *fp;
// int name2int(char c){
//     return c-'A';
// }
// char int2name(int i){
//     return i+'A';
// }

void input_adjmatrix(int a[][MAX_NODE], int *V, int *E){
    char vertex[3];
    int i, j, k;
    printf("\nInput number of node & edge\n");
    scanf("%d %d", V, E);
    for(i = 0; i<*V; i++){
        for(j = 0; j<*V; j++){
            a[i][ j] = 0;
        }
    }
    for(i = 0; i<*V; i++){
        a[i][i] = 1;
    }
    for(k = 0; k<*E; k++){
        printf("\nInput two node consist of edge ->\n");
        scanf("%s", vertex);
        i = name2int(vertex[0]);
        j = name2int(vertex[1]);
        a[i][ j] = 1;
        a[j][i] = 1;
    }
}

void print_adjmatrix(int a[][MAX_NODE], int V)
{
    int i =0, j;
    printf("%3d",i);
    for(i = 0; i<V; i++)
        printf("%3c", int2name(i));
    printf("\n");
    for(i = 0; i<V; i++){
        printf("%3c", int2name(i));
        for( j = 0; j<V; j++){
            printf("%3d", a[i][ j]);
        }
        printf("\n");
        }
}

