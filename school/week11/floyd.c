#include <stdio.h>
#include "header/queue.c"

#define MAX_NODE 100
#define INF 10000000

typedef struct _head{
    int count;
    struct node *next;
}head;
head network[MAX_NODE];

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

//need fix
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

void set_count_indegree(head net[], int V)
{
    int i, j;
    int count;
    node *t;
    for(i = 0; i<V; i++){
        count = 0;
        for( j = 0; j<V; j++)
            for(t = net[j].next; t; t->next)
                if(t->vertex == i) count++;
            net[i].count = count;
    }
}

void topsort(head net[], int V){
    int i, j, k;
    node *ptr;
    init_stack();
    set_count_indegree(net, V);
    for(int i =0;i<V;i++){
        printf("%3d",net[i]);
    }
    printf("\n\n");

    for(i = 0; i<V; i++)
        if(!net[i].count) push(i);
    for(i = 0; i<V; i++){
        if(stack_empty()) return -1;
        else{
            j = pop();
            printf("%c, ", int2name(j));
            for(ptr = net[j].next; ptr; ptr=ptr->next){
                k = ptr->vertex;
                net[k].count--;
                if(!net[k].count)
                    push(k);
            }
        }
    }
}

//--- 위 까진 정상, import 할 내용 수정 필요

int main(){
    int V,E;
    scanf("%d %d",&V,&E);
    input_adjlist(GL,&V,&V);
    for(int i =0;i<V;i++){
        network[i].next = GL[i];
    }
    topsort(network, V);
}

// 12 14 AB BC CD CE CF CG EK FH FI FJ HK IK JK KL
// TODO node 재정의, directd list input 구현, queue 가져오기