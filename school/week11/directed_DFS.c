#include <stdio.h>
#include <stdlib.h>

#define MAX_NODE 10

typedef struct _node{
    int vertex;
    struct _node *next;
}node;

node *GL[MAX_NODE];
int check[MAX_NODE];
int top;
int stack[MAX_NODE];

int name2int(char c){
    return c-'A';
}
char int2name(int i){
    return i+'A';
}

void init_stack(){
    top = -1;
}

void push(int a){
    stack[++top] = a;
}

int pop(){
    return stack[top--];
}
void visit(int a){
    printf("%-3c",int2name(a));
}
int stack_empty(){
    if(top == -1){
        return 1;
    }else{
        return 0;
    }
}

void input_adjlist(node *a[], int *V, int *E){
    char vertex[3];
    int i, j;
    FILE* fp = fopen("graph_input.txt","r");
    node *t;
    printf("\nInput number of nodes & edges\n");
    fscanf(fp, "%d %d", V, E);
    for(i = 0; i<*V; i++)
        a[i] = NULL;
    for( j = 0; j<*E; j++){
        printf("\nInput two nodes consisting of edge ->");
        fscanf(fp, "%s", vertex);
        i = name2int(vertex[0]);
        t = (node *)malloc(sizeof(node));
        t->vertex = name2int(vertex[1]);
        t->next = a[i];
        a[i] = t;
    }
}


void DFS_directed(node *a[], int V){
    int i, j;
    node *t;
    init_stack();
    for(i = 0; i<V; i++){
        for( j = 0; j<V; j++)
            check[j] = 0;
        push(i);
        check[i] = 1;
        printf("\n%c : ", int2name(i));
        while(!stack_empty()){
            j = pop();
            visit( j);
            for(t = a[j]; t != NULL; t=t->next){
                if(check[t->vertex] == 0){
                    push(t->vertex);
                    check[t->vertex] = 1;
                }
            }
        }
    }
}

int main(){
    int V,E;
    input_adjlist(GL, &V,&E);
    DFS_directed(GL,V);

}
