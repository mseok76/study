#include <stdio.h>
#include <stdlib.h>
// #include "header/graph_matrix.c"
// #include "header/graph_list.c"
#include "header/graph.h"
#define MAX_NODE 100

int stack[MAX_NODE];
int top;

void DFS_recur_list(node *a[], int V, int i);
void DFS_adjlist(node *a[], int V);
void nrDFS_adjlist(node *a[], int V);
void init_stack();
int stack_empty();
int pop();
void push(int i);
void visit(int i);

int* check;

void main(){
    int V,E;
    input_adjlist(GL,&V,&E);
    check = (int*)malloc(V*sizeof(int));
    printf("test\n");
    nrDFS_adjlist(GL,V);
    print_adjlist(GL,V);
}

void nrDFS_adjlist(node *a[], int V){
    init_stack();
    int nextnode;
    node *temp;
    for(int i =0;i<V;i++){
        if(check[i] == 0){
            nextnode = i;
            push(nextnode);
        }
        else continue;
        
        do{
            nextnode = pop();
            check[nextnode] = 1;
            visit(nextnode);
            temp = a[nextnode];
            while(temp != NULL){
                if(check[temp->vertex] == 0){
                    push(temp->vertex);
                    check[temp->vertex] = 1;
                }
                temp = temp->next;
            }
        }while(stack_empty() != 1);
    }
}

void DFS_recur_list(node *a[], int V, int i)
{
    node *t;
    check[i] = 1;
    visit(i);
    for(t = a[i]; t != NULL; t=t->next)
        if(check[t->vertex] == 0)
        DFS_recur_list(a, V, t->vertex);
}

void DFS_adjlist(node *a[], int V)
{
int i;
for(i = 0; i<V; i++) check[i] = 0;
    for(i = 0; i<V; i++)
        if(check[i] == 0) DFS_recur_list(a, V, i);
}

void init_stack() {
    top = -1;
}

int stack_empty() {
    return (top < 0);
}

int pop(){
    return stack[top--];
}

void push(int i){
    stack[++top] = i;
}

void visit(int i){
    printf("Visit: %d\n",i);
}