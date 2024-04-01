#include "header/graph.h"
#include "header/queue.c"

void count_components(int a[][MAX_NODE], int V);
void BFS_adjlist(node *a[], int V);
void BFS_adjmatrix(int a[][MAX_NODE], int V);
void visit(int i);
void init_stack();
int stack_empty();
int pop();
void push(int i);


int check[MAX_NODE];
int top;
int stack[MAX_NODE];

void main(){
    int V, E;
    scanf("%d %d",&V,&E);
    // for adjacency matrix
    input_adjmatrix(GM, &V, &E);
    print_adjmatrix(GM, V);
    // traverse the given graph
    BFS_adjmatrix(GM, V);
    count_components(GM, V);
}

void BFS_adjmatrix(int a[][MAX_NODE], int V)
{
    int i, j;
    init_queue();
    for(i = 0; i<V; i++) check[i] = 0;
    for(i = 0; i<V; i++){
        if(check[i] == 0){
            put(i);
            check[i] = 1;
            while(!queue_empty()){
                i = get();
                visit(i);
                for( j = 0; j<V; j++){
                    if(a[i][ j] != 0){
                        if(check[j] == 0){
                            put( j);
                            check[j] = 1;
                        }
                    }
                }
            }
        }
    }

}

void BFS_adjlist(node *a[], int V)
{
    int i;
    node *t;
    init_queue();
    for(i = 0; i<V; i++) check[i] = 0;
    for(i = 0; i<V; i++){
        if(check[i] == 0){
            put(i);
            check[i] = 1;
            while(!queue_empty()){
                i = get();
                visit(i);
                    for(t = a[i]; t != NULL; t=t->next){
                        if(check[t->vertex] == 0){
                            put(t->vertex);
                            check[t->vertex] = 1;
                        }
                    }
            }
        }
    }
}

void count_components(int a[][MAX_NODE], int V)
{
    int cnt = 0;
    int i, j;
    init_stack();
    for(i = 0; i<V; i++) check[i] = 0;
    for(i = 0; i<V; i++){
        if(check[i] == 0){
        cnt++;
        push(i);
        check[i] = 1;
        while(!stack_empty()){
            i = pop();
            printf("%c", int2name(i));
            for( j = 0; j<V; j++){
                if(a[i][ j] != 0){
                    if(check[j] == 0){
                        push( j);
                        check[j] = 1;
                    }
                }
            }
        }
        printf("# of CC : %d\n", cnt);
        }
    }
    
}

void visit(int i){
    printf("Visit: %d\n",i);
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

