#include<stdio.h>
//directed list implementation & import
//node is 'head'
#include "week11/floyd.c"

//need array
int earliest[MAX_NODE];
int latest[MAX_NODE];
node GL[MAX_NODE];
head GLH[MAX_NODE];


void forward_state(head net[], int V)
{
    int i, j, k;
    node *ptr;
    init_stack();
    set_count_indegree(net, V);
    for(i = 0; i<V; i++) earliest[i] = 0;
    for(i = 0; i<V; i++)
        if(!net[i].count) push(i);
    for(i = 0; i<V; i++){
        if(!stack_empty()){
            j = pop();
            for(ptr=net[j].next; ptr; ptr=ptr->next){
                k = ptr->vertex;
                net[k].count--;
                if(!net[k].count)
                    push(k);
                if(earliest[k] < earliest[ j]+ptr->weight)
                    earliest[k]=earliest[ j]+ptr->weight;
            } // for
        } // if
    } // for
}


void backward_state(head net[], int V)
{
    int i, j, k, l;
    node *ptr;
    init_stack();
    set_count_outdegree(net, V);
    for(i = 0; i<V; i++)
        latest[i] = earliest[V-1];
    for(i = 0; i<V; i++)
        if(!net[i].count) push(i);
    for(i = 0; i<V; i++){
        if(!stack_empty()){
            j = pop();
            for(l = 0; l<V; l++){
                for(ptr=net[l].next; ptr; ptr=ptr->next){
                    if(ptr->vertex == j){
                        k = l;
                        net[k].count--;
                        if(!net[k].count)
                            push(k);
                        if(latest[k] > latest[ j]–ptr->weight)
                            latest[k] = latest[ j]-ptr->weight;
                    }
                }
            }
        }
    }
}

//void print_table


void main() {
    int k, V, E;
    FILE *fp = fopen("graph_AOE.txt", "r");
    input_adjlist(GL, &V, &E);
    print_adjlist(GL, V);
    for (k = 0; k < V; k++)
        network[k].next = GL[k];
    forward_stage(network, V);
    backward_stage(network, V);
    print_critical_activity(network, V);
    fclose(fp);
}
