#include <stdio.h>
#include <stdlib.h>

typedef struct _node{
    int vertex;
    struct _node *next;
}node;

int order;
int son_of_root ;
int* check;

int AP_recur(node *a[], int i);
void AP_search(node *a[], int V);

void main(){
    
}


int AP_recur(node *a[], int i)
{
    node *t;
    int min, m;
    check[i] = min = ++order;
    for(t = a[i]; t != NULL; t=t->next){
        if(i==0 && check[t->vertex] ==0) son_of_root++;
        if(check[t->vertex] == 0){
            m = AP_recur(a, t->vertex);
            if(m < min) min = m;
            if(m >= check[i] && i != 0)
                printf("* %c %2d : %d\n", int2name(i),check[i], m);
            else
                printf("%c %2d : %d\n", int2name(i),check[i], m);
        }
        else{
            if(check[t->vertex] < min)
            min = check[t->vertex];
        }
    return min;
    }
}

void AP_search(node *a[], int V)
{
    int i, n = 0;
    node *t;
    for(i = 0; i<V; i++) check[i] = 0;
        order = son_of_root = 0;
        AP_recur(a, 0);
    if(son_of_root > 1) printf("* ");
    else printf(" ");
    printf("%c son : %d\n", int2name(0), son_of_root);
}
