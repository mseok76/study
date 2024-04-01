#include "header/linked_list.c"

void insert_nodes_circul(int k);
void delete_after_circul(node *t);
void josephus(int n, int m);

node *head;

void main(){
    head = (node*)calloc(1,sizeof(node));
    int n,m;
    printf("scanf N,M : ");
    scanf("%d %d",&n,&m);
    josephus(n,m);
    printf("\n");
}

void insert_nodes_circul(int k){
    // make a list from 1 to k
    node *t;
    int i;
    t = (node *)calloc(1, sizeof(node));
    t->key = 1;
    head = t;
    for(i=2; i<=k; i++){
        t->next = (node *)calloc(1, sizeof(node));
        t = t->next;
        t->key = i;
    }
    t->next = head;
}

void delete_after_circul(node *t){
    node *s;
    s= t->next;
    t->next = t->next->next;
    free(s);
    // just the same as a simple LL
}

void josephus(int n, int m){
    node *t;
    int i;
    insert_nodes_circul(n);
    t = head ->next;
    while(t != head){
        printf("%d ",t->key);
        t = t-> next;
    }
    printf("\n");
    t = head;
    printf("\nAnswer : ");
    while( t != t->next){
        for(i = 0; i<m-1; i++){
            t = t->next;
        }
        printf("%d ", t->next->key);
        delete_after_circul(t);
    }
    printf("%d", t->key); // last node
}