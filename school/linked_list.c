#include <stdio.h>
#include <stdlib.h>

typedef struct _node{
    int key;
    struct _node *next;
}node;

node *head, *tail;

void init_list()
{
    head = (node *)calloc(1, sizeof(node));
    tail = (node *)calloc(1, sizeof(node));
    head->next = tail;
    tail->next = tail;
} 

node *insert_after(int k, node *t)
{
node *temp;
temp = (node *)calloc(1, sizeof(node));
temp->key = k;
temp->next = t->next;
t->next = temp;
return temp;
}

node *find_node(int k)
{
node *temp;
temp = head->next;
while(temp->key != k && temp != tail)
temp = temp->next;
return temp;
}

int delete_node(int k)
{
    node *s, *p;
    p = head;
    s = p->next;
    while(s->key != k && s != tail) {
        p = p->next;
        s = p->next;
    }
    if( s != tail) {
        p->next = s->next;
        free(s);
        return 1;
    }
    else return 0;
}

node *insert_node(int t, int k)
{
    node *s, *p, *r;
    p = head;
    s = p->next;
    while(s->key != k && s != tail) {
        p = p->next;
        s = p->next;
    }
    if( s != tail) {
        r = (node *)calloc(1, sizeof(node));
        r->key = t;
        p->next = r;
        r->next = s;
    }
    return p->next;
}

node *ordered_insert(int k)
{
    node *s, *p, *r;
    p = head;
    s = p->next;
    while(s->key <= k && s != tail) {
        p = p->next;
        s = p->next;
    }
    r = (node *)calloc(1, sizeof(node));
    r->key = k;
    p->next = r;
    r->next = s;
    return r;
}

node *delete_all()
{
node *s;
node *t;
t = head->next;
while(t != tail)
{
s = t;
t = t->next;
free(s);
}
head->next = tail;
return head;
}
