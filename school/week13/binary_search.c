#include <stdio.h>
#define LEN 12

typedef struct _node{
    int key;
    struct _node *left;
    struct _node *right;
}node;

node *bti_search(int key, node *base, int *num)
{
    node *s;
    s = base->left; // root node
    while(key != s->key && s != NULL)
    {
        if(key < s->key)
            s = s->left;
        else
            s = s->right;
    }
    if(s == NULL) return NULL;
    else return s;
}

node *bti_insert(int key, node *base, int *num)
{
    node *p, *s;
    p = base;
    s = base->left;
    while(s != NULL){
        p = s;
        if(key < s->key) s = s->left;
        else s = s->right;
    }
    s = (node *)malloc(sizeof(node));
    s->key = key;
    s->left=NULL;
    s->right=NULL;
    if(key < p->key || p == base) p->left = s;
    else p->right = s;
    (*num)++;
    return s;
}

node *bti_delete1(int key, node *base, int *num)
{
node *parent, *son, *del, *nexth;
parent = base;
del = base->left;
while(key != del->key && del != NULL){
parent = del;
if(del->key > key) del = del->left;
else del = del->right;
}
if(del == NULL) return NULL;
if(del->left == NULL && del->right == NULL)
son = NULL;
else if(del->left != NULL && del->right != NULL)
{
nexth = del->right;
if(nexth->left != NULL){
while(nexth->left->left != NULL) next = next->left;
son = nexth->left;
nexth->left = son->right;
son->left = del->left;
son->right = del->right;
}
else
{
son = nexth;
son->left = del->right;
}
}
else 
{
if(del->left != NULL) son = del->left;
else son = del->right;
}
if(key < parent->key || parent == base)
parent->left = son;
else
parent->right = son;
free(del);
(*num)--;
return parent;
}

int main(){
    int k,num=0,index=0;
    int dat[LEN] = {'F','B','O','A','D','C','L','G','M','H','K','N'};
    int* sortedData = (int*)calloc(LEN,sizeof(int));
    node* temp;
    init_tree(&head);
    for(k=0;k<LEN;k++){
        temp=bti_insert(data[k],head,&num);
    }
    printf("%c\n",head->left->right->left->left->right->key);
    // bti_delete1
}


