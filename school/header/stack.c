#include <stdio.h>
#include <stdlib.h>

typedef struct _node
{
    int key;
    struct _node *next;
}node;

node *head, *tail;

int is_stack_empty();
void print_stack();
void clear();
int pop();
int push(int k);
void init_stack();

// void main(){
//     init_stack();
//     int data[6] = {3,6,9,1,6,3};
//     for(int i =0;i<6;i++){
//         push(data[i]);
//     }
//     print_stack();
//     while(head->next != tail){
//         printf("%d",pop());
//     }
//     int data2[] = {4,8,7,2,0};
//     for(int i =0;i<5;i++){
//         push(data[i]);
//     }
//     printf("\n");
//     init_stack();
//     pop();
// // }
// void main(){
//     int i =0;
//     char s[15];
//     init_stack();
//     scanf("%s",s);

//     while(s[i] != '\0'){
//         if(s[i] >= 'A' && s[i]<='Z'){
//             printf("%c",s[i]);
//         }else if(s[i] == '('){

//         }else if(s[i] == ')'){
//             printf("%c",pop());
//         }else{
//             push(s[i]);
//         }
//         i++;
//     }
// }

int is_stack_empty(){
    return (head->next == tail);
}

void init_stack()
{
    head = (node *)calloc(1, sizeof(node));
    tail = (node *)calloc(1, sizeof(node));
    head->next = tail;
    tail->next = tail;
}

int push(int k)
{
    node *t;
    if((t = (node*)malloc(sizeof(node))) == NULL)
    {
        printf("Out of memory !\n");
        return -1;
    }
    t->key = k;
    t->next = head->next;
    head->next = t;
    return k;
}

int pop()
{
    node *t;
    int k;
    if(head->next == tail)
    {
        printf("Stack underflow !\n");
        return -1;
    }
    t = head->next;
    k = t->key;
    head->next = t->next;
    free(t);
    return k;
}

void clear()
{
    node *t, *s;
    t = head->next;
    while(t != tail)
    {
        s = t;
        t = t->next;
        free(s);
    }
    head->next = tail;
}

void print_stack()
{
    node *t;
    t = head->next;
    while(t != tail) {
    printf("%-6d", t->key);
    t = t->next;
}
}