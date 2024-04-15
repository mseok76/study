#include <stdio.h>
#include <stdlib.h>
#define MAX 10
int stack[MAX];
int top;

void main()
{
    int k;
    init_stack();
    int data[6] = {3,6,9,1,6,3};
    for(int i =0;i<6;i++){
        push(data[i]);
    }
    for(int i=top; i>=0; i--){
        printf("%d ",stack[i]);
    }
    for(int i=top; i>=0; i--){
        printf("%d ",pop());
    }
    printf("\n pop_end");
    init_stack();
    pop();  
}

void init_stack(void)
{
top = -1;
}

int push(int t)
{
    if(top >= MAX-1)
    {
        printf("Stack overflow !!!\n");
        return -1;
    }
    stack[++top] = t;
    return t;
}

int pop()
{
    if(top < 0)
    {
        printf("Stack underflow !!!\n");
        return -1;
    }
    return stack[top--];
}

