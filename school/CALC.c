#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "header/stack.h"

void postfix(char *dst, char *src)
{
    char c;
    init_stack();
    while(*src) {
        if(*src == ')')
        {
            *dst++ = pop();
            *dst++ = ' ';
            src++; // what does it mean ?
        }
        else if (*src == '+' || *src == '-' || \
            *src == '*' || *src == '/')
        {
            push(*src);
            src++;
        }
        else if (*src >= '0' && *src <= '9')
        {
            do{
                *dst++ = *src++;
            }while(*src >= '0' && *src <= '9');
            *dst++ = ' ';   
        }
        else
        src++;
    } // end while
    *dst = 0;
}

int main(){
    char exp[256];
    char src[256] = "(1+(2*3))";
    postfix(exp, src);
    printf("Postfix representation : %s\n", exp);
}