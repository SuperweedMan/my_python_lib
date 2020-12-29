#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <iostream>
#define PY_SSIZE_T_CLEAN
#include <Python.h>


int myputs(char *content, char *filename){
    FILE *fp = fopen(filename, "w");
    fputs(content, fp);
    fclose(fp);
    return 0;
}

int main(){
    int i = myputs("123", "./test");
}