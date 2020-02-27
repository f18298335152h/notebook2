/*
 * =====================================================================================
 *
 *       Filename:  mat.cpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2018年10月23日 11时10分36秒
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */
#include <stdlib.h>
#include<iostream>
#include <stdio.h>
#include <typeinfo>
#define MALLOC_ALIGN  16
#define SATURATE_CAST_UCHAR(X) (unsigned char)::std::min(::std::max((int)(X), 0), 255);

using namespace std;

int load_param(FILE* fp)
{
  int id = 0;
  while (fscanf(fp, "%d=", &id) == 1)
  {
    std::cout<<"id = "<<id<<std::endl;
      char vstr[16];
      int nscan = fscanf(fp, "%15s", vstr);
      printf("\n");
    printf("vstr : %s/n",vstr);
    //bool is_float = vstr_is_float(vstr);
    printf("\n");

  }
  return 0;
}


static float half2float(unsigned short value)
{
  unsigned short sign = (value & 0x8000) >> 15;
  unsigned short exponent = (value & 0x7c00) >> 10;
  unsigned short significand = value & 0x03FF;

  fprintf(stderr, "sign = %d ,exponent = %d ,significand = %d\n", sign, exponent, significand);
}



static size_t alignSize(size_t sz, int n)
{
  size_t tmp;

  std:cout<<" in alignSize   (zs+n-1) = "<<(sz + n-1)<<std::endl;
  tmp = sz + n-1;
  size_t end = tmp & -n;
  std::cout<<" (sz + n-1) & -n  = "<<end<<std::endl;
  return (sz + n-1) & -n;
}


template<typename _Tp> static inline _Tp* alignPtr(_Tp* ptr, int n=(int)sizeof(_Tp))
{
  std::cout<<"  (size_t)ptr  "<<(size_t)ptr<<std::endl;
  return (_Tp*)(((size_t)ptr + n-1) & -n);
}

static void* fastMalloc(size_t size)
{
  std::cout<< "size + sizeof(void*) + MALLOC_ALIGN  = "<<size + sizeof(void*) + MALLOC_ALIGN<<std::endl;
  unsigned char* udata = (unsigned char*)malloc(size + sizeof(void*) + MALLOC_ALIGN);

  if (udata)
  {
    printf("the addr of udata : %x/n",udata);
    printf("\n");

    printf("\n");
    printf("(unsigned char**)udata  addr: %x/n",(unsigned char**)udata);
    printf("\n");

  } 
  unsigned char* a;
  a[0] = SATURATE_CAST_UCHAR(*udata);
  printf("a[0] addr: %x/n",a[0]);
  printf("\n");
  std::cout<<"(size_t)((unsigned char**)udata + 1)"<<(size_t)((unsigned char**)udata + 1)<<std::endl; 
  unsigned char** adata = alignPtr((unsigned char**)udata + 1, MALLOC_ALIGN);
  printf("adata  addr: %x/n",adata);
}

int main()
{
  //int total = alignSize(14, 4);
  //std::cout<<"total   = "<<total<<std::endl;
  //fastMalloc(total);
  printf("\n");
  //size_t cstep = alignSize(256*227*16,16)/16;


  FILE* fpWrite=fopen("squeezenet_v1.1.param","r"); 
  load_param(fpWrite);


  return 0;
}
