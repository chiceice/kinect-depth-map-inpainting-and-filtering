
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/core/internal.hpp"
#include <math.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <limits.h>
#include <float.h>

#undef CV_MAT_ELEM_PTR_FAST
#define CV_MAT_ELEM_PTR_FAST( mat, row, col, pix_size )  \
     ((mat).data.ptr + (size_t)(mat).step*(row) + (pix_size)*(col))

inline float
min4( float a, float b, float c, float d )
{
    a = MIN(a,b);
    c = MIN(c,d);
    return MIN(a,c);
}

#define CV_MAT_3COLOR_ELEM(img,type,y,x,c) CV_MAT_ELEM(img,type,y,(x)*3+(c))
#define KNOWN  0  //known outside narrow band
#define BAND   1  //narrow band (known)
#define INSIDE 2  //unknown
#define CHANGE 3  //servise

typedef struct CvHeapElem
{
    float T;
    int i,j;
    struct CvHeapElem* prev;
    struct CvHeapElem* next;
}
CvHeapElem;


class CvPriorityQueueFloat
{
protected:
    CvHeapElem *mem,*empty,*head,*tail;
    int num,in;

public:
    bool Init( const CvMat* f )
    {
        int i,j;
        for( i = num = 0; i < f->rows; i++ )
        {
            for( j = 0; j < f->cols; j++ )
                num += CV_MAT_ELEM(*f,uchar,i,j)!=0;
        }
        if (num<=0) return false;
        mem = (CvHeapElem*)cvAlloc((num+2)*sizeof(CvHeapElem));
        if (mem==NULL) return false;

        head       = mem;
        head->i    = head->j = -1;
        head->prev = NULL;
        head->next = mem+1;
        head->T    = -FLT_MAX;
        empty      = mem+1;
        for (i=1; i<=num; i++) {
            mem[i].prev   = mem+i-1;
            mem[i].next   = mem+i+1;
            mem[i].i      = -1;
            mem[i].T      = FLT_MAX;
        }
        tail       = mem+i;
        tail->i    = tail->j = -1;
        tail->prev = mem+i-1;
        tail->next = NULL;
        tail->T    = FLT_MAX;
        return true;
    }

    bool Add(const CvMat* f) {
        int i,j;
        for (i=0; i<f->rows; i++) {
            for (j=0; j<f->cols; j++) {
                if (CV_MAT_ELEM(*f,uchar,i,j)!=0) {
                    if (!Push(i,j,0)) return false;
                }
            }
        }
        return true;
    }

    bool Push(int i, int j, float T) {
        CvHeapElem *tmp=empty,*add=empty;
        if (empty==tail) return false;
        while (tmp->prev->T>T) tmp = tmp->prev;
        if (tmp!=empty) {
            add->prev->next = add->next;
            add->next->prev = add->prev;
            empty = add->next;
            add->prev = tmp->prev;
            add->next = tmp;
            add->prev->next = add;
            add->next->prev = add;
        } else {
            empty = empty->next;
        }
        add->i = i;
        add->j = j;
        add->T = T;
        in++;
        //      printf("push i %3d  j %3d  T %12.4e  in %4d\n",i,j,T,in);
        return true;
    }

    bool Pop(int *i, int *j) {
        CvHeapElem *tmp=head->next;
        if (empty==tmp) return false;
        *i = tmp->i;
        *j = tmp->j;
        tmp->prev->next = tmp->next;
        tmp->next->prev = tmp->prev;
        tmp->prev = empty->prev;
        tmp->next = empty;
        tmp->prev->next = tmp;
        tmp->next->prev = tmp;
        empty = tmp;
        in--;
        //      printf("pop  i %3d  j %3d  T %12.4e  in %4d\n",tmp->i,tmp->j,tmp->T,in);
        return true;
    }

    bool Pop(int *i, int *j, float *T) {
        CvHeapElem *tmp=head->next;
        if (empty==tmp) return false;
        *i = tmp->i;
        *j = tmp->j;
        *T = tmp->T;
        tmp->prev->next = tmp->next;
        tmp->next->prev = tmp->prev;
        tmp->prev = empty->prev;
        tmp->next = empty;
        tmp->prev->next = tmp;
        tmp->next->prev = tmp;
        empty = tmp;
        in--;
        //      printf("pop  i %3d  j %3d  T %12.4e  in %4d\n",tmp->i,tmp->j,tmp->T,in);
        return true;
    }

    CvPriorityQueueFloat(void) {
        num=in=0;
        mem=empty=head=tail=NULL;
    }

    ~CvPriorityQueueFloat(void)
    {
        cvFree( &mem );
    }
};

inline float VectorScalMult(CvPoint2D32f v1,CvPoint2D32f v2) {
   return v1.x*v2.x+v1.y*v2.y;
}

inline float VectorLength(CvPoint2D32f v1) {
   return v1.x*v1.x+v1.y*v1.y;
}

#define SET_BORDER1_C1(image,type,value) {\
      int i,j;\
      for(j=0; j<image->cols; j++) {\
         CV_MAT_ELEM(*image,type,0,j) = value;\
      }\
      for (i=1; i<image->rows-1; i++) {\
         CV_MAT_ELEM(*image,type,i,0) = CV_MAT_ELEM(*image,type,i,image->cols-1) = value;\
      }\
      for(j=0; j<image->cols; j++) {\
         CV_MAT_ELEM(*image,type,erows-1,j) = value;\
      }\
   }

#define COPY_MASK_BORDER1_C1(src,dst,type) {\
      int i,j;\
      for (i=0; i<src->rows; i++) {\
         for(j=0; j<src->cols; j++) {\
            if (CV_MAT_ELEM(*src,type,i,j)!=0)\
               CV_MAT_ELEM(*dst,type,i+1,j+1) = INSIDE;\
         }\
      }\
   }

using namespace cv;

void inpaint(InputArray _src, InputArray _mask, OutputArray _dst, double inpaintRange);
