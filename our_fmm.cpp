#include "our_fmm.hpp"

float FastMarching_solve(int i1,int j1,int i2,int j2, const CvMat* f, const CvMat* t)
{
    double sol, a11, a22, m12;
    a11=CV_MAT_ELEM(*t,float,i1,j1);
    a22=CV_MAT_ELEM(*t,float,i2,j2);
    m12=MIN(a11,a22);

    if( CV_MAT_ELEM(*f,uchar,i1,j1) != INSIDE )
        if( CV_MAT_ELEM(*f,uchar,i2,j2) != INSIDE )
            if( fabs(a11-a22) >= 1.0 )
                sol = 1+m12;
            else
                sol = (a11+a22+sqrt((double)(2-(a11-a22)*(a11-a22))))*0.5;
        else
            sol = 1+a11;
    else if( CV_MAT_ELEM(*f,uchar,i2,j2) != INSIDE )
        sol = 1+a22;
    else
        sol = 1+m12;

    return (float)sol;
}

static void icvCalcFMM(const CvMat *f, CvMat *t, CvPriorityQueueFloat *Heap) {
    int i, j, ii = 0, jj = 0, q;
    float dist;

    while (Heap->Pop(&ii,&jj)) {

        unsigned known=CHANGE;
        CV_MAT_ELEM(*f,uchar,ii,jj) = (uchar)known;

        for (q=0; q<4; q++) {
            i=0; j=0;
            if     (q==0) {i=ii-1; j=jj;}
            else if(q==1) {i=ii;   j=jj-1;}
            else if(q==2) {i=ii+1; j=jj;}
            else {i=ii;   j=jj+1;}
            if ((i<=0)||(j<=0)||(i>f->rows)||(j>f->cols)) continue;

            if (CV_MAT_ELEM(*f,uchar,i,j)==INSIDE) {
                dist = min4(FastMarching_solve(i-1,j,i,j-1,f,t),
                            FastMarching_solve(i+1,j,i,j-1,f,t),
                            FastMarching_solve(i-1,j,i,j+1,f,t),
                            FastMarching_solve(i+1,j,i,j+1,f,t));
                CV_MAT_ELEM(*t,float,i,j) = dist;
                CV_MAT_ELEM(*f,uchar,i,j) = BAND;
                Heap->Push(i,j,dist);
            }
        }
    }

    for (i=0; i<f->rows; i++) {
        for(j=0; j<f->cols; j++) {
            if (CV_MAT_ELEM(*f,uchar,i,j) == CHANGE) {
               CV_MAT_ELEM(*f,uchar,i,j) = KNOWN;
               CV_MAT_ELEM(*t,float,i,j) = -CV_MAT_ELEM(*t,float,i,j);
            }
        }
    }
}


static void icvTeleaInpaintFMM(const CvMat *f, CvMat *t, CvMat *out, int range, CvPriorityQueueFloat *Heap ) {
    int i = 0, j = 0, ii = 0, jj = 0, k, l, q;
    float dist;

    while (Heap->Pop(&ii,&jj)) {

        CV_MAT_ELEM(*f,uchar,ii,jj) = KNOWN;

        for(q=0; q<4; q++) {


            if     (q==0) {i=ii-1; j=jj;}
            else if(q==1) {i=ii;   j=jj-1;}
            else if(q==2) {i=ii+1; j=jj;}
            else if(q==3) {i=ii;   j=jj+1;}

            //skip if i or j are outside the image boundaries
            if ((i<=1)||(j<=1)||(i>t->rows-1)||(j>t->cols-1)) continue;

            if (CV_MAT_ELEM(*f,uchar,i,j)==INSIDE) {


                dist = min4(FastMarching_solve(i-1,j,i,j-1,f,t),
                           FastMarching_solve(i+1,j,i,j-1,f,t),
                           FastMarching_solve(i-1,j,i,j+1,f,t),
                           FastMarching_solve(i+1,j,i,j+1,f,t));
                CV_MAT_ELEM(*t,float,i,j) = dist;


                CvPoint2D32f gradI,gradT,r;
                float Ia=0,Jx=0,Jy=0,s=1.0e-20f,w,dst,lev,dir,sat;

                if (CV_MAT_ELEM(*f,uchar,i,j+1)!=INSIDE) {
                    if (CV_MAT_ELEM(*f,uchar,i,j-1)!=INSIDE) {
                        gradT.x=(float)((CV_MAT_ELEM(*t,float,i,j+1)-CV_MAT_ELEM(*t,float,i,j-1)))*0.5f;
                    } else {
                        gradT.x=(float)((CV_MAT_ELEM(*t,float,i,j+1)-CV_MAT_ELEM(*t,float,i,j)));
                    }
                } else {
                    if (CV_MAT_ELEM(*f,uchar,i,j-1)!=INSIDE) {
                        gradT.x=(float)((CV_MAT_ELEM(*t,float,i,j)-CV_MAT_ELEM(*t,float,i,j-1)));
                    } else {
                        gradT.x=0;
                    }
                }

                if (CV_MAT_ELEM(*f,uchar,i+1,j)!=INSIDE) {
                    if (CV_MAT_ELEM(*f,uchar,i-1,j)!=INSIDE) {
                        gradT.y=(float)((CV_MAT_ELEM(*t,float,i+1,j)-CV_MAT_ELEM(*t,float,i-1,j)))*0.5f;
                    } else {
                        gradT.y=(float)((CV_MAT_ELEM(*t,float,i+1,j)-CV_MAT_ELEM(*t,float,i,j)));
                    }
                } else {
                    if (CV_MAT_ELEM(*f,uchar,i-1,j)!=INSIDE) {
                        gradT.y=(float)((CV_MAT_ELEM(*t,float,i,j)-CV_MAT_ELEM(*t,float,i-1,j)));
                    } else {
                        gradT.y=0;
                    }
                }

                for (k=i-range; k<=i+range; k++) {
                    int km=k-1+(k==1),kp=k-1-(k==t->rows-2);
                    for (l=j-range; l<=j+range; l++) {
                        int lm=l-1+(l==1),lp=l-1-(l==t->cols-2);
                        if (k>0&&l>0&&k<t->rows-1&&l<t->cols-1) {
                            if ((CV_MAT_ELEM(*f,uchar,k,l)!=INSIDE)&&((l-j)*(l-j)+(k-i)*(k-i)<=range*range)) {
                                r.y     = (float)(i-k);
                                r.x     = (float)(j-l);

                                dst = (float)(1./(VectorLength(r)*sqrt(VectorLength(r))));
                                lev = (float)(1./(1+fabs(CV_MAT_ELEM(*t,float,k,l)-CV_MAT_ELEM(*t,float,i,j))));

                                dir=VectorScalMult(r,gradT);
                                if (fabs(dir)<=0.01) dir=0.000001f;
                                w = (float)fabs(dst*lev*dir);

                                if (CV_MAT_ELEM(*f,uchar,k,l+1)!=INSIDE) {
                                    if (CV_MAT_ELEM(*f,uchar,k,l-1)!=INSIDE) {
                                        gradI.x=(float)((CV_MAT_ELEM(*out,unsigned short int,km,lp+1)-CV_MAT_ELEM(*out,unsigned short int,km,lm-1)))*2.0f;
                                    } else {
                                        gradI.x=(float)((CV_MAT_ELEM(*out,unsigned short int,km,lp+1)-CV_MAT_ELEM(*out,unsigned short int,km,lm)));
                                    }
                                } else {
                                    if (CV_MAT_ELEM(*f,uchar,k,l-1)!=INSIDE) {
                                        gradI.x=(float)((CV_MAT_ELEM(*out,unsigned short int,km,lp)-CV_MAT_ELEM(*out,unsigned short int,km,lm-1)));
                                    } else {
                                        gradI.x=0;
                                    }
                                }
                                if (CV_MAT_ELEM(*f,uchar,k+1,l)!=INSIDE) {
                                    if (CV_MAT_ELEM(*f,uchar,k-1,l)!=INSIDE) {
                                        gradI.y=(float)((CV_MAT_ELEM(*out,unsigned short int,kp+1,lm)-CV_MAT_ELEM(*out,unsigned short int,km-1,lm)))*2.0f;
                                    } else {
                                        gradI.y=(float)((CV_MAT_ELEM(*out,unsigned short int,kp+1,lm)-CV_MAT_ELEM(*out,unsigned short int,km,lm)));
                                    }
                                } else {
                                    if (CV_MAT_ELEM(*f,uchar,k-1,l)!=INSIDE) {
                                        gradI.y=(float)((CV_MAT_ELEM(*out,unsigned short int,kp,lm)-CV_MAT_ELEM(*out,unsigned short int,km-1,lm)));
                                    } else {
                                        gradI.y=0;
                                    }
                                }
                                Ia += (float)w * (float)(CV_MAT_ELEM(*out,unsigned short int,km,lm));
                                Jx -= (float)w * (float)(gradI.x*r.x);
                                Jy -= (float)w * (float)(gradI.y*r.y);
                                s  += w;
                            }
                        }
                    }
                }

                sat = (float)((Ia/s+(Jx+Jy)/(sqrt(Jx*Jx+Jy*Jy)+1.0e-20f)+0.5f));
                int isat = cvRound(sat);
                CV_MAT_ELEM(*out,unsigned short int,i-1,j-1) = CV_CAST_16U(isat);
                CV_MAT_ELEM(*f,uchar,i,j) = BAND;
                Heap->Push(i,j,dist);
            }
        }
    }
}

void teleainpaint(const CvArr* _input_img, const CvArr* _inpaint_mask, CvArr* _output_img, double inpaintRange)
{

    cv::Ptr<CvMat> mask, mask16, band, f, t, out;
    cv::Ptr<CvPriorityQueueFloat> Heap, Out;
    cv::Ptr<IplConvKernel> el_cross, el_range;

    CvMat input_hdr, mask_hdr, output_hdr;
    CvMat* input_img, *inpaint_mask, *output_img;

    int range=cvRound(inpaintRange);
    int erows, ecols;

    input_img = cvGetMat( _input_img, &input_hdr );
    inpaint_mask = cvGetMat( _inpaint_mask, &mask_hdr );
    output_img = cvGetMat( _output_img, &output_hdr );

    ecols = input_img->cols + 2;
    erows = input_img->rows + 2;

    //Makes sure range value is between 1 and 100
    range = MAX(range,1);
    range = MIN(range,100);

    f = cvCreateMat(erows, ecols, CV_8UC1);
    t = cvCreateMat(erows, ecols, CV_32FC1);
    band = cvCreateMat(erows, ecols, CV_8UC1);
    mask = cvCreateMat(erows, ecols, CV_8UC1);
    mask16 = cvCreateMat(erows, ecols, CV_16UC1);
    out = cvCreateMat(erows, ecols, CV_16UC1);

    //3 x 3 star-shaped kernel
    el_cross = cvCreateStructuringElementEx(3,3,1,1,CV_SHAPE_CROSS,NULL);

    cvCopy(input_img, output_img);
    cvSet(mask,cvScalar(KNOWN,0,0,0));
    cvSet(mask16,cvScalar(KNOWN,0,0,0));
    COPY_MASK_BORDER1_C1(inpaint_mask,mask,uchar);
    COPY_MASK_BORDER1_C1(inpaint_mask,mask16,unsigned short int);

    SET_BORDER1_C1(mask,uchar,0);
    SET_BORDER1_C1(mask16,unsigned short int,0);


    cvSet(f,cvScalar(KNOWN,0,0,0));
    cvSet(t,cvScalar(1.0e6f,0,0,0));
    cvDilate(mask,band,el_cross,1);   // image with narrow band

    Heap=new CvPriorityQueueFloat;
    if (!Heap->Init(band))
        return;

    cvSub(band,mask,band,NULL);
    SET_BORDER1_C1(band,uchar,0);
    if (!Heap->Add(band))
        return;

    //for these cvSets to work, both band and mask must be 8-bit unsigned (opencv specification for all masks)
    cvSet(f,cvScalar(BAND,0,0,0),band);
    cvSet(f,cvScalar(INSIDE,0,0,0),mask);
    cvSet(t,cvScalar(0,0,0,0),band);

    el_range = cvCreateStructuringElementEx(2*range+1,2*range+1,range,range,CV_SHAPE_RECT,NULL);

    cvDilate(mask16,out,el_range,1);

    cvSub(out,mask16,out,NULL);

    Out=new CvPriorityQueueFloat;
    if (!Out->Init(out))
        return;
    if (!Out->Add(band))
        return;

    cvSub(out,band,out,NULL);
    SET_BORDER1_C1(out,unsigned short int,0);

    icvCalcFMM(out,t,Out);

    icvTeleaInpaintFMM(mask,t,output_img,range,Heap);
}

void inpaint(InputArray _src, InputArray _mask, OutputArray _dst, double inpaintRange)
{
    Mat src = _src.getMat(), mask=_mask.getMat();
    _dst.create(src.size(), src.type());
    CvMat c_src = src, c_dst = _dst.getMat(), c_mask=mask;

    if( CV_MAT_TYPE(c_src.type) != CV_16UC1 )
        CV_Error( CV_StsUnsupportedFormat, "The source must be 16-bit 1-channel image" );

    if( CV_MAT_TYPE(c_dst.type) != CV_16UC1 )
        CV_Error( CV_StsUnsupportedFormat, "The destination must be 16-bit 1-channel image" );

    teleainpaint(&c_src, &c_mask, &c_dst, inpaintRange);
}
