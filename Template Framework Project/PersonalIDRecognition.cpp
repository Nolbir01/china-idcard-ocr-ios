//
//  PersonalIDRecognition.cpp
//  Template Framework Project
//
//  Created by wl on 12/3/14.
//  Copyright (c) 2014 Daniele Galiotto - www.g8production.com. All rights reserved.
//

#include "PersonalIDRecognition.h"

// 以下定义宏为从有效身份证区域截取 姓名&地址 和 号码 子区域 用

#define NAMEROI_WIDTH 0.48
#define NAMEROI_HEIGH 0.75
#define NAMEROI_XPOS  0.175
#define NAMEROI_YPOS  0.05

#define ID_WIDTH 0.61
#define ID_HEIGH 0.15
#define ID_XPOS  0.36
#define ID_YPOS  0.8

// 以下定义宏为从 姓名&地址 和 号码 子区域 进行二值分割用

#define NAME_THRE 15
#define ID_THRE   30

//用来记录自定义一些函数的返回值
enum WB_RETURN
{
    NORMAL = 0,
    PARAM_ERROR,
    EXCEPTION_ERROR,
};

//该结构体用来记录像素信息
//可以用CvPoint代替
struct Wb_Point
{
    int nX;
    int nY;
    Wb_Point(int nx=0,int ny=0)
    {
        nX=nx;
        nY=ny;
    };
};

//该结构体用来保存定位出来的字符的位置（一个Rect对应一个字符）
//该结构体在函数DetermineAllCharactersPosition中使用
//可以用CvRect代替
struct Wb_Rect
{
    int left;
    int right;
    int top;
    int bottom;
    Wb_Rect()
    {
        left=0;
        right=0;
        top=0;
        bottom=0;
    }
};

//该结构体用来存放分割出来的文字字符（姓名和地址）
struct CharacterVector
{
    IplImage *ImageVector[100];// 允许最大字符数目是100
    int number;   // 字符数目,based on 1
    CvRect pRect[100];
    float nAverageWidth; // 字符平均高度
    float nAverageInternal; // 字符平均间隔
    bool bedged[100];  // 表示当前字符是否位于右边界，如是，则需要扩展
    
    int LineInfo[10];   //行号信息，总共不超过10信息，比如：LineInfo={3,5,6,0,0,0,0,0,0,0,0,0,0,0} 表示第一行文字有3个，第二行有5个，第三行有6个， 后面各行字符数均为0个
    CharacterVector()
    {
        for(int i=0;i<100;i++)
        {
            ImageVector[i]=0;
            pRect[i].x=0;
            pRect[i].y=0;
            pRect[i].height=0;
            pRect[i].width=0;
            nAverageWidth=0;
            nAverageInternal=0;
            bedged[i]=false;
            
        }
        
        for (int i=0;i<10;i++)
        {
            LineInfo[i]=0;
        }
        
        number=0;
    }
    ~CharacterVector()
    {
        for(int i=0;i<100;i++)
        {
            if (ImageVector[i])
            {
                cvReleaseImage(&ImageVector[i]);
                ImageVector[i]=0;
            }
            
        }
    }
    
};


struct SAccumulatorElem
{
    int totalpoints;
    int* xpos;
    int* ypos;
    int* pos;
    double EllipseParams[5];
};

WB_RETURN WriteImage(IplImage *image,const char *image1);

//根据胡哥要求新加函数，该函数将根据输入的二值图像（pBinaryImage）返回一个将汉字分割好的vector里面(第二个参数)
//由于数字与汉字的分割很不同，为保证准确率，目前该函数只针对汉字部分有效，这点与胡哥的要求可能不同
//2012-7-14
void fenge(IplImage *pBinaryImage,CharacterVector &oCharacterVector,int nMaxHeight=100, int nMinHeight=5,int nMaxWidth=100, int nMinWidth=5,int nMinInterval=2);

//使用Tesseract检测姓名与地址
//add 2012-7-10
bool fengeNameAndAddress(IplImage *& pOriginalImage,CharacterVector* &pVector);

IplImage * fengeID(IplImage *pOriginalImage);

// 将fengge产生的Rect字符坐标转换为相对于有效身份证区域的坐标
void convertpositions(int xoffset, int yoffset,int nRemoveTop, int nRemoveBottom,CharacterVector &oCharacterVector);

/***************************************************************/
//Author		:		WangBin
//Date			:		2012-04-14
//Params		:	    nLeft and nTop is the start clone coordinate on source image
//
//Description	:	    This function will clone a ROI of source image
//                      pdst represents ROI size
//                      nLeft and nTop represents ROI position
//Return		:
//Revise History:
/***************************************************************/
WB_RETURN CloneImage(IplImage *psrc,IplImage *pdst,int nLeft=0, int nTop=0);

/***************************************************************/
//Author		:		WangBin
//Date			:		2012-04-16
//Params		:	    x: x coordinate; y: y coordinate
//
//Description	:		get a particular channel of a image pixel
//                      channel based on 0, so for gray image , only zero is valid;
//                      for RGB24 image valid channel is 0(Blue),1(Green),2(Red)
//Return		:
//Revise History:
/***************************************************************/
WB_RETURN GetImagePixel(IplImage * TargetImage,int x,int y,float &pixelvalue,int channel=0);

/***************************************************************/
//Author		:		WangBin
//Date			:		2012-04-16
//Params		:	    x: x coordinate; y: y coordinate
//
//Description	:		set a particular channel of a image pixel
//                      channel based on 0, so for gray image , only zero is valid;
//                      for RGB24 image valid channel is 0(Blue),1(Green),2(Red)
//Return		:
//Revise History:
/***************************************************************/
WB_RETURN SetImagePixel(IplImage * TargetImage,int x,int y,float pixelvalue=0,int channel=0);

/***************************************************************/
//Author		:		WangBin
//Date			:		2012-06-12
//Params		:	    pInputImage		: input image;
//						pOutputImage	: output image.

//Description	:		This is used for any channel image
//Return		:		WB_RETURN
//Revise History:
/***************************************************************/
WB_RETURN RunEqualizeHistogram(IplImage* pInputImage,IplImage*pOutputImage);

/***************************************************************/
//Author		:		WangBin
//Date			:		2012-06-21
//Params		:
//
//Description	:		Calculate every character's position
//Return		:       character position list
//Revise History:
/***************************************************************/

Wb_Rect* DetermineAllCharactersPosition(IplImage* m_pBinaryImage);

/***************************************************************/
//Author		:		WangBin
//Date			:		2012-06-21
//Params		:
//
//Description	:		charge is a char's ASIC is nunber 0~9
//Return		:
//Revise History:
/***************************************************************/
bool IsNumber(char nChar);

/***************************************************************/
//Author		:		WangBin
//Date			:		2012-06-21
//Params		:	    m_pProcessedImage :A part of original image
//						m_pBinaryImage    :character is black in this image
//						paths             :11 template image store here
//Description	:		this function used for detecting numbers
//Return		:       recognition result
//Revise History:
/***************************************************************/
char* DetectPersonalNumber(IplImage* m_pProcessedImage,IplImage *&m_pBinaryImage,char** paths);

/***************************************************************/
//Author		:		WangBin
//Date			:		2012-04-22
//Params		:	    outImage: A stacked image (up is img1, down is img2)
//                      outImage = cvCreateImage( cvSize( MAX(img1->width, img2->width),img1->height + img2->height ),IPL_DEPTH_8U, img1->channel)
//                      imag1 and imag2 same  channel , image size may not equal
//Description	:		This is used for stack two images
//                      if input image is not equal, the idle pixel in out image will be padding 0
//Return		:
//Revise History:
/***************************************************************/
void StackTwoImages(IplImage *image1, IplImage *image2,IplImage *outImage);

//检测背景的函数， 返回的图像中背景均为设为255.
void SubBackGround(IplImage *pBinaryImage);

//通过三点的坐标信息计算夹角余弦.
double angle(CvPoint* pt1,CvPoint* pt2,CvPoint* pt0);

//把四边形的四个顶点排序，使之顺时针排列
void  sortPoints( CvPoint* srcPoints) ;

/*
 删除图像中较小的面积
 */
IplImage* deletelittlearea(IplImage* src,double minarea);




/*----FindSquare------识别图像中的矩形---------------------
 
 src:     待识别的图像
 返回:    四边形的端点序列
 
 ---------------------------------------------------------*/
IplImage*  FindSquare(IplImage* src,int scale);


//检测身份证号码
void fengeID2(IplImage * pOriginalImage,IplImage* &outImage);



//调用Tesseract识别
void RecognitionFromTesseract(char * pPath);

IplImage* deletelittlearea(IplImage* src,double minarea);

//add 2012-7-12
//该函数是从初始设定的数字ROI中，提取出精确的数字区域上边界
int ExtractNumbers(IplImage *pNumberROI);

int  ExtractNumbers(IplImage *pNumberROI)
{
    
    // 从ROI图像最底部进行投影，从而精确确定数字ROI区域
    int nRealTop=0;// 用来保存图像数字的真是高度
    bool bLine=false;
    float nRowSum;
    int nIter=0;   //表示发现的线行数
    for (int j=pNumberROI->height;j<0;j--)
    {
        nRowSum=0;
        // 取图像上的点进行X方向的投影（投影结果进行了归一化处理）
        for (int i=0;i<pNumberROI->width;i++)
        {
            float fPixelVal=0;
            GetImagePixel(pNumberROI,i,j,fPixelVal);
            nRowSum+=(fPixelVal/255);
            
        }
        //投影结束
        if (nRowSum<pNumberROI->width-5&&bLine==false)// 找到了一行文字，默认为一行文字至少要有6个前景点
        {
            nIter++;
            bLine=true;
        }
        if (nRowSum>pNumberROI->width-5&&bLine) //找到了空白行，将表示是否找到文字的flag设为0，，默认为一行文字至少要有6个前景点
        {
            bLine=false;
            if (nIter==1) //已经找到数字的上边缘，记录此时行坐标,RealTop 在真实上标界基础上增加10 pix
            {
                nRealTop=fmax(j-10,0);
            }
            
            
        }
    }
    
    return nRealTop;
}




//此函数为自适应canny算法，源自网络:http://my.oschina.net/liujinofhome/blog/37041
//使用自适应canny算法，是为了避免去对不同的图像进行阈值的设定，也可以检测到更多的边缘信息.
//使用时只需要将cvCannyAda()中lowthreshold与highthreshold设为-1
void AdaptiveFindThreshold(CvMat *dx, CvMat *dy, double &low, double &high)
{
    // 仿照matlab，自适应求高低两个门限
    CvSize size;
    IplImage *imge=0;
    int i,j;
    CvHistogram *hist;
    int hist_size = 255;
    float range_0[]={0,256};
    float* ranges[] = { range_0 };
    double  PercentOfPixelsNotEdges = 0.7;
    size = cvGetSize(dx);
    imge = cvCreateImage(size, IPL_DEPTH_32F, 1);
    // 计算边缘的强度, 并存于图像中
    float maxv = 0;
    for(i = 0; i < size.height; i++ )
    {
        const short* _dx = (short*)(dx->data.ptr + dx->step*i);
        const short* _dy = (short*)(dy->data.ptr + dy->step*i);
        float* _image = (float *)(imge->imageData + imge->widthStep*i);
        for(j = 0; j < size.width; j++)
        {
            _image[j] = (float)(abs(_dx[j]) + abs(_dy[j]));
            maxv = maxv < _image[j] ? _image[j]: maxv;
        }
    }
    
    // 计算直方图
    range_0[1] = maxv;
    hist_size = (int)(hist_size > maxv ? maxv:hist_size);
    hist = cvCreateHist(1, &hist_size, CV_HIST_ARRAY, ranges, 1);
    cvCalcHist( &imge, hist, 0, NULL );
    int total = (int)(size.height * size.width * PercentOfPixelsNotEdges);
    float sum=0;
    int icount = hist->mat.dim[0].size;
    
    float *h = (float*)cvPtr1D( hist->bins, 0 );
    for(i = 0; i < icount; i++)
    {
        sum += h[i];
        if( sum > total )
            break;
    }
    // 计算高低门限
    high = (i+1) * maxv / hist_size ;
    low = high * 0.4;
    cvReleaseImage( &imge );
    cvReleaseHist(&hist);
}


void cvCannyAda( const void* srcarr, void* dstarr, double lowhresh, double highhresh,int aperture_size )
{
    cv::Ptr<CvMat> dx, dy;
    cv::AutoBuffer<char> buffer;
    std::vector<uchar*> stack;
    uchar **stackop = 0, **stack_bottom = 0;
    
    CvMat srcstub, *src = cvGetMat( srcarr, &srcstub );
    CvMat dststub, *dst = cvGetMat( dstarr, &dststub );
    CvSize size;
    int flags = aperture_size;
    int low, high;
    int* mag_buf[3];
    uchar* map;
    int mapstep, maxsize;
    int i, j;
    CvMat mag_row;
    
    if( CV_MAT_TYPE( src->type ) != CV_8UC1 ||
       CV_MAT_TYPE( dst->type ) != CV_8UC1 )
        CV_Error( CV_StsUnsupportedFormat, "" );
    
    if( !CV_ARE_SIZES_EQ( src, dst ))
        CV_Error( CV_StsUnmatchedSizes, "" );
    
    if( lowhresh > highhresh )
    {
        double t;
        CV_SWAP( lowhresh, highhresh, t );
    }
    
    aperture_size &= INT_MAX;
    if( (aperture_size & 1) == 0 || aperture_size < 3 || aperture_size > 7 )
        CV_Error( CV_StsBadFlag, "" );
    
    size = cvGetSize( src );
    
    
    dx = cvCreateMat( size.height, size.width, CV_16SC1 );
    dy = cvCreateMat( size.height, size.width, CV_16SC1 );
    cvSobel( src, dx, 1, 0, aperture_size );
    cvSobel( src, dy, 0, 1, aperture_size );
    if(lowhresh == -1 && highhresh == -1)
    {
        AdaptiveFindThreshold(dx, dy, lowhresh, highhresh);
    }
    /**if( icvCannyGetSize_p && icvCanny_16s8u_C1R_p && !(flags & CV_CANNY_L2_GRADIENT) )
     {
     int buf_size=  0;
     IPPI_CALL( icvCannyGetSize_p( size, &buf_size ));
     CV_CALL( buffer = cvAlloc( buf_size ));
     IPPI_CALL( icvCanny_16s8u_C1R_p( (short*)dx->data.ptr, dx->step,
     (short*)dy->data.ptr, dy->step,
     dst->data.ptr, dst->step,
     size, (float)lowhresh,
     (float)highhresh, buffer ));
     EXIT;
     }*/
    
    if( flags & CV_CANNY_L2_GRADIENT )
    {
        Cv32suf ul, uh;
        ul.f = (float)lowhresh;
        uh.f = (float)highhresh;
        
        low = ul.i;
        high = uh.i;
    }
    else
    {
        low = cvFloor( lowhresh );
        high = cvFloor( highhresh );
    }
    
    buffer.allocate( (size.width+2)*(size.height+2) + (size.width+2)*3*sizeof(int) );
    
    mag_buf[0] = (int*)(char*)buffer;
    mag_buf[1] = mag_buf[0] + size.width + 2;
    mag_buf[2] = mag_buf[1] + size.width + 2;
    map = (uchar*)(mag_buf[2] + size.width + 2);
    mapstep = size.width + 2;
    
    maxsize = MAX( 1 << 10, size.width*size.height/10 );
    stack.resize( maxsize );
    stackop = stack_bottom = &stack[0];
    
    memset( mag_buf[0], 0, (size.width+2)*sizeof(int) );
    memset( map, 1, mapstep );
    memset( map + mapstep*(size.height + 1), 1, mapstep );
    
    /** sector numbers
     (Top-Left Origin)
     
     1   2   3
     *  *  *
     * * *
     0*******0
     * * *
     *  *  *
     3   2   1
     */
    
#define CANNY_PUSH(d)    *(d) = (uchar)2, *stackop++ = (d)
#define CANNY_POP(d)     (d) = *--stackop
    
    mag_row = cvMat( 1, size.width, CV_32F );
    
    // calculate magnitude and angle of gradient, perform non-maxima supression.
    // fill the map with one of the following values:
    //   0 - the pixel might belong to an edge
    //   1 - the pixel can not belong to an edge
    //   2 - the pixel does belong to an edge
    for( i = 0; i <= size.height; i++ )
    {
        int* _mag = mag_buf[(i > 0) + 1] + 1;
        float* _magf = (float*)_mag;
        const short* _dx = (short*)(dx->data.ptr + dx->step*i);
        const short* _dy = (short*)(dy->data.ptr + dy->step*i);
        uchar* _map;
        int x, y;
        int magstep1, magstep2;
        int prev_flag = 0;
        
        if( i < size.height )
        {
            _mag[-1] = _mag[size.width] = 0;
            
            if( !(flags & CV_CANNY_L2_GRADIENT) )
                for( j = 0; j < size.width; j++ )
                    _mag[j] = abs(_dx[j]) + abs(_dy[j]);
            /**else if( icvFilterSobelVert_8u16s_C1R_p != 0 ) // check for IPP
             {
             // use vectorized sqrt
             mag_row.data.fl = _magf;
             for( j = 0; j < size.width; j++ )
             {
             x = _dx[j]; y = _dy[j];
             _magf[j] = (float)((double)x*x + (double)y*y);
             }
             cvPow( &mag_row, &mag_row, 0.5 );
             }*/
            else
            {
                for( j = 0; j < size.width; j++ )
                {
                    x = _dx[j]; y = _dy[j];
                    _magf[j] = (float)std::sqrt((double)x*x + (double)y*y);
                }
            }
        }
        else
            memset( _mag-1, 0, (size.width + 2)*sizeof(int) );
        
        // at the very beginning we do not have a complete ring
        // buffer of 3 magnitude rows for non-maxima suppression
        if( i == 0 )
            continue;
        
        _map = map + mapstep*i + 1;
        _map[-1] = _map[size.width] = 1;
        
        _mag = mag_buf[1] + 1; // take the central row
        _dx = (short*)(dx->data.ptr + dx->step*(i-1));
        _dy = (short*)(dy->data.ptr + dy->step*(i-1));
        
        magstep1 = (int)(mag_buf[2] - mag_buf[1]);
        magstep2 = (int)(mag_buf[0] - mag_buf[1]);
        
        if( (stackop - stack_bottom) + size.width > maxsize )
        {
            int sz = (int)(stackop - stack_bottom);
            maxsize = MAX( maxsize * 3/2, maxsize + 8 );
            stack.resize(maxsize);
            stack_bottom = &stack[0];
            stackop = stack_bottom + sz;
        }
        
        for( j = 0; j < size.width; j++ )
        {
#define CANNY_SHIFT 15
#define TG22  (int)(0.4142135623730950488016887242097*(1<<CANNY_SHIFT) + 0.5)
            
            x = _dx[j];
            y = _dy[j];
            int s = x ^ y;
            int m = _mag[j];
            
            x = abs(x);
            y = abs(y);
            if( m > low )
            {
                int tg22x = x * TG22;
                int tg67x = tg22x + ((x + x) << CANNY_SHIFT);
                
                y <<= CANNY_SHIFT;
                
                if( y < tg22x )
                {
                    if( m > _mag[j-1] && m >= _mag[j+1] )
                    {
                        if( m > high && !prev_flag && _map[j-mapstep] != 2 )
                        {
                            CANNY_PUSH( _map + j );
                            prev_flag = 1;
                        }
                        else
                            _map[j] = (uchar)0;
                        continue;
                    }
                }
                else if( y > tg67x )
                {
                    if( m > _mag[j+magstep2] && m >= _mag[j+magstep1] )
                    {
                        if( m > high && !prev_flag && _map[j-mapstep] != 2 )
                        {
                            CANNY_PUSH( _map + j );
                            prev_flag = 1;
                        }
                        else
                            _map[j] = (uchar)0;
                        continue;
                    }
                }
                else
                {
                    s = s < 0 ? -1 : 1;
                    if( m > _mag[j+magstep2-s] && m > _mag[j+magstep1+s] )
                    {
                        if( m > high && !prev_flag && _map[j-mapstep] != 2 )
                        {
                            CANNY_PUSH( _map + j );
                            prev_flag = 1;
                        }
                        else
                            _map[j] = (uchar)0;
                        continue;
                    }
                }
            }
            prev_flag = 0;
            _map[j] = (uchar)1;
        }
        
        // scroll the ring buffer
        _mag = mag_buf[0];
        mag_buf[0] = mag_buf[1];
        mag_buf[1] = mag_buf[2];
        mag_buf[2] = _mag;
    }
    
    // now track the edges (hysteresis thresholding)
    while( stackop > stack_bottom )
    {
        uchar* m;
        if( (stackop - stack_bottom) + 8 > maxsize )
        {
            int sz = (int)(stackop - stack_bottom);
            maxsize = MAX( maxsize * 3/2, maxsize + 8 );
            stack.resize(maxsize);
            stack_bottom = &stack[0];
            stackop = stack_bottom + sz;
        }
        
        CANNY_POP(m);
        
        if( !m[-1] )
            CANNY_PUSH( m - 1 );
        if( !m[1] )
            CANNY_PUSH( m + 1 );
        if( !m[-mapstep-1] )
            CANNY_PUSH( m - mapstep - 1 );
        if( !m[-mapstep] )
            CANNY_PUSH( m - mapstep );
        if( !m[-mapstep+1] )
            CANNY_PUSH( m - mapstep + 1 );
        if( !m[mapstep-1] )
            CANNY_PUSH( m + mapstep - 1 );
        if( !m[mapstep] )
            CANNY_PUSH( m + mapstep );
        if( !m[mapstep+1] )
            CANNY_PUSH( m + mapstep + 1 );
    }
    
    // the final pass, form the final image
    for( i = 0; i < size.height; i++ )
    {
        const uchar* _map = map + mapstep*(i+1) + 1;
        uchar* _dst = dst->data.ptr + dst->step*i;
        
        for( j = 0; j < size.width; j++ )
            _dst[j] = (uchar)-(_map[j] >> 1);
    }
}

bool fengeNameAndAddress(IplImage * &pOriginalImage,CharacterVector* &pVector, const char *imagePath)
{
    
    IplImage *pTopImage=0;
    IplImage *pBottomImage=0;
    IplImage *pBinaryImage=0;
    // 从原始图像上分割文字区域，并储存在pNameROI
    IplImage* pNameROI=cvCreateImage(cvSize(pOriginalImage->width*NAMEROI_WIDTH,pOriginalImage->height*NAMEROI_HEIGH),pOriginalImage->depth,pOriginalImage->nChannels);
    //IplImage* pBinaryImage=NULL;
    CloneImage(pOriginalImage,pNameROI,NAMEROI_XPOS*pOriginalImage->width,NAMEROI_YPOS*pOriginalImage->height);
    
    //均衡化图像（可以使灰度图也可以是彩图）,初步消除光照影响
    RunEqualizeHistogram(pNameROI,pNameROI);
    cvReleaseImage(&pBinaryImage);
    pBinaryImage=cvCreateImage(cvGetSize(pNameROI),pNameROI->depth,1);
    
    //进一步进行阈值分割进一步消除光照影响（大于15， 设为255；小于15，设为 0）
    for (int i=0;i<pNameROI->width;i++)
    {
        for (int j=0;j<pNameROI->height;j++)
        {
            if (pNameROI->nChannels==1)  // for gray or RGB8 image
            {
                float fGray=0.0;
                GetImagePixel(pNameROI,i,j,fGray,0);
                if (fGray<NAME_THRE)
                {
                    SetImagePixel(pBinaryImage,i,j,0);
                }
                else
                {
                    SetImagePixel(pBinaryImage,i,j,255);
                }
                
            }
            else if (pNameROI->nChannels>=3) // for RGB24 or RGB32 image
            {
                float fPixR=0.0;
                float fPixG=0.0;
                float fPixB=0.0;
                GetImagePixel(pNameROI,i,j,fPixR,0);
                GetImagePixel(pNameROI,i,j,fPixG,1);
                GetImagePixel(pNameROI,i,j,fPixB,2);
                if (fPixR<NAME_THRE&&fPixG<NAME_THRE&&fPixB<NAME_THRE&&abs(fPixR-fPixG)+abs(fPixR-fPixB)+abs(fPixG-fPixB)<=NAME_THRE)
                {
                    SetImagePixel(pBinaryImage,i,j,0);
                }
                else
                {
                    SetImagePixel(pBinaryImage,i,j,255);
                }
            }
        }
    }
    // 阈值分割完毕
    
    
    
    // 下面的代码是消除文字中的性别行和生日行以及检测图像是否下上翻转
    int nRemoveTop=0,nRemoveBottom=0;
    bool bLine=false;
    float* pRowSum=new float[pBinaryImage->height];
    for(int i=0;i<pBinaryImage->height;i++)
    {
        pRowSum[i]=0;
    }
    float nRowSum;
    int nIter=0;
    for (int j=0;j<pBinaryImage->height;j++)
    {
        for (int i=0;i<pBinaryImage->width;i++)
        {
            float fPixelVal=0;
            GetImagePixel(pBinaryImage,i,j,fPixelVal);
            pRowSum[j]+=(fPixelVal/255);
        }
        
    }
    //投影结束
    //局部去噪，保证能够精确获得每个字符位置
    //cvDilate(pBinaryImage,pBinaryImage);
    
    for (int j=0;j<pBinaryImage->height;j++)
    {
        nRowSum=pRowSum[j];
        
        if (nRowSum<pBinaryImage->width-20)// 找到了一行文字
        {
            int iiter=j;
            
            
            // every character's height must bigger than 10
            while (iiter<j+10&&iiter<pBinaryImage->height)
            {
                if (pRowSum[iiter]<pBinaryImage->width-20)
                {
                    iiter++;
                }
                else
                {
                    
                    break;
                }
            }
            if (iiter==j+10)//此时满足字符高度最小为 10 pix的条件
            {
                
                if (bLine==false)
                {
                    nIter++;
                    bLine=true;
                    if (nIter==2)  // 找到了第二行文字，应该是性别行
                    {
                        nRemoveTop=j-5;
                    }
                    if (nIter==4)   //找到了第四行文字，应该是第一行地址
                    {
                        nRemoveBottom=j-5;
                    }
                }
            }
            
            
        }
        if (nRowSum>pBinaryImage->width-5) //找到了空白行，将表示是否找到文字的flag设为0
        {
            if(bLine)
            {
                bLine=false;
            }
        }
        
    }
    if (nIter<4)  //检测出来的线行数大于3 （姓名一行，性别一行，地址至少一行）
    {
        
        cvReleaseImage(&pNameROI);
        cvReleaseImage(&pBinaryImage);
        return false;
    }
    
    //合并子图像（姓名+地址）
    pTopImage=cvCreateImage(cvSize(pBinaryImage->width,nRemoveTop),pBinaryImage->depth,pBinaryImage->nChannels);
    CloneImage(pBinaryImage,pTopImage);
    pBottomImage=cvCreateImage(cvSize(pBinaryImage->width,(pBinaryImage->height-nRemoveBottom-1)),pBinaryImage->depth,pBinaryImage->nChannels);
    CloneImage(pBinaryImage,pBottomImage,0,nRemoveBottom);
    cvReleaseImage(&pBinaryImage);
    pBinaryImage=cvCreateImage(cvSize(pTopImage->width,(pTopImage->height+pBottomImage->height)),pTopImage->depth,pTopImage->nChannels);
    StackTwoImages(pTopImage,pBottomImage,pBinaryImage);
    //合并完成
    
    //在子图像中检测字符
    pVector=new CharacterVector();
    
    fenge(pBinaryImage,*pVector);//,40,15,40,15
    
    //将子图像中的字符坐标转换为身份证区域坐标
    convertpositions(NAMEROI_XPOS*pOriginalImage->width, NAMEROI_YPOS*pOriginalImage->height,nRemoveTop,nRemoveBottom,*pVector);
    
    //以下四行代码为测试检测出来的坐标用的,去掉注释后，检测出来的字符位置会在源图像上标记出来，并保存在test.jpg里面
    //之后在每个小区域进行阈值分割并获得最终结果（保存在 C盘根目录）
    for (int i=0;i<pVector->number;i++)
    {
        IplImage *pImage=0;
        IplImage *pImageGray=0;
        IplImage *pProcessImage=0;
        if (pVector->bedged[i])
        {
            pImage=cvCreateImage(cvSize(pVector->ImageVector[i]->width+15,pVector->ImageVector[i]->height),pVector->ImageVector[i]->depth,pOriginalImage->nChannels);
            pImageGray=cvCreateImage(cvGetSize(pImage),pVector->ImageVector[i]->depth,1);
            pProcessImage=cvCreateImage(cvGetSize(pImage),pVector->ImageVector[i]->depth,1);
        }
        else
        {
            pImage=cvCreateImage(cvGetSize(pVector->ImageVector[i]),pVector->ImageVector[i]->depth,pOriginalImage->nChannels);
            pImageGray=cvCreateImage(cvGetSize(pVector->ImageVector[i]),pVector->ImageVector[i]->depth,1);
            pProcessImage=cvCreateImage(cvGetSize(pVector->ImageVector[i]),pVector->ImageVector[i]->depth,1);
            
        }
        
        char *path=new char [200];
        
        CloneImage(pOriginalImage,pImage,pVector->pRect[i].x,pVector->pRect[i].y);
        cvCvtColor(pImage,pImageGray,CV_BGR2GRAY);
        
        
        
        
        cvThreshold(pImageGray,pProcessImage,100,255,CV_THRESH_OTSU);
        
        //cvSaveImage(path,pProcessImage);
        
        delete[]path;
        cvReleaseImage(&pImageGray);
        cvReleaseImage(&pProcessImage);
        cvReleaseImage(&pImage);
        
        cvRectangle(pOriginalImage,cvPoint(pVector->pRect[i].x,pVector->pRect[i].y),cvPoint((pVector->pRect[i].x+pVector->pRect[i].width),(pVector->pRect[i].y+pVector->pRect[i].height)),cvScalar(255,255,255));
    }
    
    //自动化测试
    /*
     char *path1=new char[100];
     char *path2=new char[100];
     sprintf_s(path1,100,"C://%dtest.jpg",nRecognition);
     sprintf_s(path2,100,"C://%docr.jpg",nRecognition);
     cvSaveImage(path1,pOriginalImage);
     cvSaveImage(path2,pBinaryImage);
     delete []path1;
     delete []path2;
     nRecognition++;
     */
    // 检测完毕
    
    // 所有图像均保存在pVector里面，通过循环将所有图像在硬盘上进行保存
    //char *path=new char[200];
    //for (int i=0;i<pVector->number;i++)
    //{
    //      sprintf_s(path,200,"result//%d.jpg",i);
    //   cvSaveImage(path,pVector->ImageVector[i]);
    //}
    
    
    char* imageFilePath = new char[500];
    sprintf(imageFilePath,"%soriginal.jpg",imagePath);
    cvSaveImage(imageFilePath,pOriginalImage);
    delete[] imageFilePath;
    
    imageFilePath = new char[500];
    sprintf(imageFilePath,"%sbin.jpg",imagePath);
    cvSaveImage(imageFilePath,pBinaryImage);
    delete[] imageFilePath;
    
    //system("OCR\\tesseract.exe result//name.jpg result//1 -1");
    //system("OCR\\tesseract.exe result//address.jpg result//2 -1");
    //	system("OCR\\tesseract.exe result//ocr.jpg result\\wb -l chi");//batch.nochop makebox
    //	system("OCR\\tesseract.exe result//ocr.jpg result//3-2 -lcnlp");
    
    cvReleaseImage(&pNameROI);
    cvReleaseImage(&pBinaryImage);
    cvReleaseImage(&pTopImage);
    cvReleaseImage(&pBottomImage);
    
    
    delete (pRowSum);
    //delete (path);
    return true;
}


WB_RETURN CloneImage(IplImage *psrc,IplImage *pdst,int nLeft, int nTop)
{
    
    if (!psrc||!pdst)
    {
        return PARAM_ERROR;
    }
    if ((pdst->height+nTop)>psrc->height||(pdst->width+nLeft)>psrc->width||pdst->depth>psrc->depth||pdst->nChannels>psrc->nChannels||pdst->nSize>psrc->nSize)
    {
        return PARAM_ERROR;
    }
    if (nLeft<0||nLeft>=psrc->width)
    {
        return PARAM_ERROR;
    }
    if (nTop<0||nTop>=psrc->height)
    {
        return PARAM_ERROR;
    }
    if (psrc->nChannels!=pdst->nChannels||psrc->depth!=pdst->depth)
    {
        return PARAM_ERROR;
    }
    
    IplImage * ptempImage=cvCreateImage(cvGetSize(psrc),psrc->depth,psrc->nChannels);
    
    int i, j, k;
    int height = psrc->height;
    int width = psrc->width;
    int channels = psrc->nChannels;
    int step = psrc->widthStep;
    uchar *psrcimageData = (uchar *)psrc->imageData;
    uchar *pdstimageData = (uchar *)ptempImage->imageData;
    for(i = 0; i < height; i++)
    {
        for(j = 0; j < width; j++)
        {
            for(k = 0; k < channels; k++)
            {
                pdstimageData[i*step + j*channels + k]=psrcimageData[i*step + j*channels + k];
            }
        }
    }
    
    cvZero( pdst );
    cvSetImageROI( ptempImage, cvRect(nLeft,nTop,pdst->width,pdst->height));
    cvAdd( ptempImage, pdst, pdst, NULL );
    cvReleaseImage(&ptempImage);
    
    
    return NORMAL;
}


WB_RETURN WriteImage(IplImage *image, const char *imageName)
{
    //return NORMAL;
    if(!cvSaveImage(imageName, image))
    {
        return EXCEPTION_ERROR;
    }
    return NORMAL;
}

IplImage* LoadImage(const char *imagePath)
{
    IplImage*img=cvLoadImage(imagePath,CV_LOAD_IMAGE_COLOR);
    return img;
}

WB_RETURN GetImagePixel(IplImage * TargetImage,int x,int y,float& pixelvalue,int channel)
{
    if (channel<0 || channel>=TargetImage->nChannels ||y<0 || y>=TargetImage->height ||x<0 || x>=TargetImage->width)
    {
        return PARAM_ERROR;
    }
    pixelvalue=((TargetImage->imageData + TargetImage->widthStep*y))[x*TargetImage->nChannels+channel];
    
    if (pixelvalue<0)
    {
        pixelvalue+=256;
    }
    
    return NORMAL;
}


WB_RETURN SetImagePixel(IplImage * TargetImage,int x,int y,float pixelvalue,int channel)
{
    
    if (channel<0 || channel>=TargetImage->nChannels ||y<0 || y>=TargetImage->height ||x<0 || x>=TargetImage->width)
    {
        return PARAM_ERROR;
    }
    
    ((TargetImage->imageData + TargetImage->widthStep*y))[x*TargetImage->nChannels+channel]=pixelvalue;
    return NORMAL;
}


WB_RETURN RunEqualizeHistogram(IplImage* pInputImage,IplImage*pOutputImage)
{
    if (!pInputImage||!pOutputImage)
    {
        return PARAM_ERROR;
    }
    int i=0;
    IplImage *pImageChannel[4] = { 0, 0, 0, 0 };
    for( i = 0; i < pInputImage->nChannels; i++ )
    {
        pImageChannel[i] = cvCreateImage( cvGetSize(pInputImage), pInputImage->depth, 1 );
    }
    
    // separate each channel
    cvSplit( pInputImage, pImageChannel[0], pImageChannel[1], pImageChannel[2], pImageChannel[3] );
    
    for( i = 0; i < pInputImage->nChannels; i++ )
    {
        // histogram equalization
        cvEqualizeHist( pImageChannel[i], pImageChannel[i] );
    }
    // integer each channel
    cvMerge( pImageChannel[0], pImageChannel[1], pImageChannel[2], pImageChannel[3], pOutputImage);
    
    for( i = 0; i < pInputImage->nChannels; i++ )
    {
        if ( pImageChannel[i] )
        {
            cvReleaseImage( &pImageChannel[i] );
            pImageChannel[i] = 0;
        }
    }
    
    return NORMAL;
    
}


bool IsNumber(char nChar)
{
    //当ASIC码值位于该范围时为0~9数字
    if (nChar>=48&&nChar<=57)
    {
        return true;
    }
    return false;
}

void StackTwoImages(IplImage *image1, IplImage *image2,IplImage *outImage)
{
    if (!image1||!image2||!outImage)
    {
        return ;
    }
    if (outImage->width!=MAX(image1->width,image2->width)||outImage->height!=(image1->height+image2->height))
    {
        return ;
    }
    if (image1->nChannels!=image2->nChannels)
    {
        return ;
    }
    
    cvZero( outImage );
    cvSetImageROI( outImage, cvRect( 0, 0, image1->width, image1->height ) );
    cvAdd( image1, outImage, outImage, NULL );
    cvSetImageROI( outImage, cvRect(0, image1->height, image2->width, image2->height) );
    cvAdd( image2, outImage, outImage, NULL );
    cvResetImageROI( outImage );
    
    return ;
}


IplImage* deletelittlearea(IplImage* src,double minarea)
{
    double tmparea=0.0;
    CvSeq* contour=NULL;
    CvMemStorage* storage=cvCreateMemStorage(0);
    
    IplImage* img_Clone=cvCloneImage(src);
    //访问而至图像每个点的值
    uchar *pp;
    
    IplImage* img_dst=cvCreateImage(cvGetSize(src),IPL_DEPTH_8U,1);
    
    //搜索二值图中的轮廓，并从轮廓树中删除面积小于某个阈值minarea的轮廓
    
    CvScalar color=cvScalar(0,0,0);
    CvContourScanner scanner=NULL;
    scanner=cvStartFindContours(src,storage,sizeof(CvContour),CV_RETR_CCOMP,CV_CHAIN_APPROX_NONE,cvPoint(0,0));
    //开始遍历轮廓树
    CvRect rect;
    while (contour=cvFindNextContour(scanner))
    {
        tmparea=fabs(cvContourArea(contour));
        rect=cvBoundingRect(contour,0);
        if (tmparea<minarea)
        {
            //当连通区域的中心点为白色时，而且面积较小则用黑色进行填充
            pp=(uchar*)(img_Clone->imageData+img_Clone->widthStep*(rect.y+rect.height/2)+rect.x+rect.width/2);
            if (pp[0]==255)
            {
                for (int y=rect.y;y<rect.y+rect.height;y++)
                {
                    for (int x=rect.x;x<rect.x+rect.width;x++)
                    {
                        pp=(uchar*)(img_Clone->imageData+img_Clone->widthStep*y+x);
                        if(pp[0]==255)
                        {
                            pp[0]=0;
                        }
                    }
                }
            }
        }
    }
    
    return img_Clone;
}



/*----DrawGraphics------画出所识别出的图像中形状----------
 
 graphics:    所识别出的图像中的形状
 img:         要画的图像
 num:         形状的边数
 返回:        无
 
 ---------------------------------------------------------*/
void DrawGraphics(CvSeq* graphics,IplImage* img,int num)
{
    CvSeqReader reader;
    CvPoint *pt=new CvPoint[num];
    // fixed
    CvPoint2D32f* src_mat = new CvPoint2D32f[num];
    cvStartReadSeq(graphics,&reader,0);
    
    //read the sequence elements at a time(all vertices of a graphic)
    for(int i=0;i<graphics->total ;i+=num)
    {
        CvPoint* area=pt;
        int count=num;
        
        for(int j=0;j<num;j++)
            //read vertices
        {
            CV_READ_SEQ_ELEM(pt[j],reader);
            // fixed by luo juan save the four points of the square
            src_mat[j].x = (float) pt[j].x;
            src_mat[j].y = (float) pt[j].y;
            // cvCircle(img,pt[j],j*5+5,CV_RGB(255,0,0),2,8);
        }
        //draw the graphic according to the vertices
        //maskimg = cvCreateImage(cvGetSize(img),IPL_DEPTH_8U,3);
        cvPolyLine(img,&area,&count,1,1,CV_RGB(0,255,0),1,CV_AA,0);
        
        //cvSetZero(maskimg);
        //mask多边形区域填充为1以为图像相乘做准备
        //cvFillPoly(maskimg,&area,&count,1,CV_RGB(1,1,1));
    }
    
    //show the resultant image
    //cvNamedWindow("Result",1);
    //cvShowImage("Result",img);
    //cvSaveImage("result//border.jpg",img);
    //cvNamedWindow("Mask",1);
    //cvShowImage("Mask",maskimg);
    //cvSaveImage("result//Mask.jpg",maskimg);
}



/*----FindSquare------识别图像中的矩形---------------------
 
 src:     待识别的图像
 返回:    四边形的端点序列
 
 ---------------------------------------------------------*/
IplImage*  FindSquare(IplImage* src,int scale)
{
    long op,ed;
    
    CvSeq* contours;
    int i, c, l, N = 16;// n is the threadshold number for changing image to binary image
    
    op=clock();
    CvSize sz=cvSize((int)(src->width/scale),(int)(src->height/scale));
    IplImage* timg = cvCreateImage( sz,IPL_DEPTH_8U,src->nChannels);
    cvResize(src,timg,CV_INTER_NN);
    cvSmooth(timg,timg,CV_GAUSSIAN,7,0,0,0);
    
    //IplImage* timg = cvCloneImage(pyr2);
    //创建灰度图像，单通道，大小为原图像大小(近似)
    IplImage* gray = cvCreateImage(cvGetSize(timg),IPL_DEPTH_8U,1);
    
    double smallest_angle = 1;
    //存储所识别出的图像中的矩形
    CvMemStorage* storage = cvCreateMemStorage(0);
    CvSeq* squares = cvCreateSeq( 0, sizeof(CvSeq), sizeof(CvPoint), storage );
    
    // select the maximum ROI in the image  with the width and height divisible by 2,选定图像的感兴趣区域
    // cvSetImageROI( timg, cvRect( 0, 0, sz2.width, sz2.height ));
    IplImage* tgray = cvCreateImage( cvGetSize(timg),IPL_DEPTH_8U, 1 );
    
    
    for( c = 0; c < timg->nChannels; c++ )
    {
        cvSetImageCOI( timg, c+1 );
        cvCopy( timg, tgray, 0 );
        // try several threshold levels to binarize the image,then find contours;
        for( l = 0; l < N; l++ )
        {
            if( l == 0 )
            {
                cvCannyAda(tgray,gray,-1,-1,3);//自适应的canny检测
            }
            else
            {
                // apply threshold if l!=0: tgray(x,y) = gray(x,y) < (l+1)*255/N ? 255 : 0
                cvThreshold( tgray, gray, (l+1)*255/N, 255, CV_THRESH_BINARY );
            }
            //deletelittlearea(gray,50);
            cvDilate( gray, gray, 0, 1 );
            //cvNot(gray,gray);
            //cvNamedWindow("4444",0);
            //cvShowImage("4444",gray);
            //cvWaitKey(0);
            
            cvFindContours( gray, storage, &contours, sizeof(CvContour),
                           CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE,cvPoint(0,0));
            CvSeq* result = 0;
            while( contours )  // test each contour
            {
                // approximate contour with accuracy proportional to the contour perimeter
                result = cvApproxPoly( contours, sizeof(CvContour), storage,
                                      CV_POLY_APPROX_DP, cvContourPerimeter(contours)*0.02, 0 );
                // square contours should have 4 vertices after approximation relatively large area (to filter out noisy contours)
                // and be convex.Note: absolute value of an area is used because
                // area may be positive or negative - in accordance with the contour orientation
                CvRect rect= cvBoundingRect(result,0);
                //double ar=fabs(cvContourArea(result,CV_WHOLE_SEQ));
                //double arect = gray->width*gray->height;
                bool flag=(rect.height>0.4*gray->height) && (rect.width>0.4*gray->width);
                flag=flag && (rect.height<0.95*gray->height) && (rect.width<0.95*gray->width);
                //if( result->total == 4 && (ar > 0.3*arect) &&   (ar < 0.9*arect) && cvCheckContourConvexity(result) )
                if( result->total == 4 && flag && cvCheckContourConvexity(result) )
                {
                    //printf("%lf\t%lf\t%lf\t%lf\n",ar,0.3*arect,0.9*arect,arect);
                    double s = 0, t;
                    for( i = 0; i < 5; i++ )
                    {
                        // find minimum angle between joint  edges (maximum of cosine)
                        if( i >= 2 )
                        {
                            t = fabs(angle(
                                           (CvPoint*)cvGetSeqElem( result, i ),
                                           (CvPoint*)cvGetSeqElem( result, i-2 ),
                                           (CvPoint*)cvGetSeqElem( result, i-1 )));
                            s = s > t ? s : t;// cos value, the larger ,the angle smaller
                        }
                    }
                    //记录cosine值最小的一组多边形,只存储一组多边形结果
                    if(smallest_angle >= s)
                    {
                        smallest_angle = s;
                        cvClearSeq(squares);
                        squares = cvCreateSeq( 0, sizeof(CvSeq), sizeof(CvPoint), storage );
                        for(i = 0; i < 4; i++)
                            cvSeqPush( squares,(CvPoint*)cvGetSeqElem(result, i));
                    }
                }
                contours = contours->h_next;// take the next contour
            }
        }
    }
    cvResetImageROI(src);
    cvResetImageROI(timg);
    if(squares->total <4)
        return NULL;
    
    ed=clock();
    printf("squares:%ldms\n",ed-op);
    op=clock();
    
    //cvNamedWindow("11",0);
    //cvShowImage("11",timg);
    
    //IplImage* cpy;
    //if (timg->nChannels == 1)
    //{
    //	cpy=cvCreateImage(cvGetSize(timg),timg->depth,3);
    //	cvCvtColor(timg,cpy,CV_GRAY2RGB);//得到灰度图
    //}
    //else
    //	cpy = cvCloneImage(timg);
    //DrawGraphics(squares, cpy,4);
    
    //cvNamedWindow("12",0);
    //cvShowImage("12",cpy);
    //cvWaitKey(0);
    
    int num=4;
    CvSeqReader reader;
    CvPoint *validAreaPoints=new CvPoint[num];
    cvStartReadSeq(squares,&reader,0);
    //CvPoint *innerRectPoints = new CvPoint[num];
    for(int j=0;j<num;j++)
    {
        CV_READ_SEQ_ELEM(validAreaPoints[j],reader);
        validAreaPoints[j].x=validAreaPoints[j].x*scale;
        validAreaPoints[j].y=validAreaPoints[j].y*scale;
    }
    sortPoints( validAreaPoints);
    
    validAreaPoints[0].x = validAreaPoints[0].x*1.05;
    validAreaPoints[0].y = validAreaPoints[0].y*1.05;
    validAreaPoints[1].x = validAreaPoints[1].x*1.05;
    validAreaPoints[1].y = validAreaPoints[1].y*0.98;;
    validAreaPoints[2].x = validAreaPoints[2].x*0.95;
    validAreaPoints[2].y = validAreaPoints[2].y*0.98;
    validAreaPoints[3].x = validAreaPoints[3].x*0.95;
    validAreaPoints[3].y = validAreaPoints[3].y*1.05;
    
    IplImage* mask = cvCreateImage(cvGetSize(src),IPL_DEPTH_8U,1);
    cvSetZero(mask);//mask多边形区域填充为以为图像相乘做准备
    cvFillPoly(mask,&validAreaPoints,&num,1,CV_RGB(1,1,1));
    IplImage* maskimg = cvCreateImage( cvGetSize(src), src->depth, src->nChannels );
    cvSetZero(maskimg);
    cvCopy(src, maskimg, mask); //ResetImageROI must be done on timg;
    
    //cvNamedWindow("110",0);
    //cvShowImage("110",maskimg);
    //cvWaitKey(0);
    
    CvPoint2D32f srcTri[4],dstTri[4];
    for (int i = 0; i < 4; i++)
    {
        dstTri[i].x = validAreaPoints[i].x;
        dstTri[i].y = validAreaPoints[i].y;
    }
    
    CvSize rsz;
    rsz.width=856*2;
    rsz.height=540*2;
    srcTri[0].x = 0;
    srcTri[0].y = 0;
    srcTri[1].x = 0;  //缩小一个像素
    srcTri[1].y = rsz.height;
    srcTri[2].x = rsz.width;  //bot right
    srcTri[2].y = rsz.height;
    srcTri[3].x = rsz.width;
    srcTri[3].y = 0;
    ed=clock();
    printf("4555:%ldms\n",ed-op);
    op=clock();
    IplImage *resultImage = cvCreateImage( rsz, src->depth, src->nChannels );
    //IplImage *resultImage = cvCreateImage( cvGetSize(src), src->depth, src->nChannels );
    cvSetZero(resultImage);
    CvMat* warp_mat = cvCreateMat( 3, 3, CV_32FC1 );
    resultImage ->origin = maskimg ->origin;
    cvGetPerspectiveTransform(  dstTri,srcTri, warp_mat );  //由三对点计算仿射变换
    cvWarpPerspective( maskimg,resultImage,warp_mat,CV_INTER_LINEAR,cvScalarAll(0) );  //对图像做仿射变换
    cvReleaseMat( &warp_mat );
    ed=clock();
    printf("cvGetPerspectiveTransform:%ldms\n",ed-op);
    
    delete []validAreaPoints;
    
    cvReleaseImage( &gray );
    cvReleaseImage( &tgray );
    cvReleaseImage( &timg );
    cvReleaseImage( &maskimg);
    cvReleaseImage( &mask);
    cvReleaseMemStorage(&storage);
    
    
    //cvNamedWindow("2",0);
    //cvShowImage("2",maskimg);
    //cvNamedWindow("3",0);
    //cvShowImage("3",resultImage);
    //cvWaitKey(0);
    //cvSaveImage("d://r.jpg",resultImage);
    return resultImage;
}


//sort the vertexes of rectangle to make sure they are arranged by  clock order
void  sortPoints( CvPoint* srcPoints)
{
    double  minv=100000000.0;
    double maxv = 0.0;
    int nPoints=4;
    CvPoint *dstPoints=new CvPoint[nPoints];
    int idx[4];
    for (int i = 0;i< nPoints; i++)
    {
        double ro=srcPoints[i].x*srcPoints[i].x+srcPoints[i].y*srcPoints[i].y;
        if (ro < minv)
        {
            minv = ro;
            idx[0]=i;
        }
        if (ro > maxv)
        {
            maxv = ro;
            idx[2]=i;
        }
    }
    int maxy = 0;
    int miny=100000;
    
    for (int i = 0;i< 4; i++)
    {
        if (i == idx[0] ||i == idx[2])
            continue;
        if (srcPoints[i].y >=maxy)
        {
            maxy = srcPoints[i].y;
            idx[1]=i;
        }
    }
    for (int i = 0;i< 4; i++)
    {
        if (i == idx[0] ||i == idx[2])
            continue;
        if (srcPoints[i].y <miny)
        {
            idx[3]=i;
            miny = srcPoints[i].y;
        }
    }
    
    dstPoints[0].x = srcPoints[idx[0]].x;
    dstPoints[0].y = srcPoints[idx[0]].y;
    dstPoints[1].x = srcPoints[idx[1]].x;
    dstPoints[1].y = srcPoints[idx[1]].y;
    dstPoints[2].x = srcPoints[idx[2]].x;
    dstPoints[2].y = srcPoints[idx[2]].y;
    dstPoints[3].x = srcPoints[idx[3]].x;
    dstPoints[3].y = srcPoints[idx[3]].y;
    
    for (int i=0; i < 4; i++)
    {
        srcPoints[i].x =dstPoints[i].x;
        srcPoints[i].y =dstPoints[i].y;
    }
    delete []dstPoints;
}


//通过三点的坐标信息计算夹角余弦.
double angle(CvPoint* pt1,CvPoint* pt2,CvPoint* pt0)
{
    double dx1 = pt1->x - pt0->x;
    double dy1 = pt1->y - pt0->y;
    double dx2 = pt2->x - pt0->x;
    double dy2 = pt2->y - pt0->y;
    return (dx1*dx2 + dy1*dy2)/sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}

void fenge(IplImage *pBinaryImage,CharacterVector &oCharacterVector,int nMaxHeight, int nMinHeight,int nMaxWidth, int nMinWidth,int nMinInterval)
{
    //用来储存字符的位置信息，每一个字符对应一个rect
    Wb_Rect *m_oCharacterList=new Wb_Rect[100]; // max character is 100
    
    if (!pBinaryImage)
    {
        delete [] m_oCharacterList;
        return ;
    }
    bool bFindRect=false;
    int nXBegin=0;
    int nXEnd=0;
    int nYBegin=0;
    int nYEnd=0;
    
    // 用来储存X,Y方向的投影并初始化这些变量
    CvPoint *oXInfo= new CvPoint [pBinaryImage->height*pBinaryImage->width];
    CvPoint *oYInfo=new CvPoint [pBinaryImage->height*pBinaryImage->width];
    int nXInfoTag=0;
    int nYInfoTag=0;
    for (int i=0;i<(pBinaryImage->height*pBinaryImage->width);i++)
    {
        oXInfo[i].x=0;
        oYInfo[i].x=0;
        oXInfo[i].y=0;
        oYInfo[i].y=0;
    }
    float *pRowSum=new float [pBinaryImage->height];
    float *pColSum=new float [pBinaryImage->width];
    for (int i=0;i<pBinaryImage->height;i++)
    {
        pRowSum[i]=0;
    }
    for (int i=0;i<pBinaryImage->width;i++)
    {
        pColSum[i]=0;
    }
    
    
    int nCharacterTag=0;
    
    
    
#pragma region search y acharacters
    //首先进行X方向的投影
    
    nYInfoTag=0;
    // 进行X方向投影（并归一化） 中间代码不用关心，是根据投影结果确认文字的行数
    for (int j=0;j<pBinaryImage->height;j++)
    {
        for (int i=0;i<pBinaryImage->width;i++)
        {
            float fValue=0.0;
            GetImagePixel(pBinaryImage,i,j,fValue);
            fValue=fValue/255.0;
            pRowSum[j]+=fValue;
        }
    }
    
    bFindRect=false;
    for (int i=0;i<pBinaryImage->height;i++)
    {
        //找到第一行不为不全为白点的行号（即为某一列文字的开始）
        if (pRowSum[i]<pBinaryImage->width&&bFindRect==false)
        {
            bFindRect=true;
            nYBegin=i;
            int niter=i;
            
            // every character's height must bigger than nMinheight
            while (niter<i+nMinHeight&&niter<pBinaryImage->height)
            {
                if (pRowSum[niter]<pBinaryImage->width)
                {
                    niter++;
                }
                else
                {
                    bFindRect=false;
                    nYBegin=-1;
                    break;
                }
            }
            i=niter-1;
        }
        else if ((pRowSum[i]>pBinaryImage->width-1&&bFindRect==true)||((i==pBinaryImage->height-1&&bFindRect==true)))  // end of a character position
        {
            
            int niter=i;
            // 假设上下字符之间必须满足大于nMinInterval pix的间隔
            while (niter<i+nMinInterval&&niter<pBinaryImage->height)
            {
                if (pRowSum[niter]>pBinaryImage->width-1)
                {
                    niter++;
                }
                else
                {
                    bFindRect=true;
                    break;
                }
            }
            
            if (niter==i+nMinInterval)
            {
                nYEnd=i;
                // every character's height must  smaller than nMaxheight
                if (nYEnd-nYBegin>nMaxHeight)
                {
                    oYInfo[nYInfoTag].x=-1;
                    oYInfo[nYInfoTag].y=-1;
                }
                else
                {
                    oYInfo[nYInfoTag].x=nYBegin-1;
                    oYInfo[nYInfoTag].y=nYEnd+1;
                }
                nYInfoTag++;
                bFindRect=false;
            }
        }
    }
    
#pragma endregion
    
    
#pragma region search x acharacters
    
    //进行Y方向投影，同理与X方向的投影
    nCharacterTag=0; // clear all character in list
    for (int k1=0;k1<=nYInfoTag;k1++)
    {
        
        nXBegin=0;
        nXEnd=0;
        for (int i=0;i<pBinaryImage->width;i++)
        {
            pColSum[i]=0;
        }
        nXInfoTag=0;
        CvPoint oPoint;
        for (int i=0;i<pBinaryImage->width;i++)
        {
            oPoint=oYInfo[k1];
            for (int j=oPoint.x;j<oPoint.y;j++)
            {
                float fValue=0.0;
                GetImagePixel(pBinaryImage,i,j,fValue);
                fValue=fValue/255.0;
                pColSum[i]+=fValue;
            }
            
        }
        
        bFindRect=false;
        
        for (int i=0;i<pBinaryImage->width;i++)
        {
            
            if (pColSum[i]<(oPoint.y-oPoint.x)&&bFindRect==false) // find a character's begin position
            {
                
                bFindRect=true;
                
                nXBegin=i;
                
                int niter=i;
                while (niter<i+nMinWidth&&niter<pBinaryImage->width)  // every character's width must bigger than nMinWidth
                {
                    if (pColSum[niter]<(oPoint.y-oPoint.x))
                    {
                        niter++;
                    }
                    else
                    {
                        bFindRect=false;
                        
                        nXBegin=-1;
                        break;
                    }
                }
                i=niter-1;
                
                
            }
            else if ((pColSum[i]>(oPoint.y-oPoint.x-1)&&bFindRect==true))  // end of a character position
            {
                
                int niter=i;
                // 假设左右字符之间必须满足大于nMinInterval pix的间隔
                while (niter<i+nMinInterval&&niter<pBinaryImage->width)
                {
                    if (pColSum[niter]>(oPoint.y-oPoint.x-1))
                    {
                        niter++;
                    }
                    else
                    {
                        bFindRect=true;
                        break;
                    }
                }
                
                if (niter==fmin(i+nMinInterval,pBinaryImage->width))
                {
                    
                    nXEnd=i;
                    // every character's height must  smaller than nMaxheight
                    if (nXEnd-nXBegin>nMaxWidth)
                    {
                        oXInfo[nXInfoTag].x=-1;
                        oXInfo[nXInfoTag].y=-1;
                    }
                    else
                    {
                        
                        oXInfo[nXInfoTag].x=nXBegin-1;
                        oXInfo[nXInfoTag].y=fmin(nXEnd+1,pBinaryImage->width-1);
                        
                    }
                    
                    nXInfoTag++;
                    
                    bFindRect=false;
                    
                    
                }
            }
            else if ((i==pBinaryImage->width-1&&bFindRect==true))
            {
                //当右侧边界遇到最后一个字符时，要进行特殊处理
                nXEnd=i;
                oXInfo[nXInfoTag].x=nXBegin-1;
                oXInfo[nXInfoTag].y=nXEnd;
                
                nXInfoTag++;
                bFindRect=false;
                
                
            }
        }
        Wb_Rect oRect;
        for (int k2=0;k2<=nXInfoTag;k2++)  // insert every rect range of character in m_oCharacterList
        {
            
            oRect.left=oXInfo[k2].x;
            oRect.right=oXInfo[k2].y;
            oRect.top=oYInfo[k1].x;
            oRect.bottom=oYInfo[k1].y;
            
            if (oRect.left!=-1&&oRect.right!=-1&&oRect.top!=-1&&oRect.bottom!=-1)
            {
                if (nCharacterTag>0)
                {
                    if (oRect.top==m_oCharacterList[nCharacterTag-1].top&&oRect.bottom==m_oCharacterList[nCharacterTag-1].bottom)
                    {   //从第二个Rect开始对于同行字符，判断是否在前一个Rect中包含，如果包含或者距离太远（>80pix）则认为非法
                        if (oRect.left>m_oCharacterList[nCharacterTag-1].right-10&&oRect.left<m_oCharacterList[nCharacterTag-1].right+80)
                        {
                            m_oCharacterList[nCharacterTag]=oRect;
                            nCharacterTag++;
                        }
                        
                    }
                    else
                    {
                        m_oCharacterList[nCharacterTag]=oRect;
                        nCharacterTag++;
                    }
                    
                }
                else
                {
                    m_oCharacterList[nCharacterTag]=oRect;
                    nCharacterTag++;
                }
                
                
            }
        }
    }	//Y方向检测完毕
#pragma endregion
    
    
#pragma region create valid rect
    //根据XY方向的投影结果，确定每一个汉字的Rect
    
    int nInternal=0; // 记录可用来查找同行字符间隔的计数器
    int nCharacters=0;//用来记录同行的字符数
    int nLineIndex=0; //用来记录当前的操作行号
    
    // Add By Wangbin 2012-8-7：判断分割结果是否合理（避免出现“儿”，“孔”，“北”，等左右结构汉字被错误分割的问题）
    Wb_Rect oMaxRect;
    double nAverageInternal=0;
    int nIntervalNumber=0;
    double nAverageHeight=0;
    double nAverageWidth=0;
    // 首先获得整个Rect组里面最大的Rect以及平均的字符间隔
    for (int i=0;i<=nCharacterTag;i++)
    {
        Wb_Rect oRect=m_oCharacterList[i];
        if ((oRect.right-oRect.left)>(oMaxRect.right-oMaxRect.left))
        {
            oMaxRect=oRect;
        }
        if (i>0)
        {
            if (m_oCharacterList[i].left-m_oCharacterList[i-1].right>0)
            {
                nAverageInternal+=m_oCharacterList[i].left-m_oCharacterList[i-1].right;
                nIntervalNumber++;
            }
        }
    }
    nAverageInternal/=nIntervalNumber;
    
    //通过Max Rect设置 低/高 区分阈值
    float fLowThr=0.7*(oMaxRect.right-oMaxRect.left);
    float fHigThr=1.1*(oMaxRect.right-oMaxRect.left);
    
    for (int k3=0;k3<=nCharacterTag;k3++)
    {
        Wb_Rect oRect=m_oCharacterList[k3];
        int validnum=0;
        // calculate the valid pixel number in every rect
        for (int i=oRect.left;i<oRect.right;i++)
        {
            for (int j=oRect.top;j<oRect.bottom;j++)
            {
                float fValue=0.0;
                GetImagePixel(pBinaryImage,i,j,fValue);
                if (fValue==0)
                {
                    validnum++;
                }
            }
        }
        
        //以下三个变量分别表示一个汉字的潜在的左右结构
        Wb_Rect oRect1,oRect2;
        if (k3<nCharacterTag-1)
        {
            oRect1=m_oCharacterList[k3];
            oRect2=m_oCharacterList[k3+1];
        }
        // if valid number do not satisfy a character, we remove the rect
        if (validnum<=(0.05*(oRect.right-oRect.left)*(oRect.bottom-oRect.top))||(oRect.right-oRect.left)<nMinWidth||(oRect.bottom-oRect.top)<nMinHeight)
        {
            
            m_oCharacterList[k3].left=0;
            m_oCharacterList[k3].right=0;
            m_oCharacterList[k3].top=0;
            m_oCharacterList[k3].bottom=0;
            
        }
        else if ((oRect1.right-oRect1.left)<30&&(oRect1.bottom-oRect1.top)<50)
        {
            continue;
        }
        else if ((oRect1.right-oRect1.left)<50&&(oRect1.bottom-oRect1.top)<10)
        {
            continue;
        }
        else if ((oRect1.right-oRect1.left)<=0)
        {
            continue;
        }
        else // 获得一个满足要求的字符信息，将该字符存在vector里面， 并对字符的number 加1,同时当出现汉字分错时，进行区域合并
        {
            
            IplImage *character=0;
            if (k3<nCharacterTag-1&&oRect1.top==oRect2.top&&oRect1.bottom==oRect2.bottom&&(oRect1.right-oRect1.left)<fLowThr&&(oRect1.right-oRect1.left+oRect2.right-oRect2.left)<fHigThr&&(oRect2.left-oRect1.right)<nAverageInternal&&(oRect2.right-oRect2.left)>0&&(oRect2.left>=oRect1.right))
            {           //判断标准： 位于同一行，任意一个Rect小于LowThr，之和小于HighThr，且这两个Rect间距小于AverageInternal
                //此时将合并两个区域
                
                oRect.right=oRect2.right;
                oRect.left=oRect1.left;
                if ((oRect2.right-oRect2.left)<=0)
                {
                    continue;
                }
                if ((oRect.right-oRect.left)<=0)
                {
                    continue;
                }
                character=cvCreateImage(cvSize((oRect.right-oRect.left+1),(oRect.bottom-oRect.top+1)),pBinaryImage->depth,pBinaryImage->nChannels);
                CloneImage(pBinaryImage,character,m_oCharacterList[k3].left,m_oCharacterList[k3].top);
                k3++;
                
                
            }
            else
            {
                character=cvCreateImage(cvSize((oRect.right-oRect.left+1),(oRect.bottom-oRect.top+1)),pBinaryImage->depth,pBinaryImage->nChannels);
                CloneImage(pBinaryImage,character,m_oCharacterList[k3].left,m_oCharacterList[k3].top);
            }
            if (oRect.right>pBinaryImage->width-2)
            {
                oCharacterVector.bedged[oCharacterVector.number]=true;
            }
            
            //	 cvRectangle(pBinaryImage,cvPoint(m_oCharacterList[k3].left,m_oCharacterList[k3].top),cvPoint(m_oCharacterList[k3].right,m_oCharacterList[k3].bottom),cvScalar(100,100,100));	
            oCharacterVector.ImageVector[oCharacterVector.number]=character;
            oCharacterVector.pRect[oCharacterVector.number].x=oRect.left;
            oCharacterVector.pRect[oCharacterVector.number].y=oRect.top;
            oCharacterVector.pRect[oCharacterVector.number].width=oRect.right-oRect.left+1;
            oCharacterVector.pRect[oCharacterVector.number].height=oRect.bottom-oRect.top+1;
            oCharacterVector.nAverageWidth+=oCharacterVector.pRect[oCharacterVector.number].width;
            nAverageHeight+=oCharacterVector.pRect[oCharacterVector.number].height;
            nAverageWidth+=oCharacterVector.pRect[oCharacterVector.number].width;
            if (oCharacterVector.number==0)
            {
                nCharacters++;
                oCharacterVector.LineInfo[nLineIndex]=nCharacters;
            }
            else if (oCharacterVector.number>0&&oCharacterVector.pRect[oCharacterVector.number].y==oCharacterVector.pRect[oCharacterVector.number-1].y) //查找相邻且同行字符，计算间隔
            {
                
                nInternal++;
                nCharacters++;
                oCharacterVector.LineInfo[nLineIndex]=nCharacters;
                oCharacterVector.nAverageInternal+=oCharacterVector.pRect[oCharacterVector.number].x-oCharacterVector.pRect[oCharacterVector.number-1].x-oCharacterVector.pRect[oCharacterVector.number-1].width;
            }
            else if(oCharacterVector.number>0)
            {
                nCharacters=1;
                nLineIndex++;
            }
            
            oCharacterVector.number++;
            
        }
        
    }	
    
#pragma endregion
    
    nAverageHeight/=oCharacterVector.number;
    nAverageWidth/=oCharacterVector.number;
    if ((oCharacterVector.pRect[oCharacterVector.number-2].height<=0.7*nAverageHeight)&&(oCharacterVector.pRect[oCharacterVector.number-1].height<=0.7*nAverageHeight)&&(oCharacterVector.pRect[oCharacterVector.number-1].y<=(oCharacterVector.pRect[oCharacterVector.number-2].y+oCharacterVector.pRect[oCharacterVector.number-2].height)))
    {//当最后一个字符为单独一行且为上下结构时容易出现分割错误 如“号”判断条件为： 最后两个字符高度均小于平均高度的0.7
        Wb_Rect oRect;
        oRect.left=fmin(oCharacterVector.pRect[oCharacterVector.number-2].x,oCharacterVector.pRect[oCharacterVector.number-1].x);
        oRect.top=oCharacterVector.pRect[oCharacterVector.number-2].y;
        oRect.bottom=oCharacterVector.pRect[oCharacterVector.number-1].y+oCharacterVector.pRect[oCharacterVector.number-1].height;
        oRect.right=fmax(oCharacterVector.pRect[oCharacterVector.number-2].x+oCharacterVector.pRect[oCharacterVector.number-2].width,oCharacterVector.pRect[oCharacterVector.number-1].x+oCharacterVector.pRect[oCharacterVector.number-1].width);
        cvReleaseImage(&oCharacterVector.ImageVector[oCharacterVector.number-2]);
        cvReleaseImage(&oCharacterVector.ImageVector[oCharacterVector.number-1]);
        IplImage *character=0;
        character=cvCreateImage(cvSize((oRect.right-oRect.left+1),(oRect.bottom-oRect.top+1)),pBinaryImage->depth,pBinaryImage->nChannels);
        CloneImage(pBinaryImage,character,oRect.left,oRect.top);
        
        oCharacterVector.ImageVector[oCharacterVector.number-2]=character;
        oCharacterVector.pRect[oCharacterVector.number-2].x=oRect.left;
        oCharacterVector.pRect[oCharacterVector.number-2].y=oRect.top;
        oCharacterVector.pRect[oCharacterVector.number-2].width=oRect.right-oRect.left+1;
        oCharacterVector.pRect[oCharacterVector.number-2].height=oRect.bottom-oRect.top+1;
        oCharacterVector.number--;
        
    }
    
    oCharacterVector.nAverageWidth=oCharacterVector.nAverageWidth/oCharacterVector.number;
    oCharacterVector.nAverageInternal=oCharacterVector.nAverageInternal/nInternal;
    
    
    delete m_oCharacterList;
    delete []oXInfo;
    delete []oYInfo;
    delete []pRowSum;
    delete []pColSum;
    return ;
}

void convertpositions(int xoffset, int yoffset,int nRemoveTop, int nRemoveBottom,CharacterVector &oCharacterVector)
{
    bool bfirstLine=true;
    int nfirstLine=0;
    for (int i=0;i<oCharacterVector.number;i++)
    {
        if (i==0)  // 第一行字符需要加上xoffset以及yoffset
        {
            nfirstLine=oCharacterVector.pRect[i].y;
        }
        if (nfirstLine!=oCharacterVector.pRect[i].y) // 非第一行字符除了加上xoffset和yoffset之外，还要加上去掉的姓名，生日等字符信息
        {
            oCharacterVector.pRect[i].y+=(nRemoveBottom-nRemoveTop+1);
        }
        oCharacterVector.pRect[i].x+=xoffset-1;
        oCharacterVector.pRect[i].y+=yoffset-1;
        
    }
    
    return;
}

int recgonize(IplImage* iimg, const char *imagePath)
{
    long op,ed;
    op=clock();	
    IplImage* pOriginalImage=FindSquare(iimg,8);
    
    if (pOriginalImage==NULL)
    {
        printf("无法检测到身份证在该背景中的区域，请重试！");
        return -1;
    }
    
    char* imageFilePath = new char[500];
    sprintf(imageFilePath,"%s1.jpg",imagePath);
    cvSaveImage(imageFilePath,pOriginalImage);
    delete[] imageFilePath;
    
    ed=clock();
    printf("find square:%ldms\n",ed-op);
    op=clock();	
    bool res=false;
    CharacterVector *pVector=0;
    
    res=fengeNameAndAddress(pOriginalImage,pVector,imagePath);
    if (res==false)
    {
        // 当检测失败时，将图像旋转180度； add 2012-7-10
        IplImage *pTemp=cvCreateImage(cvSize(iimg->width,iimg->height),iimg->depth,iimg->nChannels);
        
        cvFlip(iimg,pTemp,-1);
        pOriginalImage=FindSquare(pTemp,8);
        
        
        fengeNameAndAddress(pOriginalImage,pVector,imagePath);
        cvReleaseImage(&pTemp);
    }
    ed=clock();
    printf("DetectNameAndAddress%ldms\n",ed-op);
    op=clock();
    // following will detect Number
    //IplImage *pBinary=0;
    //fengeID2(pOriginalImage,pBinary);
    
    IplImage *pBinaryID=0;
    pBinaryID=fengeID(pOriginalImage);
    
    //测试返回二值图像结果
    if (pBinaryID)
    {
        char* imageFilePath = new char[500];
        sprintf(imageFilePath,"%sID.jpg",imagePath);
        WriteImage(pBinaryID,imageFilePath);
        delete[] imageFilePath;
    }
    
    
    delete pVector;
    cvReleaseImage(&pOriginalImage);
    //cvReleaseImage(&pBinary);
    cvReleaseImage(&pBinaryID);
    
    ed=clock();
    printf("DetectPersonalID%ldms\n",ed-op);
    
    return 0;
    
    
}

IplImage * fengeID(IplImage *pOriginalImage)
{
    
    // 从模版图像上分割身份证号码，并储存在pNumberROI
    IplImage* pNumberROI=cvCreateImage(cvSize(pOriginalImage->width*ID_WIDTH,pOriginalImage->height*ID_HEIGH),pOriginalImage->depth,pOriginalImage->nChannels);
    CloneImage(pOriginalImage,pNumberROI,ID_XPOS*pOriginalImage->width,ID_YPOS*pOriginalImage->height);
    //精确定位数字ROI上边界
    int nTop=ExtractNumbers(pNumberROI);
    if (nTop>0)
    {
        IplImage *pTemp=cvCreateImage(cvSize(pNumberROI->width,pNumberROI->height-nTop),pNumberROI->depth,pNumberROI->nChannels);
        CloneImage(pNumberROI,pTemp,0,nTop);
        cvReleaseImage(&pNumberROI);
        pNumberROI=cvCreateImage(cvGetSize(pTemp),pTemp->depth,pTemp->nChannels);
        CloneImage(pTemp,pNumberROI);
        cvReleaseImage(&pTemp);
    }
    //分割完毕
    IplImage* pBinaryImage=NULL;
    //cvReleaseImage(&pBinaryImage);
    pBinaryImage=cvCreateImage(cvGetSize(pNumberROI),pNumberROI->depth,1);
    RunEqualizeHistogram(pNumberROI,pNumberROI);
    
    
    // set general threshold on image
    for (int i=0;i<pNumberROI->width;i++)
    {
        for (int j=0;j<pNumberROI->height;j++)
        {
            float fPixR=0.0;
            float fPixG=0.0;
            float fPixB=0.0;
            GetImagePixel(pNumberROI,i,j,fPixR,0);
            GetImagePixel(pNumberROI,i,j,fPixG,1);
            GetImagePixel(pNumberROI,i,j,fPixB,2);
            if (fPixR<ID_THRE&&fPixG<ID_THRE&&fPixB<ID_THRE)
            {
                SetImagePixel(pBinaryImage,i,j,0);
            }
            else
            {
                SetImagePixel(pBinaryImage,i,j,255);
            }
        }
    }
    
    // 简单对图像进行开操作，去除斑点噪声
    IplConvKernel *kernal=cvCreateStructuringElementEx(3,3,2,2,CV_SHAPE_RECT);
    cvMorphologyEx(pBinaryImage,pBinaryImage,NULL,kernal,CV_MOP_DILATE);
    cvMorphologyEx(pBinaryImage,pBinaryImage,NULL,kernal,CV_MOP_ERODE);
    cvReleaseStructuringElement(&kernal);
    cvReleaseImage(&pNumberROI);
    
    
    return pBinaryImage;
}






void SubBackGround(IplImage *pBinaryImage)
{
    
    if (!pBinaryImage)
    {
        
        return ;
    }
    int nchannel=pBinaryImage->nChannels;
    int nWidth=pBinaryImage->width;
    int nHeight=pBinaryImage->height;
    if (nchannel!=1)
    {
        
        return ;
    }
    int nSeedX=1;
    int nSeedY=1;
    
    
    // 8-neighborhood directions
    int nDx[]={-1,0,1,-1,1,-1,0,1};
    int nDy[]={-1,-1,-1,0,0,1,1,1};
    // 8-neighborhood directions control flag
    int k =0;
    
    
    // define stack for storing the region coordinate and process flag
    int * pnGrowQueX ;
    int * pnGrowQueY ;
    int * pnProcessFlag;
    pnGrowQueX = new int [nWidth*nHeight];
    pnGrowQueY = new int [nWidth*nHeight];
    pnProcessFlag=new int[nWidth*nHeight];
    
    for (int i=0;i<nWidth*nHeight;i++)
    {
        pnProcessFlag[i]=0;
    }
    
    // define the start flag and end flag for the region stack
    // if nStart>nEnd, represent region is empty
    // if nStart=nEnd, represent only one point in the stack
    int nStart;
    int nEnd ;
    nStart = 0 ;
    nEnd = 0 ;
    pnGrowQueX[nEnd] = nSeedX;
    pnGrowQueY[nEnd] = nSeedY;
    
    
    // current processed pixel
    int nCurrX =nSeedX;
    int nCurrY =nSeedY;
    
    // represent one of current pixel's 8-neighborhood
    int xx=0;
    int yy=0;
    
    
    while (nStart<=nEnd)
    {
        
        
        nCurrX = pnGrowQueX[nStart];
        nCurrY = pnGrowQueY[nStart];
        
        float nPixelValue=0;
        
        GetImagePixel(pBinaryImage,nCurrX,nCurrY,nPixelValue);
        
        
        // check the current pixel's 8-neighborhood
        for (k=0; k<8; k++) 
        { 
            
            xx = nCurrX+nDx[k] ;
            yy = nCurrY+nDy[k];
            float nComparedPixelValue=0;
            GetImagePixel(pBinaryImage,xx,yy,nComparedPixelValue);
            if ( (xx < nWidth) && (xx>=0) && (yy>=0) && (yy<nHeight) && pnProcessFlag[yy*nWidth+xx]==0)
            {
                //if the pixel is in image
                //if the pixel is processed
                //if the pixel satisfy region growing condition
                
                if (abs(nPixelValue-nComparedPixelValue)<10)
                {
                    nEnd++;
                    // push (xx，yy) in stack
                    pnGrowQueX[nEnd] = xx; 
                    pnGrowQueY[nEnd] = yy;
                    pnProcessFlag[yy*nWidth+xx] = 1;
                    SetImagePixel(pBinaryImage,xx,yy,255);
                    
                }
                
            }
            
        }
        nStart++;
    }
    
    delete []pnGrowQueX;
    delete []pnGrowQueY;
    delete []pnProcessFlag;
    
    return ;
}

