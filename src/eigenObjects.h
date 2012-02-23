/****************************************************************************************\
 *                                  Eigen objects                                         *
 \****************************************************************************************/

typedef int (CV_CDECL * CvCallback)(int index, void* buffer, void* user_data);


#define CV_EIGOBJ_NO_CALLBACK     0
#define CV_EIGOBJ_INPUT_CALLBACK  1
#define CV_EIGOBJ_OUTPUT_CALLBACK 2
#define CV_EIGOBJ_BOTH_CALLBACK   3

/* Calculates covariation matrix of a set of arrays */
CVAPI(void)  cvCalcCovarMatrixEx( int nObjects, void* input, int ioFlags,
                                 int ioBufSize, uchar* buffer, void* userData,
                                 IplImage* avg, float* covarMatrix );

/* Calculates eigen values and vectors of covariation matrix of a set of
 arrays */
CVAPI(void)  cvCalcEigenObjects( int nObjects, void* input, void* output,
                                int ioFlags, int ioBufSize, void* userData,
                                CvTermCriteria* calcLimit, IplImage* avg,
                                float* eigVals );

/* Calculates dot product (obj - avg) * eigObj (i.e. projects image to eigen vector) */
CVAPI(double)  cvCalcDecompCoeff( IplImage* obj, IplImage* eigObj, IplImage* avg );

/* Projects image to eigen space (finds all decomposion coefficients */
CVAPI(void)  cvEigenDecomposite( IplImage* obj, int nEigObjs, void* eigInput,
                                int ioFlags, void* userData, IplImage* avg,
                                float* coeffs );

/* Projects original objects used to calculate eigen space basis to that space */
CVAPI(void)  cvEigenProjection( void* eigInput, int nEigObjs, int ioFlags,
                               void* userData, float* coeffs, IplImage* avg,
                               IplImage* proj );

