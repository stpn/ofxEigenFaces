
///Ported from:
//  OnlineFaceRec.cpp, by Shervin Emami (www.shervinemami.info) on 30th Dec 2011.
// Online Face Recognition from a camera using Eigenfaces.
//
// Some parts are based on the code example by Robin Hewitt (2007) at:
// "http://www.cognotics.com/opencv/servo_2007_series/part_5/index.html"
//
// Command-line Usage (for offline mode, without a webcam):
//
// First, you need some face images. I used the ORL face database.
// You can download it for free at
//    www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html
//
// List the training and test face images you want to use in the
// input files train.txt and test.txt. (Example input files are provided
// in the download.) To use these input files exactly as provided, unzip
// the ORL face database, and place train.txt, test.txt, and eigenface.exe
// at the root of the unzipped database.
//
// To run the learning phase of eigenface, enter in the command prompt:
//    OnlineFaceRec train <train_file>
// To run the recognition phase, enter:
//    OnlineFaceRec test <test_file>
// To run online recognition from a camera, enter:
//    OnlineFaceRec
//
//////////////////////////////////////////////////////////////////////////////////////
//
//  Stepan Boltalin  2/23/12.


#pragma once
#include "ofMain.h"
#include "ofxOpenCv.h"
#include "eigenObjects.h"
#include <stdio.h>

#include <stdio.h>		// For getchar() on Linux
#include <termios.h>	// For kbhit() on Linux
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>	// For mkdir(path, options) on Linux

#include <vector>
#include <string>

// #include "cvaux.h"
#include "highgui.h"

#ifndef BOOL
#define BOOL bool
#endif

class ofxEigenFace {
public:
    ofxEigenFace();
    
    string name;
    char newPersonName[256];
    string namename;
    CvRect faceRect;
    bool conf;

};


class ofxEigenFaceFinder {

public:
    
    
    ofxEigenFaceFinder();    
    
    ofxEigenFace recognizeFromCam(ofxCvColorImage img);

    
    const char *faceCascadeFilename;
    
    void startTraining();
    void stopTraining();
    void addPerson(string name);
    
    string name;
    char newPersonName[256];
    string namename;
    CvRect faceRect;
    int newPersonFaces;


    
    
    int name_x;
    int name_y;
    
    
    bool conf;

    private:

    void learnFace(string name);

    
    void storeEigenfaceImages();
    void printUsage();
    void learn(const char *szFileTrain);
    void doPCA();
    void storeTrainingData();
    int  loadTrainingData(CvMat ** pTrainPersonNumMat);
    int  findNearestNeighbor(float * projectedTestFace);
    int findNearestNeighbor(float * projectedTestFace, float *pConfidence);
    int  loadFaceImgArray(const char * filename);
    void recognizeFileList(const char *szFileTest);
    IplImage* getCameraFrame(void);
    IplImage* convertImageToGreyscale(const IplImage *imageSrc);
    IplImage* cropImage(const IplImage *img, const CvRect region);
    IplImage* resizeImage(const IplImage *origImg, int newWidth, int newHeight);
    IplImage* convertFloatImageToUcharImage(const IplImage *srcImg);
    void saveFloatImage(const char *filename, const IplImage *srcImg);
    CvRect detectFaceInImage(const IplImage *inputImg, const CvHaarClassifierCascade* cascade );
    CvMat* retrainOnline(void);

    IplImage ** faceImgArr; // array of face images
    
    int SAVE_EIGENFACE_IMAGES;		// Set to 0 if you dont want images of the Eigenvectors saved to files (for debugging).
    //#define USE_MAHALANOBIS_DISTANCE	// You might get better recognition accuracy if you enable this.
    
    
    
    
    // Global variables
    CvMat    *  personNumTruthMat; // array of person numbers
    //#define	MAX_NAME_LENGTH 256		// Give each name a fixed size for easier code.
    //char **personNames = 0;			// array of person names (indexed by the person number). Added by Shervin.
    vector<string> personNames;			// array of person names (indexed by the person number). Added by Shervin.
    int faceWidth;	// Default dimensions for faces in the face recognition database. Added by Shervin.
    int faceHeight;	//	"		"		"		"		"		"		"		"
    int nPersons; // the number of people in the training set. Added by Shervin.
    int nTrainFaces; // the number of training images
    int nEigens; // the number of eigenvalues
    IplImage * pAvgTrainImg; // the average image
    IplImage ** eigenVectArr; // eigenvectors
    CvMat * eigenValMat; // eigenvalues
    CvMat * projectedTrainFaceMat; // projected training faces
    
    CvCapture* camera;	// The camera device.
    int startState;
    int state;
    string filename; //needs a "haarcascade_frontalface_default.xml in data/bin
    
 

};
