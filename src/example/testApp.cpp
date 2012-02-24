

#include "testApp.h"


int name_x;
int name_y;


bool keyN = FALSE;
bool keyT = FALSE;




//--------------------------------------------------------------
void testApp::setup(){
    
    ofTrueTypeFont::setGlobalDpi(72);

    w = 400;
    h = 300;
    
	ofSetVerticalSync(true);
    vidGrabber.setVerbose(true);
    vidGrabber.initGrabber(w,h);
    colorImg.allocate(w,h);
    colorImg2.allocate(w,h);

    
	myFont.loadFont("verdana.ttf", 14, true, true);
    myFont.setLineHeight(18.0f);
	myFont.setLetterSpacing(1.037);

	ofSetFrameRate(30);


}



//--------------------------------------------------------------
void testApp::update(){
    
    ofBackground(100,100,100);

    bool bNewFrame = false;
    
    vidGrabber.grabFrame();
    bNewFrame = vidGrabber.isFrameNew();
	
	if (bNewFrame){
        colorImg.setFromPixels(vidGrabber.getPixels(), w,h);
        
    }

}

//--------------------------------------------------------------
void testApp::draw(){
    ofxEigenFace foundFace = faceFinder.recognizeFromCam(colorImg);

   

    colorImg.draw(0, 0);
    
    if (foundFace.conf ==  TRUE){
    ofDrawBitmapString(foundFace.namename, foundFace.faceRect.x, foundFace.faceRect.y);
    }

}

//--------------------------------------------------------------
void testApp::keyPressed(int key){
    if (key == 'n'){	
        
        faceFinder.addPerson("name");
    }
    if (key == 't'){	
        
        faceFinder.startTraining();
        
    }
    
    if (key == 's'){	
        
        faceFinder.stopTraining();        
    }

}
//--------------------------------------------------------------
void testApp::keyReleased(int key){

}

//--------------------------------------------------------------
void testApp::mouseMoved(int x, int y ){

}

//--------------------------------------------------------------
void testApp::mouseDragged(int x, int y, int button){

}

//--------------------------------------------------------------
void testApp::mousePressed(int x, int y, int button){

}

//--------------------------------------------------------------
void testApp::mouseReleased(int x, int y, int button){

}

//--------------------------------------------------------------
void testApp::windowResized(int w, int h){

}

//--------------------------------------------------------------
void testApp::gotMessage(ofMessage msg){

}

//--------------------------------------------------------------
void testApp::dragEvent(ofDragInfo dragInfo){ 

}