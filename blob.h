

#ifndef MY_BLOB
#define MY_BLOB

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>


using namespace cv;
using namespace std;


class Blob {
public:
	// member variables 
	vector<cv::Point> currentContour;

	Rect currentBoundingRect;

	vector<cv::Point> centerPositions;

	double dblCurrentDiagonalSize;
	double dblCurrentAspectRatio;

	bool blnCurrentMatchFoundOrNewBlob;

	bool blnStillBeingTracked;

	int intNumOfConsecutiveFramesWithoutAMatch;

	Point predictedNextPosition;

	// function prototypes ////////////////////////////////////////////////////////////////////////
	Blob(vector<Point> _contour);
	void predictNextPosition(void);

};

#endif    // MY_BLOB
