#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/videoio.hpp>

#include<iostream>
#include<mutex>         
#include<thread>

#include "blob.h"

using namespace cv;
using namespace std;

//voids
void matchCurrentFrameBlobsToExistingBlobs(vector<Blob>& existingBlobs, vector<Blob>& currentFrameBlobs);
void addBlobToExistingBlobs(Blob& currentFrameBlob, vector<Blob>& existingBlobs, int& intIndex);
void addNewBlob(Blob& currentFrameBlob, vector<Blob>& existingBlobs);
double distanceBetweenPoints(Point point1, Point point2);
void drawAndShowContours(Size imageSize, vector< vector< Point> > contours, string strImageName);
void drawAndShowContours(Size imageSize, vector<Blob> blobs, string strImageName);
void drawBlobInfoOnImage(vector<Blob>& blobs, Mat& frame2Copy);



//colours
const Scalar BLACK = Scalar(0.0, 0.0, 0.0);
const Scalar WHITE = Scalar(255.0, 255.0, 255.0);
const Scalar YELLOW = Scalar(0.0, 255.0, 255.0);
const Scalar GREEN = Scalar(0.0, 200.0, 0.0);
const Scalar RED = Scalar(0.0, 0.0, 255.0);


VideoCapture capVideo;
mutex mtx;
Mat frame1;
Mat frame2;
bool firstFrame = true;
int frameCount = 2;
char CheckForEsc;


void camera() {
	capVideo.open(0);
	mtx.lock();

	capVideo.set(CAP_PROP_FRAME_WIDTH, 640);
	capVideo.set(CAP_PROP_FRAME_HEIGHT, 360);

	//capturing 2 frames
	capVideo.read(frame1);
	capVideo.read(frame2);
	mtx.unlock();
}



int main(void) {


	while (true) {
		thread cameras(camera);

		if (cameras.joinable()) {
			cameras.join();
		}
		while (capVideo.isOpened()) {
			vector<Blob> blobs;			//create the blobs vector
			vector<Blob> currentFrameBlobs;

			Mat frame1Copy = frame1.clone();
			Mat frame2Copy = frame2.clone();


			//constructing the black and white images with only the moving countours
			cvtColor(frame1Copy, frame1Copy, COLOR_BGR2GRAY);
			cvtColor(frame2Copy, frame2Copy, COLOR_BGR2GRAY);

			GaussianBlur(frame1Copy, frame1Copy, Size(5, 5), 0);
			GaussianBlur(frame2Copy, frame2Copy, Size(5, 5), 0);

			Mat imgDifference;
			Mat thresh;

			absdiff(frame1Copy, frame2Copy, imgDifference);
			threshold(imgDifference, thresh, 30, 255.0, THRESH_BINARY);
			imshow("thresh", thresh);

			//start to fill in the countours by dilating every pixel inside the binary by 9, 7, 5, or 3 pixels to fill in a countour
			Mat kernel3 = getStructuringElement(MORPH_RECT, Size(3, 3));
			Mat kernel5 = getStructuringElement(MORPH_RECT, Size(5, 5));

			dilate(thresh, thresh, kernel3, Point(-1, -1), 5);
			erode(thresh, thresh, kernel3, Point(-1, -1), 5);


			//find the countours of the moving parts and show it
			vector<vector<Point> > contours;
			Mat threshCopy = thresh.clone();

			findContours(threshCopy, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
			drawAndShowContours(thresh.size(), contours, "Img Threshold");

			//create the convex hulls and show it
			vector<vector<Point> > convexHulls(contours.size());

			for (unsigned int i = 0; i < contours.size(); i++) {
				convexHull(contours[i], convexHulls[i]);
			}

			drawAndShowContours(thresh.size(), convexHulls, "imgConvexHulls");

			// filtering unlikely blobs,
			// every blob is pushed back into the blob vectors
			for (auto& convexHull : convexHulls) {
				Blob possibleBlob(convexHull);

				if (possibleBlob.currentBoundingRect.area() > 600 &&
					possibleBlob.dblCurrentAspectRatio > 0.3 &&
					possibleBlob.dblCurrentAspectRatio < 7.0 &&
					possibleBlob.currentBoundingRect.width > 60 &&
					possibleBlob.currentBoundingRect.height > 60 &&
					possibleBlob.dblCurrentDiagonalSize > 120.0 &&
					(contourArea(possibleBlob.currentContour) / (double)possibleBlob.currentBoundingRect.area()) > 0.50) {
					currentFrameBlobs.push_back(possibleBlob);
				}
			}

			drawAndShowContours(thresh.size(), currentFrameBlobs, "imgCurrentFrameBlobs");


			//update the blob vectors 
			if (firstFrame == true) {
				for (auto& currentFrameBlob : currentFrameBlobs) {
					blobs.push_back(currentFrameBlob);
				}
			}
			else {
				matchCurrentFrameBlobsToExistingBlobs(blobs, currentFrameBlobs);
			}

			drawAndShowContours(thresh.size(), blobs, "imgBlobs");

			frame2Copy = frame2.clone();          // get another copy of frame 2 since we changed the previous frame 2 copy in the processing above


			//draw the bounding rectangles
			drawBlobInfoOnImage(blobs, frame2Copy);


			imshow("Frame2Copy", frame2Copy);

			// waitKey(0);                 // uncomment this line to go frame by frame for debugging

			// now we prepare for the next set of frames

			currentFrameBlobs.clear();

			frame1 = frame2.clone();           // move frame 1 up to where frame 2 is


			if ((capVideo.get(CAP_PROP_POS_FRAMES) + 1) < capVideo.get(CAP_PROP_FPS)) {
				capVideo.read(frame2);
			}
			else {
				cout << "end of video\n";
				break;
			}

			firstFrame = false;
			frameCount++;
			CheckForEsc = waitKey(1);

		}

		if (CheckForEsc != 27) {               // if the user did not press esc (i.e. we reached the end of the video)
			waitKey(0);                         // hold the windows open to allow the "end of video" message to show
		}

		// note that if the user did press esc, we don't need to hold the windows open, we can simply let the program end which will close the windows
		return(0);
	}
}





void addBlobToExistingBlobs(Blob& currentFrameBlob, vector<Blob>& existingBlobs, int& intIndex) {

	existingBlobs[intIndex].currentContour = currentFrameBlob.currentContour;
	existingBlobs[intIndex].currentBoundingRect = currentFrameBlob.currentBoundingRect;

	existingBlobs[intIndex].centerPositions.push_back(currentFrameBlob.centerPositions.back());

	existingBlobs[intIndex].dblCurrentDiagonalSize = currentFrameBlob.dblCurrentDiagonalSize;
	existingBlobs[intIndex].dblCurrentAspectRatio = currentFrameBlob.dblCurrentAspectRatio;

	existingBlobs[intIndex].blnStillBeingTracked = true;
	existingBlobs[intIndex].blnCurrentMatchFoundOrNewBlob = true;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
void addNewBlob(Blob& currentFrameBlob, vector<Blob>& existingBlobs) {

	currentFrameBlob.blnCurrentMatchFoundOrNewBlob = true;

	existingBlobs.push_back(currentFrameBlob);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
double distanceBetweenPoints(Point point1, Point point2) {

	int intX = abs(point1.x - point2.x);
	int intY = abs(point1.y - point2.y);
	//pyhtagorean theorem to find the shortest distance between predictions
	return(sqrt(pow(intX, 2) + pow(intY, 2)));
}

///////////////////////////////////////////////////////////////////////////////////////////////////
void drawAndShowContours(Size imageSize, vector< vector< Point> > contours, string strImageName) {
	Mat image(imageSize, CV_8UC3, BLACK);

	drawContours(image, contours, -1, WHITE, -1);

	imshow(strImageName, image);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
void drawAndShowContours(Size imageSize, vector<Blob> blobs, string strImageName) {

	Mat image(imageSize, CV_8UC3, BLACK);

	vector< vector< Point> > contours;

	for (auto& blob : blobs) {
		if (blob.blnStillBeingTracked == true) {
			contours.push_back(blob.currentContour);
		}
	}

	drawContours(image, contours, -1, WHITE, -1);

	imshow(strImageName, image);
}


///////////////////////////////////////////////////////////////////////////////////////////////////
void drawBlobInfoOnImage(vector<Blob>& blobs, Mat& frame2Copy) {

	for (unsigned int i = 0; i < blobs.size(); i++) {

		if (blobs[i].blnStillBeingTracked == true) {
			rectangle(frame2Copy, blobs[i].currentBoundingRect, RED, 2);

			//uncomment here to display the tracking info on the rectangles
				/*int intFontFace = FONT_HERSHEY_SIMPLEX;
				double dblFontScale = blobs[i].dblCurrentDiagonalSize / 60.0;
				int intFontThickness = (int) round(dblFontScale * 1.0);
				putText(frame2Copy,  to_string(i), blobs[i].centerPositions.back(), intFontFace, dblFontScale, GREEN, intFontThickness);
				*/
		}
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////
void drawtriggerOnImage(int& triggerIn, int& triggerOut, Mat& frame2Copy) {

	int intFontFace = FONT_HERSHEY_SIMPLEX;
	double dblFontScale = (frame2Copy.rows * frame2Copy.cols) / 300000.0;
	int intFontThickness = (int)round(dblFontScale * 1.5);

	Size textSizeIn = getTextSize(to_string(triggerIn), intFontFace, dblFontScale, intFontThickness, 0);
	Size textSizeOut = getTextSize(to_string(triggerOut), intFontFace, dblFontScale, intFontThickness, 0);
	Point textIn;
	Point textOut;

	textIn.x = frame2Copy.cols - 1 - (int)((double)textSizeIn.width * 1.25);
	textIn.y = (int)((double)textSizeIn.height * 1.25);

	textOut.x = frame2Copy.cols - 200 - (int)((double)textSizeOut.width * 1.25);
	textOut.y = (int)((double)textSizeOut.height * 1.25);

	putText(frame2Copy, to_string(triggerIn), textIn, intFontFace, dblFontScale, GREEN, intFontThickness);
	putText(frame2Copy, to_string(triggerOut), textOut, intFontFace, dblFontScale, YELLOW, intFontThickness);


}


void matchCurrentFrameBlobsToExistingBlobs(vector<Blob>& existingBlobs, vector<Blob>& currentFrameBlobs) {

	for (auto& existingBlob : existingBlobs) {

		existingBlob.blnCurrentMatchFoundOrNewBlob = false;

		existingBlob.predictNextPosition();
	}

	for (auto& currentFrameBlob : currentFrameBlobs) {

		int intIndexOfLeastDistance = 0;
		double dblLeastDistance = 100000.0;

		for (unsigned int i = 0; i < existingBlobs.size(); i++) {

			if (existingBlobs[i].blnStillBeingTracked == true) {

				double dblDistance = distanceBetweenPoints(currentFrameBlob.centerPositions.back(), existingBlobs[i].predictedNextPosition);

				if (dblDistance < dblLeastDistance) {
					dblLeastDistance = dblDistance;
					intIndexOfLeastDistance = i;
				}
			}
		}

		if (dblLeastDistance < currentFrameBlob.dblCurrentDiagonalSize * 0.5) {
			addBlobToExistingBlobs(currentFrameBlob, existingBlobs, intIndexOfLeastDistance);
		}
		else {
			addNewBlob(currentFrameBlob, existingBlobs);
		}

	}

	for (auto& existingBlob : existingBlobs) {

		if (existingBlob.blnCurrentMatchFoundOrNewBlob == false) {
			existingBlob.intNumOfConsecutiveFramesWithoutAMatch++;
		}

		if (existingBlob.intNumOfConsecutiveFramesWithoutAMatch >= 5) {
			existingBlob.blnStillBeingTracked = false;
		}

	}

}
