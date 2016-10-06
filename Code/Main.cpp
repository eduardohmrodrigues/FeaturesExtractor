#define _USE_MATH_DEFINES

#include "Main.h"
#include "util.h"
#include <math.h>
#include <Windows.h>

// OpenCV includes
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"

#include <iostream>
#include <ctype.h>
#include "GL\glut.h"

#include "psapi.h"

using namespace cv;

//When we equalize the histogram, the feature detection works worse than in normal lighting situations
//However when the scene changes the lighting, equalize the histogram makes the feature detection works better
#define EQUALIZE_TEXTURE_HISTOGRAM	false
#define EQUALIZE_CAMERA_HISTOGRAM	false

#define GAUSSIAN_BLUR_TEXTURE	false
#define GAUSSIAN_BLUR_CAMERA	false
#define GAUSSIAN_SIZE	9

#define MEDIAN_BLUR_TEXTURE false
#define MEDIAN_BLUR_CAMERA	false
#define MEDIAN_BLUR_SIZE	3

#define GENERATE_RELATORY	false

#define RESIZE_CAMERA_RESOLUTION	false
#define RESOLUTION_PARAM			3

// Use to convert bytes to MB
#define DIV 1024

enum algorithm{
	SIFT_ALG = 0,
	SURF_ALG = 1,
	ORB_ALG = 2,
	RANSAC_ALG = 3,
	LMEDS_ALG = 4,
	REGULAR_ALG = 5
};

string window1 = "Webcam";
string window2 = "Keypoints";
string filepath = "Resources/Textures/manga.jpg";
string video_filepath = "Resources/InputData/videoManga.mp4";

string sift_ransac_memory_txt = "Resources/MemoryUsage/sift_ransac_memory.txt";
string sift_lmeds_memory_txt = "Resources/MemoryUsage/sift_lmeds_memory.txt";
string sift_regular_memory_txt = "Resources/MemoryUsage/sift_regular_memory.txt";
string surf_ransac_memory_txt = "Resources/MemoryUsage/surf_ransac_memory.txt";
string surf_lmeds_memory_txt = "Resources/MemoryUsage/surf_lmeds_memory.txt";
string surf_regular_memory_txt = "Resources/MemoryUsage/surf_regular_memory.txt";

string sift_ransac_processor_txt = "Resources/processorUsage/sift_ransac_processor.txt";
string sift_lmeds_processor_txt = "Resources/processorUsage/sift_lmeds_processor.txt";
string sift_regular_processor_txt = "Resources/processorUsage/sift_regular_processor.txt";
string surf_ransac_processor_txt = "Resources/processorUsage/surf_ransac_processor.txt";
string surf_lmeds_processor_txt = "Resources/processorUsage/surf_lmeds_processor.txt";
string surf_regular_processor_txt = "Resources/processorUsage/surf_regular_processor.txt";

bool LOAD_ERROR = false;

Mat						cameraImage, originalGrayTexture, originalColorTexture,
keypointsImage, descriptor1, descriptor2, img_matches,
cameraGrayImage, laplacianCamera;

VideoCapture			capture;
vector<KeyPoint>		cameraKeypoints, imageKeypoints;
vector<Point3d>			features3Dtexture, features3Dcamera;
vector<Point2d>			features2Dtexture, features2Dcamera;

FlannBasedMatcher		matcher;
vector<DMatch>			matches, goodMatches;

SiftFeatureDetector		siftDetector;
SiftDescriptorExtractor siftExtractor;

OrbFeatureDetector		orbDetector;
OrbDescriptorExtractor	orbExtractor;

SurfFeatureDetector*		surfDetector;
SurfDescriptorExtractor	surfExtractor;


algorithm				featureDetectorAlgorithm = SIFT_ALG,
						poseEstimationAlgorithm = RANSAC_ALG;



//to calculate fps
int total_video_frames;
time_t startTime;
time_t nowtime = std::time(nullptr);
time_t one_second = std::time(nullptr);
int fps = 1, calc_fps = 0;
//----

//To find the object
vector<Point2f> obj;
vector<Point2f> scene;
Mat homography;
vector<Point2f> obj_corners(4);
vector<Point2f> scene_corners(4);
//---

//Auxiliar
Mat textureGrayTemp, cameraGrayTemp, cameraColorTemp;

//To calculate memory usage
ofstream myMemoryFile, myProcessorFile; //to write nowtimes in txt

PROCESS_MEMORY_COUNTERS testa;
HANDLE					thisProcess;
vector<double>			memoryUsage;
double					lastMemoryUsage;
//----------

//To calculate cpu usage
static ULARGE_INTEGER lastCPU, lastSysCPU, lastUserCPU;
static int numProcessors;
double lastCPUstored = 0;
double atualCPU = 0;
vector<double> cpuUsage;
//-----

//To calculate pose
vector< Mat> rvecs, tvecs;
Mat intrinsic_Matrix(3, 3, CV_64F), distortion_coeffs;

Mat rvec, tvec;

struct cornerInformation{
	float x;
	float y;
	float x3;
	float y3;
	float z3;
};

int board_w = 5, board_h = 5;
int n_boards = 4;
float measure = 38;	//mm
Size imageSize;

vector< vector< Point2f> > imagePoints;
vector< vector< Point3f> > objectPoints;

Mat rotation, viewMatrix(4, 4, CV_64F);
//-----

//openGL
GLuint texture;

//

/**
*This function convert a vector of keypoints in a vector of 3D point
*@param keypoints input vector of keypoints
*@param points output vector of 3D points
*@param z value of z coordinate
*/
vector<Point3d> keypoint2point3d(vector<KeyPoint> keypoints, double z){
	vector<Point3d> points;
	Point3d aux;
	aux.z = z;
	for (int i = 0; i < keypoints.size(); i++){
		aux.x = keypoints[i].pt.x;
		aux.y = keypoints[i].pt.y;
		points.push_back(aux);
	}
	return points;
}

/**
*This function convert a vector of keypoints in a vector of 2D point
*@param keypoints input vector of keypoints
*@param points output vector of 2D points
*/
vector<Point2d> keypoint2point2d(vector<KeyPoint> keypoints){
	vector<Point2d> points;
	Point2d aux;
	for (int i = 0; i < keypoints.size(); i++){
		aux.x = keypoints[i].pt.x;
		aux.y = keypoints[i].pt.y;
		points.push_back(aux);
	}
	return points;
}

bool loadImageFeatures(Mat grey, Mat color, Mat descriptor, vector<KeyPoint> keypoints, string filepath, SiftFeatureDetector detector, SiftDescriptorExtractor extractor){
	bool haveError = false;

	grey = imread(filepath, CV_LOAD_IMAGE_GRAYSCALE);
	color = imread(filepath, CV_LOAD_IMAGE_ANYCOLOR | CV_LOAD_IMAGE_ANYDEPTH);

	if (grey.empty() || color.empty()){
		cout << "Image Not Loaded" << endl;
		haveError = true;
	}
	else{
		detector.detect(grey, keypoints);
		extractor.compute(grey, keypoints, descriptor);
	}

	return haveError;
}

/**
*This function create a .png image
*@param image Mat structure with the image
*@param keypointsImage the keypoints extracted
*@param color Color to draw the features on image
*@param filepath the path to save the file
*@param filename the filename
*@param createWindow bool to create or not a window with the image created
*@param windowName the name of window created
*/
void createPngWithFeatures(Mat image, vector<KeyPoint> keypointsImage, int color, string filepath, string filename, bool createWindow = false, string windowName = ""){
	Mat sift_extract_features;
	if (color == -1)
		drawKeypoints(image, keypointsImage, sift_extract_features, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	else
		drawKeypoints(image, keypointsImage, sift_extract_features, color, DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	if (createWindow)
		imshow(windowName, sift_extract_features);
	imwrite(filepath + "/" + filename, sift_extract_features);
}


void initCPU(){
	SYSTEM_INFO sysInfo;
	FILETIME ftime, fsys, fuser;

	GetSystemInfo(&sysInfo);
	numProcessors = sysInfo.dwNumberOfProcessors;

	GetSystemTimeAsFileTime(&ftime);
	memcpy(&lastCPU, &ftime, sizeof(FILETIME));

	GetProcessTimes(thisProcess, &ftime, &ftime, &fsys, &fuser);

	memcpy(&lastSysCPU, &fsys, sizeof(FILETIME));
	memcpy(&lastUserCPU, &fuser, sizeof(FILETIME));

}

double getCurrentProcessorUsage(){
	FILETIME ftime, fsys, fuser;
	ULARGE_INTEGER now, sys, user;
	double percent;


	GetSystemTimeAsFileTime(&ftime);
	memcpy(&now, &ftime, sizeof(FILETIME));


	GetProcessTimes(thisProcess, &ftime, &ftime, &fsys, &fuser);
	memcpy(&sys, &fsys, sizeof(FILETIME));
	memcpy(&user, &fuser, sizeof(FILETIME));
	percent = (sys.QuadPart - lastSysCPU.QuadPart) +
		(user.QuadPart - lastUserCPU.QuadPart);
	percent /= (now.QuadPart - lastCPU.QuadPart);
	percent /= numProcessors;
	lastCPU = now;
	lastUserCPU = user;
	lastSysCPU = sys;


	return percent * 100;
}

void orthogonalStart()
{
	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	gluOrtho2D(-800 / 3, 800 / 3, -600 / 3, 600 / 3);
	glMatrixMode(GL_MODELVIEW);
}

void orthogonalEnd()
{
	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glMatrixMode(GL_MODELVIEW);
}

void background()
{

	/*
	glDisable(GL_DEPTH_TEST);
	glBindTexture(GL_TEXTURE_2D, texture);

	orthogonalStart();

	// texture width/height
	const int iw = 800;
	const int ih = 600;

	glPushMatrix();
	glTranslatef(-iw / 2, -ih / 2, 0);
	glBegin(GL_QUADS);
	glTexCoord2i(0, 0); glVertex2i(0, 0);
	glTexCoord2i(1, 0); glVertex2i(iw, 0);
	glTexCoord2i(1, 1); glVertex2i(iw, ih);
	glTexCoord2i(0, 1); glVertex2i(0, ih);
	glEnd();
	glPopMatrix();

	orthogonalEnd();
	glEnable(GL_DEPTH_TEST);
	*/
	glEnable(GL_TEXTURE_2D);
	glDisable(GL_DEPTH_TEST);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	

	glOrtho(0, 800, 600, 0, -1.0, 1.0);

	glRasterPos2d(0, 600);
	cv::Mat resized;
	resize(cameraImage, resized, Size(800, 600));
	cv::flip(resized, resized, 0); // openni/xtion
	glDrawPixels(resized.cols, resized.rows, GL_BGR_EXT, GL_UNSIGNED_BYTE, resized.data); //para openni
	glEnable(GL_DEPTH_TEST);
	glDisable(GL_TEXTURE_2D);
}

void display(){

	int key = 0;

	total_video_frames = capture.get(CV_CAP_PROP_FRAME_COUNT);

	while (key != 27 && !LOAD_ERROR){

		capture >> cameraImage;

		if (cameraImage.data){
			cvtColor(cameraImage, cameraColorTemp, CV_BGR2RGB);
			texture = gluBuild2DMipmaps(GL_TEXTURE_2D, GL_RGB, cameraColorTemp.cols, cameraColorTemp.rows,
				GL_RGB, GL_UNSIGNED_BYTE, cameraColorTemp.data);

			if (RESIZE_CAMERA_RESOLUTION)
				resize(cameraImage, cameraImage, Size(cameraImage.cols / RESOLUTION_PARAM, cameraImage.rows / RESOLUTION_PARAM));

			intrinsic_Matrix.at<double>(0, 0) = cameraImage.cols;		intrinsic_Matrix.at<double>(0, 1) = 0;					intrinsic_Matrix.at<double>(0, 2) = cameraImage.cols / 2;	//intrinsic_Matrix.at<double>(0, 3) = 0;
			intrinsic_Matrix.at<double>(1, 0) = 0;						intrinsic_Matrix.at<double>(1, 1) = cameraImage.rows;	intrinsic_Matrix.at<double>(1, 2) = cameraImage.rows / 2;	//intrinsic_Matrix.at<double>(1, 3) = 0;
			intrinsic_Matrix.at<double>(2, 0) = 0;						intrinsic_Matrix.at<double>(2, 1) = 0;					intrinsic_Matrix.at<double>(2, 2) = 1;						//intrinsic_Matrix.at<double>(2, 3) = 0;
			//intrinsic_Matrix.at<double>(3, 0) = 0;						intrinsic_Matrix.at<double>(3, 1) = 0;					intrinsic_Matrix.at<double>(3, 2) = 0;					  intrinsic_Matrix.at<double>(3, 3) = 1;


			cvtColor(cameraImage, cameraGrayImage, CV_BGR2GRAY);

			if (EQUALIZE_CAMERA_HISTOGRAM){
				equalizeHist(cameraGrayImage, cameraGrayTemp);
				cameraGrayTemp.copyTo(cameraGrayImage);
			}

			if (GAUSSIAN_BLUR_CAMERA){
				GaussianBlur(cameraGrayImage, cameraGrayTemp, Size(GAUSSIAN_SIZE, GAUSSIAN_SIZE), 0);
				cameraGrayTemp.copyTo(cameraGrayImage);
			}

			if (featureDetectorAlgorithm == SIFT_ALG){
				siftDetector.detect(cameraGrayImage, cameraKeypoints);
				siftExtractor.compute(cameraGrayImage, cameraKeypoints, descriptor2);
			}
			else if (featureDetectorAlgorithm == SURF_ALG){
				surfDetector->detect(cameraGrayImage, cameraKeypoints);
				surfExtractor.compute(cameraGrayImage, cameraKeypoints, descriptor2);
			}
			else if (featureDetectorAlgorithm == ORB_ALG){
				orbDetector.detect(cameraGrayImage, cameraKeypoints);
				orbExtractor.compute(cameraGrayImage, cameraKeypoints, descriptor2);
			}

			if (EQUALIZE_CAMERA_HISTOGRAM || GAUSSIAN_BLUR_CAMERA || MEDIAN_BLUR_CAMERA)
				imshow("Remastered Camera", cameraGrayImage);

			if (EQUALIZE_TEXTURE_HISTOGRAM || GAUSSIAN_BLUR_TEXTURE || MEDIAN_BLUR_TEXTURE)
				imshow("Remastered Texture", originalGrayTexture);

			features2Dcamera = keypoint2point2d(cameraKeypoints);
			features3Dcamera = keypoint2point3d(cameraKeypoints, 1);

			matcher.clear();
			matches.clear();
			goodMatches.clear();

			matcher.match(descriptor2, descriptor1, matches);

			/*
			//sem desafio
			Mat img_matches;
			drawMatches(cameraImage, cameraKeypoints, originalColorTexture, imageKeypoints,
			matches, img_matches, Scalar::all(-1), Scalar::all(-1),
			vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
			*/

			//*
			//Calculating the min distances between keypoints
			//distance is the score of similarity between the two descriptors of a match.
			//This way we improve the feature detection of the algorithm
			double min_dist = 100;
			for (int i = 0; i < matches.size(); i++)
			{
				double dist = matches[i].distance;
				if (dist < min_dist) min_dist = dist;
			}
			//end

			for (int i = 0; i < matches.size(); i++)
			{
				if (matches[i].distance <= max(2 * min_dist, 0.02))
				{
					goodMatches.push_back(matches[i]);
				}
			}

			drawMatches(cameraImage, cameraKeypoints, originalColorTexture, imageKeypoints,
				goodMatches, img_matches, Scalar::all(-1), Scalar::all(-1),
				vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
			//*/

			if (featureDetectorAlgorithm == SIFT_ALG && poseEstimationAlgorithm == RANSAC_ALG){
				putText(img_matches, "SIFT + RANSAC", Point(0, cameraImage.rows + 15), FONT_HERSHEY_PLAIN, 1, Scalar(0, 0, 255));
			}
			else if (featureDetectorAlgorithm == SIFT_ALG && poseEstimationAlgorithm == LMEDS_ALG){
				putText(img_matches, "SIFT + LMEDS", Point(0, cameraImage.rows + 15), FONT_HERSHEY_PLAIN, 1, Scalar(0, 0, 255));
			}
			else if (featureDetectorAlgorithm == SIFT_ALG && poseEstimationAlgorithm == REGULAR_ALG){
				putText(img_matches, "SIFT + REGULAR", Point(0, cameraImage.rows + 15), FONT_HERSHEY_PLAIN, 1, Scalar(0, 0, 255));
			}
			else if (featureDetectorAlgorithm == SURF_ALG && poseEstimationAlgorithm == RANSAC_ALG){
				putText(img_matches, "SURF + RANSAC", Point(0, cameraImage.rows + 15), FONT_HERSHEY_PLAIN, 1, Scalar(0, 0, 255));
			}
			else if (featureDetectorAlgorithm == SURF_ALG && poseEstimationAlgorithm == LMEDS_ALG){
				putText(img_matches, "SURF + LMEDS", Point(0, cameraImage.rows + 15), FONT_HERSHEY_PLAIN, 1, Scalar(0, 0, 255));
			}
			else if (featureDetectorAlgorithm == SURF_ALG && poseEstimationAlgorithm == REGULAR_ALG){
				putText(img_matches, "SURF + REGULAR", Point(0, cameraImage.rows + 15), FONT_HERSHEY_PLAIN, 1, Scalar(0, 0, 255));
			}
			else if (featureDetectorAlgorithm == ORB_ALG){
				putText(img_matches, "ORB", Point(0, cameraImage.rows + 15), FONT_HERSHEY_PLAIN, 1, Scalar(0, 0, 255));
			}

			putText(img_matches, "KeyPoints Texture:   " + std::to_string(imageKeypoints.size()), Point(0, cameraImage.rows + 30), FONT_HERSHEY_PLAIN, 1, Scalar(0, 0, 255));
			putText(img_matches, "KeyPoints Camera:    " + std::to_string(cameraKeypoints.size()), Point(0, cameraImage.rows + 45), FONT_HERSHEY_PLAIN, 1, Scalar(0, 0, 255));
			putText(img_matches, "Matches   Keypoints: " + std::to_string(goodMatches.size()), Point(0, cameraImage.rows + 60), FONT_HERSHEY_PLAIN, 1, Scalar(0, 0, 255));
			putText(img_matches, "Running  Time:        " + std::to_string(nowtime - startTime) + " seconds", Point(0, cameraImage.rows + 105), FONT_HERSHEY_PLAIN, 1, Scalar(0, 0, 255));
			putText(img_matches, "FPS: " + std::to_string(fps), Point(0, cameraImage.rows + 120), FONT_HERSHEY_PLAIN, 1, Scalar(0, 0, 255));
			putText(img_matches, "FPS: " + std::to_string(fps), Point(0, 10), FONT_HERSHEY_PLAIN, 1, Scalar(0, 0, 0));
			//==============
			//calculating pose
			if (goodMatches.size() >= 4){
				obj.clear();
				scene.clear();

				for (int i = 0; i < goodMatches.size(); i++)
				{
					//-- Get the keypoints from the good matches
					scene.push_back(cameraKeypoints[goodMatches[i].queryIdx].pt);
					obj.push_back(imageKeypoints[goodMatches[i].trainIdx].pt);
				}


				/*param
				*0 - a regular method using all the points
				*CV_RANSAC - RANSAC - based robust method
				*CV_LMEDS - Least - Median robust method
				*/
				if (poseEstimationAlgorithm == RANSAC_ALG)
					homography = findHomography(obj, scene, CV_RANSAC);
				else if (poseEstimationAlgorithm == LMEDS_ALG)
					homography = findHomography(obj, scene, CV_LMEDS);
				else if (poseEstimationAlgorithm == REGULAR_ALG)
					homography = findHomography(obj, scene, 0);

				obj_corners[0] = cvPoint(0, 0);
				obj_corners[1] = cvPoint(originalGrayTexture.cols, 0);
				obj_corners[2] = cvPoint(originalGrayTexture.cols, originalGrayTexture.rows);
				obj_corners[3] = cvPoint(0, originalGrayTexture.rows);

				perspectiveTransform(obj_corners, scene_corners, homography);

				//-- Draw lines between the corners (the mapped object in the scene - image_2 )
				line(img_matches, scene_corners[0], scene_corners[1], Scalar(0, 255, 0), 4);
				line(img_matches, scene_corners[1], scene_corners[2], Scalar(0, 255, 0), 4);
				line(img_matches, scene_corners[2], scene_corners[3], Scalar(0, 255, 0), 4);
				line(img_matches, scene_corners[3], scene_corners[0], Scalar(0, 255, 0), 4);
				//==============

				//====pose

				//vector<Point3f> objectPoints = { Point3f(-1, 1, 0), Point3f(1, 1, 0), Point3f(1, -1, 0), Point3f(-1, -1, 0) };
				//Mat objectPointsMat(objectPoints);
				Point3d aux;
				vector<Point3d> temp;
				aux.z = 1;
				for (int i = 0; i < scene.size(); i++){
					aux.x = scene[i].x;
					aux.y = scene[i].y;
					temp.push_back(aux);
				}

				solvePnP(temp, obj, intrinsic_Matrix, distortion_coeffs, rvec, tvec);

				Rodrigues(rvec, rotation);
				//= == == == =

			}
			//calculate FPS, CPU usage and Memory usage
			atualCPU = getCurrentProcessorUsage();

			calc_fps += 1;
			nowtime = std::time(nullptr);
			if (nowtime >= one_second + 1){
				one_second = std::time(nullptr);
				fps = calc_fps;
				calc_fps = 0;

				GetProcessMemoryInfo(thisProcess, &testa, sizeof(testa));
				if (!memoryUsage.empty()){
					lastMemoryUsage = memoryUsage[memoryUsage.size() - 1];
					if (lastMemoryUsage != testa.WorkingSetSize){
						memoryUsage.push_back(testa.WorkingSetSize);
						myMemoryFile << testa.WorkingSetSize << " bytes at time: " + to_string(nowtime - startTime) + " sec" << endl;
					}
				}
				else
					memoryUsage.push_back(testa.WorkingSetSize);

				if (!cpuUsage.empty()){
					lastCPUstored = cpuUsage[cpuUsage.size() - 1];
					if (lastCPUstored != atualCPU){
						cpuUsage.push_back(atualCPU);
						myProcessorFile << atualCPU << "% at time " << nowtime - startTime << " sec" << endl;
					}
				}
				else
					cpuUsage.push_back(atualCPU);

			}
			//end
			putText(img_matches, "Memory    Usage:      " + std::to_string(testa.WorkingSetSize) + " bytes", Point(0, cameraImage.rows + 75), FONT_HERSHEY_PLAIN, 1, Scalar(0, 0, 255));

			if (atualCPU > 0)
				putText(img_matches, "Processor Usage:      " + std::to_string(atualCPU) + " percent", Point(0, cameraImage.rows + 90), FONT_HERSHEY_PLAIN, 1, Scalar(0, 0, 255));

			//Mat temp;
			//drawKeypoints(cameraImage, cameraKeypoints, temp, 255, DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
			//imshow("Orientation", temp);

			//if (key == 'q' || key == 'Q'){
			//	setWindowProperty(window1, CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
			//	setWindowProperty(window1, CV_WND_PROP_ASPECTRATIO, CV_WINDOW_KEEPRATIO);
			//}
			imshow(window1, img_matches);

		}
		else{
			LOAD_ERROR = true;
		}

		double aux[9], temp[4][4];
		for (int i = 0; i < 3; i++){
			for (int j = 0; j < 3; j++){
				aux[i + j] = intrinsic_Matrix.at<double>(j, i);
			}
		}

		//temp[0][0]  = rotation.at<double>(0, 0);  temp[0][1] = rotation.at<double>(0, 1);  temp[0][2] = rotation.at<double>(0, 2);	temp[0][3] = tvec.at<double>(0, 0);
		//temp[1][0]  = rotation.at<double>(1, 0);  temp[1][1] = rotation.at<double>(1, 1);  temp[1][2] = rotation.at<double>(1, 2);	temp[1][3] = tvec.at<double>(0, 1);
		//temp[2][0]  = rotation.at<double>(2, 0);  temp[2][1] = rotation.at<double>(2, 1);  temp[2][2] = rotation.at<double>(2, 2);	temp[2][3] = tvec.at<double>(0, 2);
		//temp[3][0]  = 0;							temp[3][1] = 0;						     temp[3][2] = 0;							temp[3][3] = 1;

		if (rotation.dims != 0){
			temp[0][0] = rotation.at<double>(0, 0);  temp[0][1] = rotation.at<double>(1, 0);  temp[0][2] = rotation.at<double>(2, 0);	temp[0][3] = 0;
			temp[1][0] = rotation.at<double>(0, 1);  temp[1][1] = rotation.at<double>(1, 1);  temp[1][2] = rotation.at<double>(2, 1);	temp[1][3] = 0;
			temp[2][0] = rotation.at<double>(0, 2);  temp[2][1] = rotation.at<double>(1, 2);  temp[2][2] = rotation.at<double>(2, 2);	temp[2][3] = 0;
			temp[3][0] = 0;							 temp[3][1] = 0;						  temp[3][2] = 0;							temp[3][3] = 1;
			//temp[3][0] = tvec.at<double>(0, 0);		 temp[3][1] = tvec.at<double>(1, 0);	  temp[3][2] = tvec.at<double>(2, 0);		temp[3][3] = 1;

			//cout << tvec.rows << " x " << tvec.cols << endl;
			//cout << tvec << endl << endl;;

			glClearColor(0.0, 0.0, 0.0, 0.0);
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

			background();

			glMatrixMode(GL_PROJECTION);
			glLoadIdentity();
			//glMultMatrixd(aux);
			//glFrustum(-(800 / 2) / 600, (800 / 2) / 600, -1, 1, 2.5, 500);
			gluPerspective(45, (800 / 2) / 600, 2.5, 500);
			glMatrixMode(GL_MODELVIEW);
			glLoadIdentity();
			glLoadMatrixd(temp[0]);

			glScalef(0.5, 0.5, 0.5);

			glBegin(GL_QUADS);
			glColor3f(1.0, 0.0, 0.0);     glVertex3f(0.5, -0.5, -0.5);      // P1 é vermelho
			glColor3f(0.0, 1.0, 0.0);     glVertex3f(0.5, 0.5, -0.5);      // P2 é verde
			glColor3f(0.0, 0.0, 1.0);     glVertex3f(-0.5, 0.5, -0.5);      // P3 é azul
			glColor3f(1.0, 0.0, 1.0);     glVertex3f(-0.5, -0.5, -0.5);      // P4 é roxo

			// Lado branco - TRASEIRA
			glBegin(GL_POLYGON);
			glColor3f(1.0, 1.0, 1.0);
			glVertex3f(0.5, -0.5, 0.5);
			glVertex3f(0.5, 0.5, 0.5);
			glVertex3f(-0.5, 0.5, 0.5);
			glVertex3f(-0.5, -0.5, 0.5);
			glEnd();

			// Lado roxo - DIREITA
			glBegin(GL_POLYGON);
			glColor3f(1.0, 0.0, 1.0);
			glVertex3f(0.5, -0.5, -0.5);
			glVertex3f(0.5, 0.5, -0.5);
			glVertex3f(0.5, 0.5, 0.5);
			glVertex3f(0.5, -0.5, 0.5);
			glEnd();

			// Lado verde - ESQUERDA
			glBegin(GL_POLYGON);
			glColor3f(0.0, 1.0, 0.0);
			glVertex3f(-0.5, -0.5, 0.5);
			glVertex3f(-0.5, 0.5, 0.5);
			glVertex3f(-0.5, 0.5, -0.5);
			glVertex3f(-0.5, -0.5, -0.5);
			glEnd();

			// Lado azul - TOPO
			glBegin(GL_POLYGON);
			glColor3f(0.0, 0.0, 1.0);
			glVertex3f(0.5, 0.5, 0.5);
			glVertex3f(0.5, 0.5, -0.5);
			glVertex3f(-0.5, 0.5, -0.5);
			glVertex3f(-0.5, 0.5, 0.5);
			glEnd();

			// Lado vermelho - BASE
			glBegin(GL_POLYGON);
			glColor3f(1.0, 0.0, 0.0);
			glVertex3f(0.5, -0.5, -0.5);
			glVertex3f(0.5, -0.5, 0.5);
			glVertex3f(-0.5, -0.5, 0.5);
			glVertex3f(-0.5, -0.5, -0.5);
			glEnd();
			//gluLookAt(0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
			//glTranslatef(0, 0, 10);
			glEnd();
		}


		glutSwapBuffers();

		if (total_video_frames == capture.get(CV_CAP_PROP_POS_FRAMES)){
			capture.open(video_filepath);
		}

		key = waitKey(30);
	}

	if (key == 27)
		exit(0);
	else if (key == 'q' || key == 'Q')
		glutFullScreen();
	else if (key == 'e' || key == 'E')
		glutReshapeWindow(800, 600);
}

GLuint LoadTexture(){
	unsigned char data[] = { 255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255 };

	glGenTextures(1, &texture);
	glBindTexture(GL_TEXTURE_2D, texture);
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
	glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);

	//even better quality, but this will do for now.
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	//to the edge of our shape. 
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

	//Generate the texture
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 2, 2, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
	return texture; //return whether it was successful
}

void init(){
	initCPU();

	surfDetector = new SurfFeatureDetector(1500);

	thisProcess = OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, FALSE, GetCurrentProcessId());

	capture.open(video_filepath);
	//capture.open(0); //open webcam
}

void initCV(){
	//MODULARIZAR funcaoLêImageAndRetiraFeatures(Mat salvaImagemCinza, Mat salvaImagemColorida, Mat descritorDaImagem, 
	//											vector<keypoint> salvaKeypointsDaImagem, string filepath)
	originalGrayTexture = imread(filepath, CV_LOAD_IMAGE_GRAYSCALE);
	originalColorTexture = imread(filepath, CV_LOAD_IMAGE_ANYCOLOR | CV_LOAD_IMAGE_ANYDEPTH);

	if (originalGrayTexture.empty() || originalColorTexture.empty()){
		cout << "Image Not Loaded" << endl;
		LOAD_ERROR = true;
	}

	if (EQUALIZE_TEXTURE_HISTOGRAM){
		equalizeHist(originalGrayTexture, textureGrayTemp);
		textureGrayTemp.copyTo(originalGrayTexture);
	}

	if (GAUSSIAN_BLUR_TEXTURE){
		GaussianBlur(originalGrayTexture, textureGrayTemp, Size(GAUSSIAN_SIZE, GAUSSIAN_SIZE), 0);
		textureGrayTemp.copyTo(originalGrayTexture);
	}

		if (featureDetectorAlgorithm == SIFT_ALG){
		siftDetector.detect(originalGrayTexture, imageKeypoints);
		siftExtractor.compute(originalGrayTexture, imageKeypoints, descriptor1);

	}
	else if (featureDetectorAlgorithm == SURF_ALG){
		surfDetector->detect(originalGrayTexture, imageKeypoints);
		surfExtractor.compute(originalGrayTexture, imageKeypoints, descriptor1);
	}
	else if (featureDetectorAlgorithm == ORB_ALG){
		orbDetector.detect(originalGrayTexture, imageKeypoints);
		orbExtractor.compute(originalGrayTexture, imageKeypoints, descriptor1);
	}

	//loadImageFeatures(originalGrayTexture, originalGrayTexture, descriptor1, imageKeypoints, filepath, siftDetector, siftExtractor);
	//end
}

void handleKeyboard(GLubyte key, GLint x, GLint y)
{

}

void initGL(){
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
	glEnable(GL_DEPTH_TEST);
	glShadeModel(GL_SMOOTH);
	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);

	glutInitWindowSize(800, 600);
	glutCreateWindow("Aspect Ratio");

	glutDisplayFunc(display);
	glutKeyboardFunc(handleKeyboard);

	texture = LoadTexture();
	glutMainLoop();
}

void initFile(){
	if (featureDetectorAlgorithm == SIFT_ALG && poseEstimationAlgorithm == RANSAC_ALG)
	{
		myMemoryFile.open(sift_ransac_memory_txt);
		myProcessorFile.open(sift_ransac_processor_txt);
	}
	else if (featureDetectorAlgorithm == SIFT_ALG && poseEstimationAlgorithm == LMEDS_ALG)
	{
		myMemoryFile.open(sift_lmeds_memory_txt);
		myProcessorFile.open(sift_lmeds_processor_txt);
	}
	else if (featureDetectorAlgorithm == SIFT_ALG && poseEstimationAlgorithm == REGULAR_ALG)
	{
		myMemoryFile.open(sift_regular_memory_txt);
		myProcessorFile.open(sift_regular_processor_txt);
	}
	else if (featureDetectorAlgorithm == SURF_ALG && poseEstimationAlgorithm == RANSAC_ALG)
	{
		myMemoryFile.open(surf_ransac_memory_txt);
		myProcessorFile.open(surf_ransac_processor_txt);
	}
	else if (featureDetectorAlgorithm == SURF_ALG && poseEstimationAlgorithm == LMEDS_ALG)
	{
		myMemoryFile.open(surf_lmeds_memory_txt);
		myProcessorFile.open(surf_lmeds_processor_txt);
	}
	else if (featureDetectorAlgorithm == SURF_ALG && poseEstimationAlgorithm == REGULAR_ALG)
	{
		myMemoryFile.open(surf_regular_memory_txt);
		myProcessorFile.open(surf_regular_processor_txt);
	}

	myMemoryFile << "Memory Usage" << endl;
	myProcessorFile << "Processor Usage" << endl;
}

void endFile(){
	double memoryMedian = 0;
	for (int i = 0; i < memoryUsage.size(); i++){
		memoryMedian += memoryUsage[i] / memoryUsage.size();
	}
	myMemoryFile << endl << "Median (bytes) = " << memoryMedian << endl;
	myMemoryFile << "Median (MB) = " << (memoryMedian / DIV) << endl;
	myMemoryFile << "Running Time = " << nowtime - startTime << " seconds" << endl;
	myMemoryFile.close();


	double processorMedian = 0;

	for (int i = 0; i < cpuUsage.size(); i++){
		processorMedian += cpuUsage[i] / cpuUsage.size();
	}
	//processorMedian *= 100;

	myProcessorFile << endl << "Median = " << processorMedian << "%" << endl;
	myProcessorFile << "Running Time = " << nowtime - startTime << " seconds" << endl;
	myProcessorFile.close();
}

int main(){
	cout << "****************************************************" << endl;
	cout << "* Texture Detector - Version 1.0 - Created by EHMR *" << endl;
	cout << "* Contact: ehmr@cin.ufpe.br                        *" << endl;
	cout << "* About me: www.cin.ufpe.br/~ehmr                  *" << endl;
	cout << "* Last Updated: 7/28/2015                          *" << endl;
	cout << "*                                                  *" << endl;
	cout << "****************************************************\n\n\n" << endl;

	init();

	initCV();

	if (GENERATE_RELATORY)
		initFile();

	features3Dtexture = keypoint2point3d(imageKeypoints, 1);
	features2Dtexture = keypoint2point2d(imageKeypoints);


	String aux_filename = filepath.substr(filepath.find_last_of("/")+1, filepath.size());
	for (int i = aux_filename.size() - 1; i >= 0; i--){
		if (aux_filename.at(i) == '.'){
			aux_filename = aux_filename.substr(0, i);
			i = -1;
		}
	}

	createPngWithFeatures(originalGrayTexture, imageKeypoints, 255, "Resources/FeaturesPng", aux_filename+"_sift_extract_features_grey.png", true, "Grey Image With Features");
	createPngWithFeatures(originalColorTexture, imageKeypoints, -1, "Resources/FeaturesPng", aux_filename+"_sift_extract_features_color.png", true, "Color Image With Features");


	//namedWindow(window1, WINDOW_NORMAL);
	//resizeWindow(window1, 727*2, 1000);
	startTime = time(nullptr);

	initGL();

	if (GENERATE_RELATORY)
		endFile();

	return 0;
}