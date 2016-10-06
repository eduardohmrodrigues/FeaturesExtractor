#pragma once
#include "Main.h"
#include "util.h"
#include <math.h>
#include <Windows.h>
#include <math.h>

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

using namespace cv;

enum imageColor{
	CV_CIN_GRAY = 0,
	CV_CIN_COLOR = 1
};

class cvcin
{
public:
	cvcin();
	~cvcin();

	bool loadImageFeatures(Mat grey, Mat color, Mat descriptor, vector<KeyPoint> keypoints, string filepath, SiftFeatureDetector detector, SiftDescriptorExtractor extractor);

	/**
	*This function convert a vector of keypoints in a vector of 3D point
	*@param keypoints input vector of keypoints
	*@param points output vector of 3D points
	*@param z value of z coordinate
	*/
	static void keypoints2points3D(vector<KeyPoint> input, vector<Point3d> points, double z);


	/**
	*This function convert a vector of keypoints in a vector of 2D point
	*@param keypoints input vector of keypoints
	*@param points output vector of 2D points
	*/
	static void keypoint2point2d(vector<KeyPoint> keypoints, vector<Point2d> points);


	//static void extractSiftFeatures2PNG(Mat input, vector<KeyPoint> keypoints, imageColor imageColor, int featureColor);
};

