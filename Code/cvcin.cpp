#include "cvcin.h"


cvcin::cvcin()
{
}


cvcin::~cvcin()
{
}

bool cvcin::loadImageFeatures(Mat grey, Mat color, Mat descriptor, vector<KeyPoint> keypoints, string filepath, SiftFeatureDetector detector, SiftDescriptorExtractor extractor){
	bool error = false;

	grey = imread(filepath, CV_LOAD_IMAGE_GRAYSCALE);
	color = imread(filepath, CV_LOAD_IMAGE_ANYCOLOR | CV_LOAD_IMAGE_ANYDEPTH);

	if (grey.empty() || color.empty()){
		cout << "Image Not Loaded" << endl;
		error = true;
	}
	else{
		detector.detect(grey, keypoints);
		extractor.compute(grey, keypoints, descriptor);
	}

	return error;
}

void cvcin::keypoints2points3D(vector<KeyPoint> keypoints, vector<Point3d> points, double z){
	Point3d aux;
	aux.z = z;
	for (int i = 0; i < keypoints.size(); i++){
		aux.x = keypoints[i].pt.x;
		aux.y = keypoints[i].pt.y;
		points.push_back(aux);
		cout << aux << endl;
	}
}

void cvcin::keypoint2point2d(vector<KeyPoint> keypoints, vector<Point2d> points){
	Point2d aux;
	for (int i = 0; i < keypoints.size(); i++){
		aux.x = keypoints[i].pt.x;
		aux.y = keypoints[i].pt.y;
		points.push_back(aux);
	}
}

/*
void cvcin::extractSiftFeatures2PNG(Mat input, vector<KeyPoint> keypoints, imageColor imagecolor, int featureColor){
	Mat sift_extract_feature;

	if (imagecolor == imageColor::CV_CIN_GRAY){
		drawKeypoints(input, keypoints, sift_extract_feature, featureColor, DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
		//imshow("Grey image with features", sift_extract_feature);
		imwrite("Resources/sift_extract_features_grey.png", sift_extract_feature);

	} else if (imagecolor == imageColor::CV_CIN_COLOR){
		drawKeypoints(input, keypoints, sift_extract_feature, featureColor, DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
		//imshow("Grey image with features", sift_extract_feature);
		imwrite("Resources/sift_extract_features_color.png", sift_extract_feature);
	}
}
*/
