#**Features Extractor**

## Synopsis

This project was fully written in C++11 early 2015, using the [OpenCV](opencv.org) and [OpenGL](www.opengl.org) libraries to the Computer Graphics course at CIn UFPE. It is a computer vision algorithm that extract features of a given texture, find the texture as a surface on the webcam image, calculate the pose of the detected surface and project a 3D object on the surface using the texture as a AR marker.

## About the project

To create this application i used the SIFT algorithm to make the feature matching and find the respective surface on the webcam image. After that i used others techniques to calculate the surface pose and then the algorithm use this informations to create a 3D object using [OpenGL](www.opengl.org) and render the object with the respective transformations.

## Installation

You only need to install the [Visual Studio IDE](www.visualstudio.com) and open the solution.

## Contact
eduardohmrodrigues@gmail.com or ehmr@cin.ufpe.br