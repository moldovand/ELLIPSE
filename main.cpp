/********************************************************************************
*
*
*  This program is demonstration for ellipse fitting. Program finds
*  contours and approximate it by ellipses.
*
*  Trackbar specify threshold parametr.
*
*  White lines is contours. Red lines is fitting ellipses.
*
*
*  Autor:  Denis Burenkov.
*
*
*
********************************************************************************/
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include <opencv2/core/core.hpp>
#include <iostream>
#include <stdio.h>
using namespace cv;
using namespace std;

// detect ellipsoidal dials in the image
void manometerDetect(Mat& image, Mat& colorimage);

void ellipseHoughCircle(Mat& src_gray, Mat src);

void rotateResize(Mat& src, Mat& pointsContour, Mat& dst);

int main( int argc, char** argv )
{
    // image as intensity
    Mat image;
    // image in BGR format
    Mat colorimage;
    const char* filename = argc == 2 ? argv[1] : (char*)"stuff.jpg";
    // load the image as intensity
    image = imread(filename, 0);

    colorimage = imread(filename, 1);
    if( image.empty() ){
        cout << "Couldn't open image " << filename << "\nUsage: fitellipse <image_name>\n";
        return 0;
    }

    manometerDetect(image, colorimage);

//    ellipseHoughCircle(image, colorimage);

    // Wait for a key stroke
    waitKey();
    return 0;
}

void rotateResize(Mat& src, Mat& pointsContour, Mat& dst)
{
    RotatedRect box;
    Mat M, rotated;
    float angle;
    Size rect_size;

    // fitting an ellipse to the points that make the contour
    box = fitEllipse(pointsContour);
    angle = box.angle;
    rect_size = box.size;
    // get the rotation matrix
    M = getRotationMatrix2D(box.center, angle, 1.0);
    // perform the affine transformation
    warpAffine(src, rotated, M, src.size(), INTER_CUBIC);
    imwrite("rotated.jpg",rotated);
    // crop the resulting image
    getRectSubPix(rotated, rect_size, box.center, dst);
    resize(dst,dst,Size(300,300),0,0,1);
    imwrite("M.jpg",dst);

}


void manometerDetect(Mat& image, Mat& colorimage)
{

    Mat drawImage;
    Mat image_canny;
    Mat circleManom;

    RotatedRect box;

    double contourEndDist;

    int thresh = 100;
    // drawImage is an intensity image
    image.copyTo(drawImage);

    vector<Vec4i> hierarchy;

    vector<vector<Point> > contours;
    vector<KeyPoint> keypoints;

    // Setup SimpleBlobDetector parameters.
    SimpleBlobDetector::Params params;
    // Change thresholds
    params.minThreshold = 10;
    params.maxThreshold = 1000;

//    // Filter by Area.
//    params.filterByArea = false;
//    params.minArea = 1500;

//    // Filter by Circularity
    params.filterByCircularity = true;
    params.minCircularity = 0.7;

//    // Filter by Convexity
    params.filterByConvexity = true;
    params.minConvexity = 0.87;

//    // Filter by Inertia
//    params.filterByInertia = false;
//    params.minInertiaRatio = 0.01;

//    // Set up detector with params
    SimpleBlobDetector detector(params);

    detector.detect(image, keypoints);

    // define the size of region of interest (ara that contains the ellipse)
    int w = 200, h=200;
    Rect region_of_interest;

    for (size_t i = 0; i < keypoints.size(); ++i){

        cout << "New Keypoint = " << i << endl;
        region_of_interest = Rect(keypoints[i].pt.x-w/2, keypoints[i].pt.y-h/2, w, h);

        // extract the ROI that has blob's coordinates as its center

        if(region_of_interest.x<=0 || region_of_interest.y<=0 || region_of_interest.x+w>drawImage.cols || region_of_interest.y+h>drawImage.rows){
            cout << "region_of_interest = " << i << ";  too close to image border; skipping it." << endl;
            continue;
        }
        Mat image_roi(drawImage,region_of_interest);
        imwrite("resultROI.png", image_roi);

        // the center of the region_of_interest
        Point2f shiftP;
        shiftP.x=keypoints[i].pt.x-h/2;
        shiftP.y=keypoints[i].pt.y-w/2;

        // Detect edges using canny
        Canny( image_roi, image_canny, thresh, thresh*2, 3 );
        imwrite("resultROI_Canny.png", image_canny);

        // Find contours
        findContours(image_canny, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE );

        // Fit ellipses to detected contours
        for(size_t j = 0; j < contours.size(); ++j){
            // we are interested only in big sizes of the contour
            size_t count = contours[j].size();
            if ( count < 100 ){
                cout << "j = " << j << "; small contour; count = " << count << endl;
                continue;
            }

            // extract the points of the contour
            Mat pointsf;
            //TODO calculate convex hull area
//            vector<Point2f> convexHull;
//            vector<Point2f> contour;  // Convex hull contour points
//            double epsilon = 0.001; // Contour approximation accuracy;
            Mat(contours[j]).convertTo(pointsf, CV_32F);

            // Calculate convex hull of original points (which points positioned on the boundary)
//            convexHull(pointsf,convexHull,false);

            // fitting an ellipse to the points that make the contour
            box = fitEllipse(pointsf);
            double area = contourArea(contours[j]);
            cout << "j = " << j << " area =" << area << endl;

            // if the contour is too elongated skip it
            if( MAX(box.size.width, box.size.height) > MIN(box.size.width, box.size.height)*2 )
                continue;

            Mat mask, out;
            mask = Mat::zeros(image_roi.size(), image_roi.type());
            out = Mat::zeros(image_roi.size(), image_roi.type());

            if(area > 500.0){
                // save the contour points
//                FILE *temp = fopen("contoursData.dat", "w");
//                for(size_t p = 0; p < count; p++)
//                    fprintf(temp,"%d %d \n", contours[j][p].x, contours[j][p].y);
//                fclose(temp);

                // draw ellipse contour on the initial big image
                drawContours(colorimage, contours, j, cv::Scalar(0,255,0), 1, 8, hierarchy, 0, region_of_interest.tl()); // filled (green)
                imwrite("result_Contour.jpg", colorimage);

                // generate a filled in area on the region of interest
                drawContours(mask, contours, j, 255, -1, 8, hierarchy, 0);
                // extract the ellipse (by masking) from the region of interest
                image_roi.copyTo(out, mask);
                // rotate the region of interest with extracted ellipse; crop and resize.
                rotateResize(out,pointsf,circleManom);

                cout << "j = " << j << ";  count = " << count << endl;
            }
        }
    }
}


void ellipseHoughCircle(Mat& src_gray, Mat src)
{
    /// Reduce the noise so we avoid false circle detection
    GaussianBlur( src_gray, src_gray, Size(9, 9), 2, 2 );

    vector<Vec3f> circles;

    /// Apply the Hough Transform to find the circles
    HoughCircles( src_gray, circles, CV_HOUGH_GRADIENT, 1, 10, 200, 20, 0, 0 );

    /// Draw the circles detected
    for( size_t i = 0; i < circles.size(); i++ )
    {
        Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
        int radius = cvRound(circles[i][2]);
        // circle center
        circle( src, center, 3, Scalar(0,255,0), -1, 8, 0 );
        // circle outline
        circle( src, center, radius, Scalar(0,0,255), 3, 8, 0 );
     }

    /// Show your results
    namedWindow( "Hough Circle Transform Demo", CV_WINDOW_AUTOSIZE );
    imshow( "Hough Circle Transform Demo", src );

}
