#include <fstream>                                                                                                                                                                                      
#include <iostream>
#include <opencv2/opencv.hpp>
#include "estimate_pose_dlt.hpp"
#include <mgl2/mgl.h>
#include <mgl2/glut.h>
#include <unistd.h>

cv::Point2f reprojectPoints(cv::Point3f& P, cv::Mat& M, cv::Mat& k);

int main( int argc, char** argv )
{
    if ( argc != 2 )
    {
        std::cout << "usage: use path_to_dataset" << std::endl;
        return 1;
    }
    std::string path_to_dataset = argv[1];
    // 设置内参矩阵
    std::string k_file = path_to_dataset + "/K.txt";
    std::ifstream kin(k_file);
    if(!kin) std::cout<<"error"<<std::endl;
    std::string k11, k12, k13, k21, k22, k23, k31, k32, k33;
    kin >> k11 >> k12 >> k13 >> k21 >> k22 >> k23 >> k31 >> k32 >> k33;
    cv::Mat k = (cv::Mat_<float> (3,3) <<
                 std::stof(k11), std::stof(k12), std::stof(k13),
                 std::stof(k21), std::stof(k22), std::stof(k23),
                 std::stof(k31), std::stof(k32), std::stof(k33));

    // 获取角点的3D坐标
    std::string corners_file = path_to_dataset + "/p_W_corners.txt";
    std::string cx, cy, cz;
    std::ifstream cin(corners_file);
    std::vector<cv::Point3f> corners;
    for (int i = 0; i<12; i++) {
        cin >> cx >> cy >> cz;
        corners.push_back(cv::Point3f(std::stof(cx), std::stof(cy), std::stof(cz)));
    }

    // 设置2D点文件搜索路径
    std::string points_file = path_to_dataset + "/detected_corners.txt";
    std::string p11, p12, p21, p22, p31, p32, p41, p42, p51, p52, p61, p62, p71, p72, p81, p82, p91, p92, p101, p102, p111, p112, p121, p122;
    std::ifstream pin(points_file);
    std::vector< std::vector<cv::Point2f> > points_sets;
    for (int point_index = 0; point_index < 200; point_index++) {
        pin>>p11>>p12>>p21>>p22>>p31>>p32>>p41>>p42>>p51>>p52>>p61>>p62>>p71>>p72>>p81>>p82>>p91>>p92>>p101>>p102>>p111>>p112>>p121>>p122;
        std::vector<cv::Point2f> points;
        points.push_back(cv::Point2f(std::stof(p11), std::stof(p12)));
        points.push_back(cv::Point2f(std::stof(p21), std::stof(p22)));
        points.push_back(cv::Point2f(std::stof(p31), std::stof(p32)));
        points.push_back(cv::Point2f(std::stof(p41), std::stof(p42)));
        points.push_back(cv::Point2f(std::stof(p51), std::stof(p52)));
        points.push_back(cv::Point2f(std::stof(p61), std::stof(p62)));
        points.push_back(cv::Point2f(std::stof(p71), std::stof(p72)));
        points.push_back(cv::Point2f(std::stof(p81), std::stof(p82)));
        points.push_back(cv::Point2f(std::stof(p91), std::stof(p92)));
        points.push_back(cv::Point2f(std::stof(p101), std::stof(p102)));
        points.push_back(cv::Point2f(std::stof(p111), std::stof(p112)));
        points.push_back(cv::Point2f(std::stof(p121), std::stof(p122)));
        points_sets.push_back(points);
    }

    // mglGLUT gr(sample, "MathGL examples");
    
    for (int img_index = 1; img_index<=200; img_index++) {
        // 获取图片
        char buffer[20];
        sprintf(buffer, "img_%04d.jpg", img_index);
        std::string img_file = path_to_dataset + "images_undistorted/" + buffer;
        cv::Mat img = cv::imread(img_file, -1);

        // 利用DLT求解
        EstimatePoseDLT dlt(points_sets[img_index-1], corners, k);
        cv::Mat R;
        cv::Mat t;
        dlt.computePose(R, t);
        cv::Mat P(3, 4, CV_32F);
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                P.at<float>(i,j) = R.at<float>(i,j);
            }
        }
        for (int i = 0; i < 3; i++)
        {
            P.at<float>(i,3) = t.at<float>(i,0);
        }
        for (int i = 0; i < dlt.number_of_correspondences_; i++)
        {
            cv::circle(img, reprojectPoints(corners[i], P, k), 5, cv::Scalar(0 , 0, 125));
            cv::drawMarker(img, points_sets[img_index-1][i], cv::Scalar(0, 0, 125));
        }
        cv::imshow("", img);
        cv::waitKey(30);
    }
    return 0;
}

cv::Point2f reprojectPoints(cv::Point3f& P, cv::Mat& M, cv::Mat& k)
{
    cv::Point2f pixel_p;
    cv::Mat project_matrix(k*M);

    pixel_p.x = (P.x*project_matrix.at<float>(0,0) + P.y*project_matrix.at<float>(0,1) + P.z*project_matrix.at<float>(0,2) + project_matrix.at<float>(0,3))
                /(P.x*project_matrix.at<float>(2,0) + P.y*project_matrix.at<float>(2,1) + P.z*project_matrix.at<float>(2,2) + project_matrix.at<float>(2,3));

    pixel_p.y = (P.x*project_matrix.at<float>(1,0) + P.y*project_matrix.at<float>(1,1) + P.z*project_matrix.at<float>(1,2) + project_matrix.at<float>(1,3))
                /(P.x*project_matrix.at<float>(2,0) + P.y*project_matrix.at<float>(2,1) + P.z*project_matrix.at<float>(2,2) + project_matrix.at<float>(2,3));

    return pixel_p;
}