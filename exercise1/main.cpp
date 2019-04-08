#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>

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
    kin>>k11>>k12>>k13>>k21>>k22>>k23>>k31>>k32>>k33;
    cv::Mat k = (cv::Mat_<float> (3,3) << 
                std::stof(k11), std::stof(k12), std::stof(k13),
                std::stof(k21), std::stof(k22), std::stof(k23),
                std::stof(k31), std::stof(k32), std::stof(k33)
                );

    // 设置矫正参数
    std::string d_file = path_to_dataset + "D.txt";
    std::string d1, d2;
    std::ifstream din(d_file);
    din>>d1>>d2;

    // 获取外参文件
    std::string pose_file = path_to_dataset + "poses.txt";
    std::string r1, r2, r3, tl1, tl2, tl3;
    std::ifstream pin(pose_file);


    for (int img_index=1; img_index<700; img_index++) {
        // 获取图片
        char buffer[20];
        sprintf(buffer, "img_%04d.jpg", img_index);
        std::string img_file = path_to_dataset + "images/"+buffer;
        cv::Mat img = cv::imread(img_file, -1);

        // 设置外参
        pin>>r1>>r2>>r3>>tl1>>tl2>>tl3;
        cv::Mat rvec = (cv::Mat_<float>(3,1)<<std::stof(r1), std::stof(r2), std::stof(r3));
        cv::Mat rmat(3, 3, CV_32F);
        Rodrigues(rvec, rmat);
        cv::Mat tmat = (cv::Mat_<float>(3,4)<< rmat.at<float>(0,0), rmat.at<float>(0,1), rmat.at<float>(0,2), std::stod(tl1),
                                            rmat.at<float>(1,0), rmat.at<float>(1,1), rmat.at<float>(1,2), std::stod(tl2),
                                            rmat.at<float>(2,0), rmat.at<float>(2,1), rmat.at<float>(2,2), std::stod(tl3));

        // 设置点的位置
        std::vector<cv::Point3f> points;
        points.push_back(cv::Point3f(3*0.04,1*0.04, 0));
        points.push_back(cv::Point3f(5*0.04,1*0.04, 0));
        points.push_back(cv::Point3f(3*0.04,3*0.04, 0));
        points.push_back(cv::Point3f(5*0.04,3*0.04, 0)); 
        points.push_back(cv::Point3f(3*0.04,1*0.04, -2*0.04));
        points.push_back(cv::Point3f(5*0.04,1*0.04, -2*0.04));
        points.push_back(cv::Point3f(3*0.04,3*0.04, -2*0.04));
        points.push_back(cv::Point3f(5*0.04,3*0.04, -2*0.04)); 

        // 计算点的像素坐标
        std::vector<cv::Point2f> pixels;
        for(int i = 0; i < (int) points.size(); i++) {
            // 计算世界坐标位置
            cv::Mat pos(3, 1, CV_32F);
            pos = (cv::Mat_<float>(4,1)<< points[i].x, points[i].y, points[i].z, 1);
            cv::Mat pos_c = tmat * pos; 
            pos_c = pos_c * (1/pos_c.at<float>(2,0)); // 获得归一化坐标

            // 矫正畸变
            float r = pos_c.at<float>(0,0) * pos_c.at<float>(0,0) + pos_c.at<float>(1,0) * pos_c.at<float>(1,0);
            pos_c.at<float>(0,0) = pos_c.at<float>(0,0)*(1 + std::stof(d1)*r + std::stof(d2)*r*r);
            pos_c.at<float>(1,0) = pos_c.at<float>(1,0)*(1 + std::stof(d1)*r + std::stof(d2)*r*r);
            // 转换到相机坐标系
            pos_c = k*pos_c;

            pixels.push_back(cv::Point2f(pos_c.at<float>(0,0), pos_c.at<float>(1,0)));
        }

        // 画图
        cv::line(img, pixels[0], pixels[1], cv::Scalar(0, 0, 255));
        cv::line(img, pixels[1], pixels[3], cv::Scalar(0, 0, 255));
        cv::line(img, pixels[3], pixels[2], cv::Scalar(0, 0, 255));
        cv::line(img, pixels[2], pixels[0], cv::Scalar(0, 0, 255));
        cv::line(img, pixels[0], pixels[4], cv::Scalar(0, 0, 255));
        cv::line(img, pixels[1], pixels[5], cv::Scalar(0, 0, 255));
        cv::line(img, pixels[2], pixels[6], cv::Scalar(0, 0, 255));
        cv::line(img, pixels[3], pixels[7], cv::Scalar(0, 0, 255));
        cv::line(img, pixels[4], pixels[5], cv::Scalar(0, 0, 255));
        cv::line(img, pixels[5], pixels[7], cv::Scalar(0, 0, 255));
        cv::line(img, pixels[7], pixels[6], cv::Scalar(0, 0, 255));
        cv::line(img, pixels[6], pixels[4], cv::Scalar(0, 0, 255));

        cv::imshow("",img);
        cv::waitKey(30);
    }
}