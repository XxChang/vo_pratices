#include "estimate_pose_dlt.hpp"

EstimatePoseDLT::EstimatePoseDLT(const std::vector<cv::Point2f>& p_pixel,
                                 const std::vector<cv::Point3f>& p_world,
                                 const cv::Mat& camera_matrix)
{
    CV_Assert( camera_matrix.size() == cv::Size( 3,3 ) && p_pixel.size() == p_world.size() );
    number_of_correspondences_ = p_pixel.size();
    // 将像素坐标赋予成员变量
    us = p_pixel;
 
    // 将世界坐标赋予成员变量
    pws = p_world;

    // 将像素坐标变到归一化坐标系下
    cv::invert(camera_matrix, k_inves);

    calibratePoints();

    // 设置Q矩阵
    q.create(2*number_of_correspondences_, 12, CV_32F); 

    // choose_control_points(); 
    // 填充Q矩阵的元素
    fillQ();

}

void EstimatePoseDLT::computePose(cv::Mat& R, cv::Mat& t)
{
    // svd分解
    cv::Mat DC;
    cv::Mat UCt;
    cv::Mat CU;
    cv::Mat r(3, 3, CV_32F);
    cv::SVDecomp(q, DC, UCt, CU, cv::SVD::FULL_UV);

    // 注意CU是倒置后的
    cv::Mat u(CU.t());
    if(u.at<float>(0,11)<0)
    {
        u = -1*u;
    }

    // 提取出R
    // Orthogonal Procrustes problem
    r = (cv::Mat_<float>(3,3)<< u.at<float>(0,11), u.at<float>(1,11), u.at<float>(2,11),
                                u.at<float>(4,11), u.at<float>(5,11), u.at<float>(6,11),
                                u.at<float>(8,11), u.at<float>(9,11), u.at<float>(10,11));

    cv::Mat U;
    cv::Mat Vt;
    cv::Mat eigen_values;
    cv::SVDecomp(r, eigen_values, U, Vt, cv::SVD::FULL_UV);

    R = U*Vt;

    // 求t
    // 回复投影矩阵M的尺度
    float scale = cv::norm(R, cv::NORM_L2)/cv::norm(r, cv::NORM_L2);
    t = scale*(cv::Mat_<float>(3,1)<< u.at<float>(3,11),
                                      u.at<float>(7,11),
                                      u.at<float>(11,11));

}

void EstimatePoseDLT::calibratePoints(void)
{
    for ( auto u:us)
    {
        float x = u.x*k_inves(0,0) + u.y*k_inves(0,1) + 1 * k_inves(0,2);
        float y = u.x*k_inves(1,0) + u.y*k_inves(1,1) + 1 * k_inves(1,2);
    
        cs.push_back(cv::Point2f(x,y));
    }
}

void EstimatePoseDLT::fillQ()
{
    for(int i = 0; i < number_of_correspondences_; i++)
    {
        q.at<float>( 2 * i, 0 ) = pws[i].x;
        q.at<float>( 2 * i, 1 ) = pws[i].y;
        q.at<float>( 2 * i, 2 ) = pws[i].z;
        q.at<float>( 2 * i, 3 ) = 1;
        q.at<float>( 2 * i, 4 ) = 0;
        q.at<float>( 2 * i, 5 ) = 0;
        q.at<float>( 2 * i, 6 ) = 0;
        q.at<float>( 2 * i, 7 ) = 0;
        q.at<float>( 2 * i, 8 ) = -cs[i].x*pws[i].x;
        q.at<float>( 2 * i, 9 ) = -cs[i].x*pws[i].y;
        q.at<float>( 2 * i, 10 ) = -cs[i].x*pws[i].z;
        q.at<float>( 2 * i, 11 ) = -cs[i].x;

        q.at<float>( 2 * i + 1, 0 ) = 0;
        q.at<float>( 2 * i + 1, 1 ) = 0;
        q.at<float>( 2 * i + 1, 2 ) = 0;
        q.at<float>( 2 * i + 1, 3 ) = 0;
        q.at<float>( 2 * i + 1, 4 ) = pws[i].x;
        q.at<float>( 2 * i + 1, 5 ) = pws[i].y;
        q.at<float>( 2 * i + 1, 6 ) = pws[i].z;
        q.at<float>( 2 * i + 1, 7 ) = 1;
        q.at<float>( 2 * i + 1, 8 ) = -cs[i].y*pws[i].x;
        q.at<float>( 2 * i + 1, 9 ) = -cs[i].y*pws[i].y;
        q.at<float>( 2 * i + 1, 10 ) = -cs[i].y*pws[i].z;
        q.at<float>( 2 * i + 1, 11 ) = -cs[i].y;     
    }
}