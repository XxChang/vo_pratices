#include <opencv2/opencv.hpp>

class EstimatePoseDLT
{
public:
    EstimatePoseDLT(const std::vector<cv::Point2f>& p_pixel,
                    const std::vector<cv::Point3f>& p_world,
                    const cv::Mat& camera_matrix);
    

    void computePose(cv::Mat& R, cv::Mat& t);

    int number_of_correspondences_;
    cv::Mat q;
    std::vector<cv::Point2f> cs;
    std::vector<cv::Point2f> us;
    std::vector<cv::Point3f> pws;

    void fillQ();

    void calibratePoints(void);
    cv::Matx33f k_inves;
    cv::Matx43f cws, ccs;
};