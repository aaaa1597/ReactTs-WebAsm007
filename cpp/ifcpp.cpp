#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include "../src/MainProcess.h"

void printMatInfo(const cv::Mat& mat, const std::string &str = "");

float markerLength = 0.065f;  /* ホントは3.5cmなんだけど、表示が大きすぎなので補正 */

/********/
/* main */
/********/
int main(int argc, char *argv[]) {
  cv::Mat cameraMatrix2 = (cv::Mat_<double>(3,3) << 1.42068235e+03,0.00000000e+00,9.49208512e+02, 0.00000000e+00,1.37416685e+03,5.39622051e+02,0.00000000e+00,0.00000000e+00,1.00000000e+00);
  cv::Mat distCoeffs2 = (cv::Mat_<double>(5,1) << 1.69926613e-01,-7.40003491e-01,-7.45655262e-03,-1.79442353e-03, 2.46650225e+00);
  printMatInfo(cameraMatrix2, "cameraMatrix2");
  printMatInfo(distCoeffs2, "distCoeffs2");

  // Set coordinate system
  cv::Mat objPoints(4, 1, CV_32FC3);
  objPoints.ptr<cv::Vec3f>(0)[0] = cv::Vec3f(-markerLength/2.f,  markerLength/2.f, 0);
  objPoints.ptr<cv::Vec3f>(0)[1] = cv::Vec3f( markerLength/2.f,  markerLength/2.f, 0);
  objPoints.ptr<cv::Vec3f>(0)[2] = cv::Vec3f( markerLength/2.f, -markerLength/2.f, 0);
  objPoints.ptr<cv::Vec3f>(0)[3] = cv::Vec3f(-markerLength/2.f, -markerLength/2.f, 0);

  cv::aruco::DetectorParameters detectorParams = cv::aruco::DetectorParameters();
  cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_100);
  cv::aruco::ArucoDetector detector(dictionary, detectorParams);

  const std::vector<std::string> files = {"../../IMG_000.jpg", "../../IMG_001.jpg", "../../IMG_002.jpg", "../../IMG_003.jpg", "../../IMG_004.jpg",
                                          "../../IMG_005.jpg", "../../IMG_006.jpg", "../../IMG_007.jpg", "../../IMG_008.jpg", "../../IMG_009.jpg",
                                          "../../IMG_010.jpg", "../../IMG_011.jpg", "../../IMG_012.jpg", "../../IMG_013.jpg", "../../IMG_014.jpg",
                                          "../../IMG_015.jpg", "../../IMG_016.jpg", "../../IMG_017.jpg", "../../IMG_018.jpg", "../../IMG_019.jpg",
                                          "../../IMG_020.jpg", "../../IMG_021.jpg", "../../IMG_022.jpg", "../../IMG_023.jpg", };

  for(const std::string &file : files) {
    cv::Mat image = cv::imread(file);
    cv::Mat imageCopy;
    image.copyTo(imageCopy);
    std::vector<int> ids;
    std::vector<std::vector<cv::Point2f>> corners;
    detector.detectMarkers(image, corners, ids);

    /* If at least one marker detected */
    if (ids.size() > 0) {
        cv::aruco::drawDetectedMarkers(imageCopy, corners, ids);
        int nMarkers = corners.size();
        std::vector<cv::Vec3d> rvecs(nMarkers), tvecs(nMarkers);
        /* Calculate pose for each marker */
        for(int lpi = 0; lpi < nMarkers; lpi++) {
            solvePnP(objPoints, corners.at(lpi), cameraMatrix2, distCoeffs2, rvecs.at(lpi), tvecs.at(lpi));
        }
        /* Draw axis for each marker */
        for(unsigned int lpj = 0; lpj < ids.size(); lpj++) {
            cv::drawFrameAxes(imageCopy, cameraMatrix2, distCoeffs2, rvecs[lpj], tvecs[lpj], 0.1f);
        }
    }

    /* Output RGB image. */
    int path_i = file.find_last_of("/") + 1;
    int ext_i = file.find_last_of(".");
    std::string pathname = file.substr(0, path_i);
    std::string filename = file.substr(path_i, ext_i-path_i);
    std::string extname = file.substr(ext_i, file.size()-ext_i);
    std::string outfile = pathname + "out" + filename + extname;
    cv::imwrite(outfile, imageCopy);
  }

  return 0;
}

/********************************************************/
/* printMatInfo                                         */
/*   cv::Matの情報を表示する関数(cv::Matは分かりづらいから) */
/********************************************************/
void printMatInfo(const cv::Mat& mat, const std::string &str) {
  std::cout << str << "(row:" << mat.rows << " col:" << mat.cols;
  /* 要素の型 */
  std::cout << " type:" << (
      mat.type() == CV_8UC1  ? "CV_8UC1" :
      mat.type() == CV_8UC2  ? "CV_8UC2" :
      mat.type() == CV_8UC3  ? "CV_8UC3" :
      mat.type() == CV_8UC4  ? "CV_8UC4" :
      mat.type() == CV_8SC1  ? "CV_8SC1" :
      mat.type() == CV_8SC2  ? "CV_8SC2" :
      mat.type() == CV_8SC3  ? "CV_8SC3" :
      mat.type() == CV_8SC4  ? "CV_8SC4" :
      mat.type() == CV_16UC1 ? "CV_16UC1" :
      mat.type() == CV_16UC2 ? "CV_16UC2" :
      mat.type() == CV_16UC3 ? "CV_16UC3" :
      mat.type() == CV_16UC4 ? "CV_16UC4" :
      mat.type() == CV_16SC1 ? "CV_16SC1" :
      mat.type() == CV_16SC2 ? "CV_16SC2" :
      mat.type() == CV_16SC3 ? "CV_16SC3" :
      mat.type() == CV_16SC4 ? "CV_16SC4" :
      mat.type() == CV_32SC1 ? "CV_32SC1" :
      mat.type() == CV_32SC2 ? "CV_32SC2" :
      mat.type() == CV_32SC3 ? "CV_32SC3" :
      mat.type() == CV_32SC4 ? "CV_32SC4" :
      mat.type() == CV_32FC1 ? "CV_32FC1" :
      mat.type() == CV_32FC2 ? "CV_32FC2" :
      mat.type() == CV_32FC3 ? "CV_32FC3" :
      mat.type() == CV_32FC4 ? "CV_32FC4" :
      mat.type() == CV_64FC1 ? "CV_64FC1" :
      mat.type() == CV_64FC2 ? "CV_64FC2" :
      mat.type() == CV_64FC3 ? "CV_64FC3" :
      mat.type() == CV_64FC4 ? "CV_64FC4" :
      mat.type() == CV_16FC1 ? "CV_16FC1" :
      mat.type() == CV_16FC2 ? "CV_16FC2" :
      mat.type() == CV_16FC3 ? "CV_16FC3" :
      mat.type() == CV_16FC4 ? "CV_16FC4" :
      "other"
      );

  /* 深度 */
  std::cout << " depth:" << (
      mat.depth() == CV_8U  ? "CV_8U" :
      mat.depth() == CV_8S  ? "CV_8S" :
      mat.depth() == CV_16U ? "CV_16U" :
      mat.depth() == CV_16S ? "CV_16S" :
      mat.depth() == CV_32S ? "CV_32S" :
      mat.depth() == CV_32F ? "CV_32F" :
      mat.depth() == CV_64F ? "CV_64F" :
      mat.depth() == CV_16F ? "CV_16F" :
      "other"
      );

  /* チャンネル数 */
  std::cout << " channels:" << mat.channels();

  /* バイト列が連続しているか */
  std::cout << " continuous:" <<
      (mat.isContinuous() ? "true" : "false") << ")" << std::endl;

  /* matの中身 */
  std::cout << mat << std::endl;
  return;
}
