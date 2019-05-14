#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QTimer>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <stdlib.h>

#include <dlib/opencv.h>
#include <dlib/image_io.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>


using namespace cv;

#include <iostream>
using namespace std;

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();



private slots:
    void on_pushButton_open_webcam_clicked();

    void on_pushButton_close_webcam_clicked();

    void update_window();
    void detectFaceOpenCVDNN(cv::dnn::Net net, Mat frame);

private:
    Ui::MainWindow *ui;

    QTimer *timer;
    VideoCapture cap;

    Mat frame;
    QImage qt_image;
    const double inScaleFactor = 1.0;
    const float confidenceThreshold = 0.7;
    cv::dnn::Net net;
    const std::string caffeConfigFile = "/home/sam/Dev/deploy.prototxt";
    const std::string caffeWeightFile = "/home/sam/Dev/res10_300x300_ssd_iter_140000_fp16.caffemodel";
    const std::string tensorflowConfigFile = "/home/sam/Dev/opencv_face_detector.pbtxt";
    const std::string tensorflowWeightFile = "/home/sam/Dev/opencv_face_detector_uint8.pb";
//    cv::Rect *face;
    dlib::rectangle face;
    std::vector<cv::Rect> faces;
    std::vector<dlib::rectangle> dets;
    dlib::shape_predictor sp;
    int x1,x2,y1,y2;



//    dlib::frontal_face_detector detector;
//    dlib::shape_predictor shape_model;
};

#endif // MAINWINDOW_H
