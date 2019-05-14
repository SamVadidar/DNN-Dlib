#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    timer = new QTimer(this);
}

MainWindow::~MainWindow()
{
    delete ui;
}

//#define CAFFE
void MainWindow::detectFaceOpenCVDNN(cv::dnn::Net net, Mat frame)
{
    int frameHeight = frame.rows;
    int frameWidth = frame.cols;
    const cv::Scalar meanVal(104.0, 177.0, 123.0);

//#ifdef CAFFE
        cv::Mat inputBlob = cv::dnn::blobFromImage(frame, inScaleFactor, cv::Size(300, 300 ), meanVal, false, false);
//#else
//        cv::Mat inputBlob = cv::dnn::blobFromImage(frameOpenCVDNN, inScaleFactor, cv::Size(320, 240), meanVal, true, false);
//#endif

    net.setInput(inputBlob, "data");
    cv::Mat detection = net.forward("detection_out");

    cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

    for(int i = 0; i < detectionMat.rows; i++)
    {
        float confidence = detectionMat.at<float>(i, 2);

        if(confidence > confidenceThreshold)
        {
            x1 = static_cast<int>(detectionMat.at<float>(i, 3) * frameWidth);
            y1 = static_cast<int>(detectionMat.at<float>(i, 4) * frameHeight);
            x2 = static_cast<int>(detectionMat.at<float>(i, 5) * frameWidth);
            y2 = static_cast<int>(detectionMat.at<float>(i, 6) * frameHeight);

            cv::rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0),2, 4);
//            cv::rectangle(frame, cv::Point(50, 50), cv::Point(140, 140), cv::Scalar(0, 255, 0),2, 4);
            face = dlib::rectangle(x1,y1,x2,y2);

//            dlib::rectangle()
//            faces[0]=face;
//            cout << faces[0].height << endl;
        }
    }

}

void MainWindow::on_pushButton_open_webcam_clicked()
{
    cap.open(2);
//#ifdef CAFFE
    net = cv::dnn::readNetFromCaffe(caffeConfigFile, caffeWeightFile);
//#else
//  cv::dnn::Net net = cv::dnn::readNetFromTensorflow(tensorflowWeightFile, tensorflowConfigFile);
//#endif

//    dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
    dlib::deserialize("/home/sam/Dev/shape_predictor_5_face_landmarks.dat") >> sp;

    if(!cap.isOpened())  // Check if we succeeded
    {
        cout << "camera is not open" << endl;
    }
    else
    {
        cout << "camera is open" << endl;

        connect(timer, SIGNAL(timeout()), this, SLOT(update_window()));
        timer->start(20);
    }
}

void MainWindow::update_window()
{
    cap >> frame;
    cv::resize(frame, frame, cv::Size(320,240));
    detectFaceOpenCVDNN (net, frame);
    dlib::cv_image<dlib::bgr_pixel> cimg(frame);
    dlib::full_object_detection shape = sp(cimg,face);

//    cout<<shape.num_parts()<<endl;
    for (int i=0; i<5; ++i){
        cv::circle(frame, cv::Point(shape.part(i).x(), shape.part(i).y()), 2, cv::Scalar(0, 0, 255), cv::FILLED);
    }
//    shape.part(0)
//    dets = face;

//    dlib::array2d<dlib::bgr_pixel> dlib_image;
//    dlib::assign_image(dlib_image, dlib::cv_image<dlib::bgr_pixel>(frame));

//    std::vector<dlib::rectangle> detected_faces = detector(dlib_image);
//    int number_of_detected_faces = detected_faces.size();

//    std::vector<dlib::full_object_detection> shapes;

//    for (int i=0; i<number_of_detected_faces; i++){
//        dlib::full_object_detection shape= shape_model(dlib_image, detected_faces[i]);

//        for (int j=0; j<68; j++){
//            cv::circle(frame, cv::Point(shape.part(j).x(), shape.part(j).y()), 2, cv::Scalar(0, 0, 255), 1, cv::FILLED);
//        }
//    }

    cvtColor(frame, frame, CV_BGR2RGB);

    qt_image = QImage((const unsigned char*) (frame.data), frame.cols, frame.rows, QImage::Format_RGB888);

    ui->label->setPixmap(QPixmap::fromImage(qt_image));

//    ui->label->resize(ui->label->pixmap()->size());
}

void MainWindow::on_pushButton_close_webcam_clicked()
{
    disconnect(timer, SIGNAL(timeout()), this, SLOT(update_window()));
    cap.release();

    Mat image = Mat::zeros(frame.size(),CV_8UC3);

    qt_image = QImage((const unsigned char*) (image.data), image.cols, image.rows, QImage::Format_RGB888);

    ui->label->setPixmap(QPixmap::fromImage(qt_image));

    ui->label->resize(ui->label->pixmap()->size());

    cout << "camera is closed" << endl;
}
