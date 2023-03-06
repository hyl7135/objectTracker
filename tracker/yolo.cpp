#include "yolo.h"

#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <set>

#include <filesystem>
#include <memory>

#include "opencv2/core.hpp"
#include "opencv2/dnn.hpp"
#include "opencv2/dnn/all_layers.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>


#include "KalmanTracker.h"
#include "Hungarian.h"
#include "tracker.h"

// 프레임
// yolo size
// 이미지 크기, 앵글

#define YOLO_SIZE 320
// 320 416 608

#define IMAGE 1
#define VIDEO 2

#define TRACKING 1

int frame_count = 1;
std::vector<KalmanTracker> trackers;

cv::Mat process(const std::string& file, const std::vector<cv::String>& output_names,							// 객체 검출
	const std::vector<std::string>& class_names, cv::dnn::dnn4_v20200609::Net& net, cv::Mat& frame);
void forwardImage(const std::string& file, const std::vector<cv::String>& output_names,							// 객체검출(이미지)
	const std::vector<std::string>& class_names, cv::dnn::dnn4_v20200609::Net& net);
void forwardVideo(const std::string& file, const std::vector<cv::String>& output_names,							// 객체검출(동영상)
	const std::vector<std::string>& class_names, cv::dnn::dnn4_v20200609::Net& net);



int yolo() {
	//std::string path = "./../../image_person/human1.jpg";
	std::string path = "./sample/resize2_fps_6.avi";
	std::vector<std::string> class_names;
	KalmanTracker::kf_count = 0;

	std::ifstream class_file("./yolov4_weights/coco.names");
	if (!class_file) {
		std::cerr << "failed to open coco.names" << std::endl;
		return -1;
	}

	auto net = cv::dnn::readNetFromDarknet("./yolov4_weights/yolov4.cfg", "./yolov4_weights/yolov4.weights");
	net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);			//GPU 사용
	net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);				
	//net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);		//CPU 사용
	//net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);	
	auto output_names = net.getUnconnectedOutLayersNames();			//출력 레이어

	//forwardImage(path, output_names, class_names, net);
	forwardVideo(path, output_names, class_names, net);				

	/*
	for (const auto& str) {
		std::cout << str << std::endl;

		switch (fileType(str)) {
		case 1:
			forwardImage(str, output_names, class_names, net);
			break;
		case 2:
			forwardVideo(str, output_names, class_names, net);
			break;
		default: std::cerr << "cannot load this file." << std::endl;
		}
	}
	*/

	return 0;
}

cv::Mat process(const std::string& file, const std::vector<cv::String>& output_names,
	const std::vector<std::string>& class_names, cv::dnn::dnn4_v20200609::Net& net, cv::Mat& frame) {

	cv::TickMeter tm;
	tm.start();

	cv::Mat blob;
	std::vector<cv::Mat> detections;
	
	cv::dnn::blobFromImage(frame, blob, 0.00392, cv::Size(YOLO_SIZE, YOLO_SIZE), cv::Scalar(), true, false, CV_32F);
	net.setInput(blob);
	net.forward(detections, output_names);

	std::vector<int> indices[NUM_CLASSES];
	std::vector<cv::Rect> boxes[NUM_CLASSES];
	std::vector<float> scores[NUM_CLASSES];

	std::vector<TrackingBox> rectData;

	/*
	std::ofstream resultsFile;
	resultsFile.open("./../data/vtest.txt", ios::app);

	if (!resultsFile.is_open())
	{
		std::cerr << "Error: can not create file " << std::endl;
		//return;
	}
	*/

	for (auto& output : detections)
	{
		//auto output = detections[1];
		const auto num_boxes = output.rows;
		for (int i = 0; i < num_boxes; i++)
		{

			//std::cout << "rect x : " << rect.x << ", " << "rect.y : " << rect.y << std::endl;

			for (int c = 2; c < 3; c++)
			{
				auto confidence = *output.ptr<float>(i, 5 + c);
				if (confidence >= CONFIDENCE_THRESHOLD)
				{
					auto x = output.at<float>(i, 0) * frame.cols;
					auto y = output.at<float>(i, 1) * frame.rows;
					auto width = output.at<float>(i, 2) * frame.cols;
					auto height = output.at<float>(i, 3) * frame.rows;
					cv::Rect rect(x - width / 2, y - height / 2, width, height);

					boxes[c].push_back(rect);
					scores[c].push_back(confidence);
				}
			}
		}
	}

	for (int c = 0; c < NUM_CLASSES; c++)
		cv::dnn::NMSBoxes(boxes[c], scores[c], 0.0, NMS_THRESHOLD, indices[c]);


	int idxNum = 2;
	for (int c = 0 + idxNum; c < 1 + idxNum; c++)		// c < 1; 사람만
	{
		for (size_t i = 0; i < indices[c].size(); ++i)
		{
			auto idx = indices[c][i];
			const auto& rect = boxes[c][idx];
			TrackingBox temp;
			temp.frame = frame_count;
			temp.id = -1;
			temp.box.x = boxes[c][idx].x;
			temp.box.y = boxes[c][idx].y;
			temp.box.width = boxes[c][idx].width;
			temp.box.height = boxes[c][idx].height;
			rectData.push_back(temp);
			/*
			std::cout << frame_count << "," << -1 << "," << boxes[c][idx].x << "," << boxes[c][idx].y << ","
			*/
		}
	}
	//TestSORT(rectData, trackers);
	if (TRACKING) {
		for (const auto& i : TestSORT(rectData, trackers)) {
			std::string id = std::to_string(i.id);
			cv::putText(frame, id, cv::Point(i.box.x, i.box.y), 1, 2, cv::Scalar::all(255), 3);
			cv::rectangle(frame, i.box, cv::Scalar::all(255), 2, 8, 0);
		}
	}
	
	//resultsFile.close();
	frame_count++;
	tm.stop();
	std::cout << frame.cols << " X " << frame.rows << " takes " << tm.getAvgTimeSec() << "sec." << std::endl;
	
	return frame;
}

void forwardImage(const std::string& file, const std::vector<cv::String>& output_names,
	const std::vector<std::string>& class_names, cv::dnn::dnn4_v20200609::Net& net)
{
	cv::Mat frame = cv::imread(file, cv::IMREAD_COLOR);
	frame = process(file, output_names, class_names, net, frame);

	cv::namedWindow("output");
	cv::imshow("output", frame);
	cv::waitKey(0);

}

void forwardVideo(const std::string& file, const std::vector<cv::String>& output_names,
	const std::vector<std::string>& class_names, cv::dnn::dnn4_v20200609::Net& net)
{
	cv::VideoCapture source(file);
	cv::Mat frame;
	while (cv::waitKey(1) < 1) {
		source >> frame;
		if (frame.empty()) {
			cv::waitKey(0);
			break;
		}

		frame = process(file, output_names, class_names, net, frame);
		if (1) {
			cv::namedWindow("output");
			cv::imshow("output", frame);
		}
	}
}