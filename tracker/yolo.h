#pragma once

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

const constexpr float CONFIDENCE_THRESHOLD = 0.5;
const constexpr float NMS_THRESHOLD = 0.4;																		// 곂침 방지 계수
const constexpr int NUM_CLASSES = 80;

const cv::Scalar colors[] = {
	{0, 255, 255},
	{255, 255, 0},
	{0, 255, 0},
	{255, 0, 0}
};
const auto NUM_COLORS = sizeof(colors) / sizeof(colors[0]);

cv::Mat process(const std::string& file, const std::vector<cv::String>& output_names,							// 객체 검출
	const std::vector<std::string>& class_names, cv::dnn::dnn4_v20200609::Net& net, cv::Mat& frame);
void forwardImage(const std::string& file, const std::vector<cv::String>& output_names,							// 객체검출(이미지)
	const std::vector<std::string>& class_names, cv::dnn::dnn4_v20200609::Net& net);
void forwardVideo(const std::string& file, const std::vector<cv::String>& output_names,							// 객체검출(동영상)
	const std::vector<std::string>& class_names, cv::dnn::dnn4_v20200609::Net& net);
int yolo();