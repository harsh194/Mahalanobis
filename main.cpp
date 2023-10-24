#pragma once
#include "test.h"


int main() {

	std::vector<string> CLASS_NAMES = { "bottle", "cable", "capsule", "carpet", "grid",
		"hazelnut", "leather", "metal_nut", "pill", "screw",
		"tile", "toothbrush", "transistor", "wood", "zipper" };
	std::string root_path = "C:/Users/000501/source/repos/maha/data";
	std::string class_name = "bottle";
	//bool is_train = true;
	bool is_train = false;
	int resize = 256;
	int cropsize = 224;

	MVTecDataset m(root_path, class_name, is_train, resize, cropsize);
	cv::Mat src = cv::imread("000.png");
	//cv::Mat src = cv::imread("000_mask.png");

	//m.start(src);
    //DatasetInfo dataset = m.loadDatasetFolder();
	m.start();

	//Ort::Env env;

	////model location
	//auto modelPath = L"C:/Users/000501/source/repos/Mahalanobis/model/efficientnet-b4.onnx";

	//// create session
	//Ort::Session session(env, modelPath, Ort::SessionOptions());

	//std::cout << "Number of model inputs:- " << session.GetInputCount() << endl;
	//std::cout << "Number of model outputs:- " << session.GetOutputCount() << endl;

	//Ort::AllocatorWithDefaultOptions allocator;
	//
	//cout << session.GetInputNameAllocated(0, allocator)<< endl;
	//cout << session.GetOutputNameAllocated(0, allocator) << endl;

	//auto inputShape = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
	//cout << inputShape << endl;
	//
	//auto outputShape = session.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
	//cout << outputShape << endl;
	////where to allocate the tensors
	//auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

	//std::vector<float> inputValues = { 4,5,6 };

	//Create the input tensor (this is not a deep copy)
	//auto inputOnnxTensor = Ort::Value::CreateTensor<float>(memoryInfo, inputValues.data(), inputValues.size(),
	//	inputShape.data(), inputShape.size());


	return 0;
}




