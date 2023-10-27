#pragma once
#include "test.h"


// Constructor
MVTecDataset::MVTecDataset(const std::string& root_path, const std::string& class_name, bool is_train, int resize, int cropsize)
	: root_path(root_path), class_name(class_name), is_train(is_train), resize(resize), cropsize(cropsize) {
	
}

// Function Defination
void MVTecDataset::start() {
	PParameter pPara = PParameter(new Parameter());
	bool debug_flag = start_operation;

	static TParas tp = []() -> TParas {

		return tp;
	}();

	if (debug_flag) {
		static bool trackbar_flag = [&]() -> bool {

			CreateWindow(1);
			Track("show", 1, tp.show, 0, NULL);

			return true;
		}();
	}
	 
	do {
		std::vector<string> CLASS_NAMES = { "bottle", "cable", "capsule", "carpet", "grid",
	    "hazelnut", "leather", "metal_nut", "pill", "screw",
	    "tile", "toothbrush", "transistor", "wood", "zipper" };

		// ONNX environment
		Ort::Env env;	
		auto modelPath = L"C:/Users/000501/source/repos/Mahalanobis/model/efficientnet-b4.onnx";    //model location	
		Ort::RunOptions runOptions;
		Ort::Session session(env, modelPath, Ort::SessionOptions());

		constexpr int64_t numChannels = 3;
		constexpr int64_t width = 224;
		constexpr int64_t height = 224;
		constexpr int64_t numClasses = 1000;
		constexpr int64_t numInputElements = numChannels * height * width;

		// define shape
		const std::array<int64_t, 4> inputShape = { 1, numChannels, height, width };
		const std::array<int64_t, 2> outputShape = { 1, numClasses };

		// define array
		std::array<float, numInputElements> input;
		std::array<float, numClasses> results;

		Ort::AllocatorWithDefaultOptions allocator;

		cout << session.GetInputNameAllocated(0, allocator) << endl;
		cout << session.GetOutputNameAllocated(0, allocator) << endl;
		
		// define names
		Ort::AllocatorWithDefaultOptions ort_alloc;
		Ort::AllocatedStringPtr inputName = session.GetInputNameAllocated(0, ort_alloc);
		Ort::AllocatedStringPtr outputName = session.GetOutputNameAllocated(0, ort_alloc);
		const std::array<const char*, 1> inputNames = { inputName.get() };
		const std::array<const char*, 1> outputNames = { outputName.get() };
		
		inputName.release();
		outputName.release();

		//Result folder
		string RT = "C:/Users/000501/source/repos/Mahalanobis";
		string ResultDir = RT + "/" + "result";

		if (!std::filesystem::exists(ResultDir))
		{
			std::filesystem::create_directories(ResultDir);
			cout << "Result Directory created" << endl;
		}
		else
		{
			cout << "Result directory already exists" << endl;
		}

		//Temp folder
		string TempDir = ResultDir + "/" + "temp";
		if (!std::filesystem::exists(TempDir))
		{
			std::filesystem::create_directories(TempDir);
			cout << "Temp Directory created" << endl;
		}
		else
		{
			cout << "Temp Directory already exists" << endl;
		}

		std::vector<float> total_roc_auc;
		int batch_size = 32;

		for (const auto& class_name : CLASS_NAMES)
		{
			MVTecDataset train(root_path, class_name, true, resize, cropsize);
			MVTecDataset test(root_path, class_name, false, resize, cropsize);
			
			DatasetInfo train_dataset = train.loadDatasetFolder();
			DatasetInfo test_dataset = test.loadDatasetFolder();

			vector<vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>>> train_dataloader;
			vector<vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>>> test_dataloader;

			train_dataloader = createBatch(train_dataset);
			test_dataloader = createBatch(test_dataset);

			cout << class_name << " - Train dataset size - " << train_dataloader.size() << endl;
			cout << class_name << " - Test dataset size - " << test_dataloader.size() << endl;

			std::vector<std::vector<torch::Tensor>> train_outputs;
			std::vector<std::vector<torch::Tensor>> test_outputs;

			// Extract train set features
			std::filesystem::path train_feat_filepath = TempDir;
			train_feat_filepath /= "train_" + class_name + "_" + "efficientnet-b4.onnx";
			//train_feat_filepath /= "train_" + class_name + "_" + "efficientnet-b4.txt";
			cout << train_feat_filepath << endl;

			//Writing the features
			if (!std::filesystem::exists(train_feat_filepath))
			{
				std::ofstream outputChunkFile(train_feat_filepath, std::ios::binary); // Open the output file once
				//std::vector<Ort::Value> output_tensors;
				int m = 1;
				for (const auto& batch : train_dataloader)
				{ 
					std::string progress_bar_text = "| feature extraction | train | " + class_name + " |" + " batch " + std::to_string(m);
					cout << progress_bar_text << endl;
					cout << endl;

					std::vector<torch::Tensor> batch_outputs;
					for (const auto& data : batch)
					{
						torch::Tensor x = std::get<0>(data);
						torch::Tensor y = std::get<1>(data);
						torch::Tensor mask = std::get<2>(data);
					
						torch::NoGradGuard no_grad;   // Disable Gradient computation
						auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU); //memory allocation
						auto inputTensor = Ort::Value::CreateTensor<float>(memoryInfo, x.data_ptr<float>(), input.size(), inputShape.data(), inputShape.size());
						auto outputTensor = Ort::Value::CreateTensor<float>(memoryInfo, results.data(), results.size(), outputShape.data(), outputShape.size());

						// run inference
						try {
							session.Run(runOptions, inputNames.data(), &inputTensor, 1, outputNames.data(), &outputTensor, 1);
							cout << "ONNX running" << endl;
						}
						catch (Ort::Exception& e) {
							std::cout << e.what() << std::endl;
						}

						// sort results
						std::vector<std::pair<size_t, float>> indexValuePairs;

						// Check the results size and see the value of each
						//cout <<"Size of result - " << results.size() << endl;
						for (size_t i = 0; i < results.size(); ++i) {
							indexValuePairs.emplace_back(i, results[i]);
							//cout << " value - "<<i<<" - " << results[i]<<endl;
						}
						
						int n = results.size();
						//cout << "The size of the result - git " << n << endl;
						
						torch::Tensor feat = torch::from_blob(outputTensor.GetTensorMutableData<float>(), { outputShape[1] });
						
						//std::cout << "Feat tensor:- " << feat << endl;
						batch_outputs.push_back(feat);
						


						// Write the extracted features to the output file
						//outputChunkFile.write(reinterpret_cast<char*>(results.data()), results.size() * sizeof(float));
					}
					m = m + 1;
					train_outputs.push_back(batch_outputs);
					

				}

				calcMeanCovariance(train_outputs);

				std::ofstream output_file(train_feat_filepath, std::ios::binary);
				if (output_file.is_open())
				{
					for (const auto& inner_vector : train_outputs)
					{
						for (const auto& tensor : inner_vector) 
						{
							auto tensor_data = tensor.contiguous().to(torch::kFloat).data_ptr<float>();
							int64_t data_size = tensor.numel() * sizeof(float);
							output_file.write(reinterpret_cast<const char*>(&data_size), sizeof(int64_t));
							output_file.write(reinterpret_cast<const char*>(tensor_data), data_size);
						}
					}
					output_file.close();
				}
				else
				{
					std::cerr << "Error: Failed to open file for writing" << endl;
				}
				
			}
			//Loading the features 
			else
			{
				readFeatures(train_feat_filepath, train_outputs);
				//cout << "load train set feature distribution from - " << train_feat_filepath << endl;

				//std::ifstream input_file(train_feat_filepath, std::ios::binary);
				//if (input_file.is_open()) {
				//	train_outputs.clear(); // Clear the existing data if needed

				//	while (!input_file.eof()) {
				//		int64_t data_size = 0;
				//		input_file.read(reinterpret_cast<char*>(&data_size), sizeof(int64_t));

				//		if (data_size <= 0) {
				//			break;
				//		}

				//		std::vector<float> tensor_data(data_size / sizeof(float));
				//		input_file.read(reinterpret_cast<char*>(tensor_data.data()), data_size);
				//		torch::Tensor tensor = torch::from_blob(tensor_data.data(), { static_cast<long>(tensor_data.size()) }, torch::kFloat);
				//		train_outputs.push_back(std::vector<torch::Tensor>{tensor});
				//	}
				//	input_file.close();
				//}
				//else {
				//	std::cerr << "Error: Failed to open file for reading." << std::endl;
				//}
			}		
		}

		if (debug_flag) {
			vector<pair<Mat, string>> ProcessImages{

			};

			cvex::ShowProcess(ProcessImages, tp, pPara);

			switch (cv::waitKey(1)) {

			case 's':
				debug_flag = false;
				break;

			}

		}
	} while (debug_flag);
}

void MVTecDataset::readFeatures(std::filesystem::path train_feat_filepath, std::vector<std::vector<torch::Tensor>>& train_outputs) {

	bool debug_flag = read_features;

	static TParas tp = []() -> TParas {

		return tp;
	}();

	if (debug_flag) {
		static bool trackbar_flag = [&]() -> bool {

			CreateWindow(1);
			Track("show", 1, tp.show, 0, NULL);

			return true;
		}();
	}

	do {
		cout << "load train set feature distribution from - " << train_feat_filepath << endl;

		std::ifstream input_file(train_feat_filepath, std::ios::binary);
		if (input_file.is_open()) {
			train_outputs.clear();

			while (!input_file.eof()) {
				int64_t data_size = 0;
				input_file.read(reinterpret_cast<char*>(&data_size), sizeof(int64_t));

				if (data_size <= 0) {
					break;
				}
				std::vector<float> tensor_data(data_size / sizeof(float));
				input_file.read(reinterpret_cast<char*>(tensor_data.data()), data_size);
				torch::Tensor tensor = torch::from_blob(tensor_data.data(), { static_cast<long>(tensor_data.size()) }, torch::kFloat);
				train_outputs.push_back(std::vector<torch::Tensor>{tensor});
			}
			input_file.close();
		}
		else {
			std::cerr << "Error: Failed to open file for reading." << std::endl;
		}


		if (debug_flag) {
			vector<pair<Mat, string>> ProcessImages{

			};

			cvex::ShowProcess(ProcessImages, tp, pPara);

			switch (cv::waitKey(1)) {

			case 's':
				debug_flag = false;
				break;

			}

		}
	} while (debug_flag);
}


void MVTecDataset::calcMeanCovariance(std::vector<std::vector<torch::Tensor>>& train_outputs) {

	bool debug_flag = calc_mean_covariance;

	static TParas tp = []() -> TParas {

		return tp;
	}();

	if (debug_flag) {
		static bool trackbar_flag = [&]() -> bool {

			CreateWindow(1);
			Track("show", 1, tp.show, 0, NULL);

			return true;
		}();
	}

	do {
		for (int i = 0; i < train_outputs.size(); i++)
		{
			std::vector<torch::Tensor> train_output = train_outputs[i];
			torch::Tensor mean = torch::zeros(train_output[0].sizes());

			for (const torch::Tensor& tensor : train_output)
			{
				mean += tensor;
			}
			mean /= static_cast<float>(train_output.size());
			int rank = mean.size(0);
			auto mean_data = mean.to(torch::kFloat).data_ptr<float>();

			Eigen::MatrixXf covariance = Eigen::MatrixXf::Zero(1, 1);
			for (const torch::Tensor& tensor : train_output)
			{
				torch::Tensor centered = tensor - mean;
				auto centered_data = centered.to(torch::kFloat).data_ptr<float>();
				Eigen::Map<Eigen::MatrixXf> eigen_centered(centered_data, centered.numel(), 1);
				covariance += eigen_centered.transpose() * eigen_centered;
			}
			covariance /= static_cast<float>(train_output.size() - 1);
			train_outputs[i].clear();
			train_outputs[i].push_back(torch::from_blob(mean_data, { mean.size(0), 1 }));
			train_outputs[i].push_back(torch::from_blob(covariance.data(), { covariance.rows(), covariance.cols() }));
		}

		if (debug_flag) {
			vector<pair<Mat, string>> ProcessImages{

			};

			cvex::ShowProcess(ProcessImages, tp, pPara);

			switch (cv::waitKey(1)) {

			case 's':
				debug_flag = false;
				break;

			}
		}
	} while (debug_flag);
}


void MVTecDataset::runONNX() {

	bool debug_flag = run_ONNX;

	static TParas tp = []() -> TParas {

		return tp;
	}();

	if (debug_flag) {
		static bool trackbar_flag = [&]() -> bool {

			CreateWindow(1);
			Track("show", 1, tp.show, 0, NULL);

			return true;
		}();
	}

	do {
		// ONNX environment
		Ort::Env env;
		auto modelPath = L"C:/Users/000501/source/repos/Mahalanobis/model/efficientnet-b4.onnx";    //model location	
		Ort::RunOptions runOptions;
		Ort::Session session(env, modelPath, Ort::SessionOptions());

		constexpr int64_t numChannels = 3;
		constexpr int64_t width = 224;
		constexpr int64_t height = 224;
		constexpr int64_t numClasses = 1000;
		constexpr int64_t numInputElements = numChannels * height * width;

		// define shape
		const std::array<int64_t, 4> inputShape = { 1, numChannels, height, width };
		const std::array<int64_t, 2> outputShape = { 1, numClasses };

		// define array
		std::array<float, numInputElements> input;
		std::array<float, numClasses> results;

		Ort::AllocatorWithDefaultOptions allocator;

		cout << session.GetInputNameAllocated(0, allocator) << endl;
		cout << session.GetOutputNameAllocated(0, allocator) << endl;

		auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU); //memory allocation
		auto inputTensor = Ort::Value::CreateTensor<float>(memoryInfo, input.data(), input.size(), inputShape.data(), inputShape.size());
		auto outputTensor = Ort::Value::CreateTensor<float>(memoryInfo, results.data(), results.size(), outputShape.data(), outputShape.size());

		// define names
		Ort::AllocatorWithDefaultOptions ort_alloc;
		Ort::AllocatedStringPtr inputName = session.GetInputNameAllocated(0, ort_alloc);
		Ort::AllocatedStringPtr outputName = session.GetOutputNameAllocated(0, ort_alloc);
		const std::array<const char*, 1> inputNames = { inputName.get() };
		const std::array<const char*, 1> outputNames = { outputName.get() };

		inputName.release();
		outputName.release();

		// run inference
		try {
			session.Run(runOptions, inputNames.data(), &inputTensor, 1, outputNames.data(), &outputTensor, 1);
			cout << "ONNX running" << endl;
		}
		catch (Ort::Exception& e) {
			std::cout << e.what() << std::endl;
		}

		if (debug_flag) {
			vector<pair<Mat, string>> ProcessImages{

			};

			cvex::ShowProcess(ProcessImages, tp, pPara);

			switch (cv::waitKey(1)) {

			case 's':
				debug_flag = false;
				break;

			}

		}
	} while (debug_flag);
}

std::vector<vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>>> MVTecDataset::createBatch(DatasetInfo dataset) {
	PParameter pPara = PParameter(new Parameter());
	bool debug_flag = create_batch;

	static TParas tp = []() -> TParas {

		return tp;
	}();

	if (debug_flag) {
		static bool trackbar_flag = [&]() -> bool {

			CreateWindow(1);
			Track("show", 1, tp.show, 0, NULL);

			return true;
		}();
	}

	vector<vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>>> dataloader;
	do {
		int len_train = dataset.x.size();
		int batch_size = 32;
		vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>> load_dataset;

		for (int j = 0; j < len_train; j++)
		{
			std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> train_load = transformItem(dataset, j);
			load_dataset.push_back(train_load);
		}

		for (int i = 0; i < dataset.x.size(); i += batch_size)
		{
			vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>> train_batch;
			for (int k = i; k < i + batch_size && k < dataset.x.size(); k++)
			{
				train_batch.push_back(transformItem(dataset, k));
			}
			dataloader.push_back(train_batch);
		}

		if (debug_flag) {
			vector<pair<Mat, string>> ProcessImages{

			};

			cvex::ShowProcess(ProcessImages, tp, pPara);

			switch (cv::waitKey(1)) {

			case 's':
				debug_flag = false;
				break;

			}

		}
	} while (debug_flag);
	return dataloader;
}

cv::Mat MVTecDataset::transformX(std::string image) {

	PParameter pPara = PParameter(new Parameter());
	bool debug_flag = transform_X;

	static TParas tp = []() -> TParas {

		return tp;
	}();

	if (debug_flag) {
		static bool trackbar_flag = [&]() -> bool {

			CreateWindow(1);
			Track("show", 1, tp.show, 0, NULL);

			return true;
		}();
	}

	cv::Mat div;

	do {
		cv::Mat src = cv::imread(image, cv::IMREAD_COLOR);
		cv::cvtColor(src, src, cv::COLOR_BGR2RGB);
		cv::Mat resizedImage;
		cv::resize(src, resizedImage, cv::Size(resize, resize), 0, 0, cv::INTER_LANCZOS4);

		// Center crop the resized image
		int startX = (resizedImage.cols - cropsize) / 2;
		int startY = (resizedImage.rows - cropsize) / 2;
		cv::Rect cropRegion(startX, startY, cropsize, cropsize);
		cv::Mat croppedImage;
		croppedImage = resizedImage(cropRegion);
	
		// Convert the image to a 3-channel floating-point matrix
		cv::Mat cropped_float;
		croppedImage.convertTo(cropped_float, CV_32FC3, 1.0 / 255.0);

		// Normalize the image
		cv::Scalar mean = cv::Scalar(0.485, 0.456, 0.406);
		cv::Scalar stdDev = cv::Scalar(0.229, 0.224, 0.225);
		cv::Mat subs;
		cv::subtract(cropped_float, mean, subs);		
		cv::divide(subs, stdDev, div);

		if (debug_flag) {
			vector<pair<Mat, string>> ProcessImages{
				{src, "src"},
				{resizedImage, "Resize image"},
				{croppedImage, "cropped image"},
				{cropped_float,"floating matrix"},
				{subs,"images after substraction"},
				{div, "images after division"},
			};

			cvex::ShowProcess(ProcessImages, tp, pPara);

			switch (cv::waitKey(1)) {

			case 's':
				debug_flag = false;
				break;
			}
		}		
	} while (debug_flag);	

	return div;
}

cv::Mat MVTecDataset::transformMask(std::string mask){

	PParameter pPara = PParameter(new Parameter());
	bool debug_flag = transform_Mask;

	static TParas tp = []() -> TParas {

		return tp;
	}();

	if (debug_flag) {
		static bool trackbar_flag = [&]() -> bool {

			CreateWindow(1);
			Track("show", 1, tp.show, 0, NULL);

			return true;
		}();
	}

	cv::Mat convert_mask;

	do {
		// Resize the mask using the NEAREST interpolation method
		cv::Mat src = cv::imread(mask,cv::IMREAD_GRAYSCALE);

		cv::Mat resizedMask;
		cv::resize(src, resizedMask, cv::Size(this->resize, this->resize), 0, 0, cv::INTER_NEAREST);

		// Center crop the resized mask
		int startX = (resizedMask.cols - cropsize) / 2;
		int startY = (resizedMask.rows - cropsize) / 2;
		cv::Rect cropRegion(startX, startY, cropsize, cropsize);
		cv::Mat croppedMask;
		croppedMask = resizedMask(cropRegion);

		// Convert the mask to a 1-channel floating-point matrix		
		croppedMask.convertTo(convert_mask, CV_32FC1, 1.0 / 255.0);

		if (debug_flag) {
			vector<pair<Mat, string>> ProcessImages{
				{src, "src"},
				{resizedMask, "Resized image"},
				{croppedMask, "Cropped image"},
				{convert_mask,"Converted mask"},
			};

			cvex::ShowProcess(ProcessImages, tp, pPara);

			switch (cv::waitKey(1)) {

			case 's':
				debug_flag = false;
				break;

			}

		}
	} while (debug_flag);
	return convert_mask;
}

int MVTecDataset::len() {

	PParameter pPara = PParameter(new Parameter());
	bool debug_flag = length;

	static TParas tp = []() -> TParas {

		return tp;
	}();

	if (debug_flag) {
		static bool trackbar_flag = [&]() -> bool {

			CreateWindow(1);
			Track("show", 1, tp.show, 0, NULL);

			return true;
		}();
	}

	DatasetInfo data;

	do {

		

		if (debug_flag) {
			vector<pair<Mat, string>> ProcessImages{

			};

			cvex::ShowProcess(ProcessImages, tp, pPara);

			switch (cv::waitKey(1)) {

			case 's':
				debug_flag = false;
				break;

			}

		}
	} while (debug_flag);
	return data.x.size();
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> MVTecDataset::transformItem(DatasetInfo data, int idx) {

	PParameter pPara = PParameter(new Parameter());
	bool debug_flag = transform_item;

	static TParas tp = []() -> TParas {

		return tp;
	}();

	if (debug_flag) {
		static bool trackbar_flag = [&]() -> bool {

			CreateWindow(1);
			Track("show", 1, tp.show, 0, NULL);

			return true;
		}();
	}
	//DatasetInfo data;
	int y = data.y[idx];
	torch::Tensor yTensor;
	torch::Tensor xTensor;
	torch::Tensor maskTensor;
	do {
		std::string x = data.x[idx];
		
		std::string mask = data.mask[idx];

		// Apply the transformation to x
		cv::Mat transformedX = transformX(x);

		// Load and transform the mask
		cv::Mat transformedMask;
		if (y == 0) {
			transformedMask = cv::Mat::zeros(this->cropsize, this->cropsize, CV_32FC1);
		}
		else {
			transformedMask = transformMask(mask);
		}

		// Convert transformedX and transformedMask to torch::Tensor
		xTensor = torch::from_blob(transformedX.data, { 1, this->cropsize, this->cropsize }, torch::kFloat32);
		maskTensor = torch::from_blob(transformedMask.data, { 1, this->cropsize, this->cropsize }, torch::kFloat32);
		yTensor = torch::tensor(y, torch::kFloat32);
		//cout <<"Y Tensor - " << yTensor << endl;

		if (debug_flag) {
			vector<pair<Mat, string>> ProcessImages{
				{transformedX, "transformed-image"},
				{transformedMask, "transformed mask"},
			};

			cvex::ShowProcess(ProcessImages, tp, pPara);

			switch (cv::waitKey(1)) {

			case 's':
				debug_flag = false;
				break;

			}

		}
	} while (debug_flag);
	return std::make_tuple(xTensor, yTensor, maskTensor);
}

DatasetInfo MVTecDataset::loadDatasetFolder(){

	PParameter pPara = PParameter(new Parameter());
	bool debug_flag = load_dataset;

	static TParas tp = []() -> TParas {

		return tp;
	}();

	if (debug_flag) {
		static bool trackbar_flag = [&]() -> bool {

			CreateWindow(1);
			Track("show", 1, tp.show, 0, NULL);

			return true;
		}();
	}

	DatasetInfo dataset;

	do {
		std::string rootFolder = this->root_path;
		std::string className = this->class_name;
		bool isTrain = this->is_train;
		std::string path = "mvtec_anomaly_detection";
		std::string mvtecFolder = rootFolder + "/" + path;

		std::string phase = isTrain ? "train" : "test";
		
		std::string imgDir = mvtecFolder + "/" + className + "/" + phase;
		std::string gtDir = mvtecFolder + "/" + className + "/ground_truth";

		//Storing the folder under train/test
		std::vector<std::string> imgTypes;
		for (const auto& entry : std::filesystem::directory_iterator(imgDir))
		{
			if (entry.is_directory())
			{
				imgTypes.push_back(entry.path().filename().string());
			}
		}

		for (const std::string& imgType : imgTypes) {
			std::string imgTypeDir = imgDir + "/" + imgType;
			//Storing the image
			for (const auto& entry : std::filesystem::directory_iterator(imgTypeDir)) {
				if (entry.is_regular_file() && entry.path().extension() == ".png") {
					std::string imgPath = entry.path().string();
					//cout << imgPath << endl;
					cv::Mat src = imread(imgPath);
					//imshow("image", src);
					//waitKey(100);
					dataset.x.push_back(imgPath);

					if (imgType == "good") {
						dataset.y.push_back(0);
						dataset.mask.push_back("");
					}
					else {
						dataset.y.push_back(1);
						std::string imgName = entry.path().stem().string();
						std::string gtPath = gtDir + "/" + imgType + "/" + imgName + "_mask.png";
						dataset.mask.push_back(gtPath);
					}
				}
			}
		}

		// Confirm the size of x and y is equal or not.
		if (dataset.x.size() != dataset.y.size()) {
			std::cerr << "Number of x and y should be the same." << std::endl;		
		}
		//waitKey(0);

		if (debug_flag) {
			vector<pair<Mat, string>> ProcessImages{

			};

			cvex::ShowProcess(ProcessImages, tp, pPara);

			switch (cv::waitKey(1)) {

			case 's':
				debug_flag = false;
				break;

			}

		}
	} while (debug_flag);
	return dataset;
}