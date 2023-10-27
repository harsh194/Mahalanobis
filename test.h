#pragma once
#include <Parameter.h>

struct DatasetInfo{
    std::vector<std::string> x;
    std::vector<int> y;
    std::vector<std::string> mask;
};

class MVTecDataset :cvex::Func{
public:
    MVTecDataset(const std::string& root_path, const std::string& class_name, bool is_train, int resize, int cropsize);

    // Define Function
    void start();
    DatasetInfo loadDatasetFolder();
    cv::Mat transformX(std::string image);
    cv::Mat transformMask(std::string mask);
    int len();
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> transformItem(DatasetInfo data, int idx);
    std::vector<vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>>> createBatch(DatasetInfo dataset);
    void runONNX();
    void calcMeanCovariance(std::vector<std::vector<torch::Tensor>>& train_outputs);
    void writeFeatures(std::filesystem::path train_feat_filepath, std::vector<std::vector<torch::Tensor>> train_outputs);
    void readFeatures(std::filesystem::path train_feat_filepath, std::vector<std::vector<torch::Tensor>>& train_outputs);

protected:
    bool start_operation = 0;
    bool transform_X = 0;
    bool transform_Mask = 0;
    bool load_dataset = 0;
    bool length = 0;
    bool transform_item = 0;
    bool create_batch = 0;
    bool run_ONNX = 0;
    bool calc_mean_covariance = 0;
    bool write_features = 0;
    bool read_features = 0;

private:
    std::string root_path;
    std::string class_name;
    bool is_train;
    int resize;
    int cropsize;
    //std::vector<std::string> x;  // Paths to image files
    //std::vector<int> y;         // Binary labels (0(good) or 1(defect))
    //std::vector<std::string> mask;  // Paths to mask images
};






