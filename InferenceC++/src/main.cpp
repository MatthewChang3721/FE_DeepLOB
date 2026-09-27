#include <iostream>
#include <vector>
#include <onnxruntime_cxx_api.h>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>

// 全局常量：模型输入维度配置
// Proprietary core algorithm redacted for confidentiality.

// ONNX 推理引擎封装类
class ModelEngine {
    private:
        Ort::Env env;                    // ONNX Runtime 环境
        Ort::Session session{nullptr};   // 模型会话
        Ort::MemoryInfo mem_info;        // 内存信息

        std::vector<float> mem_momentum, mem_pic, mem_lt, mem_st;          // 输入数据缓冲区
        std::vector<int64_t> shape_momentum, shape_pic, shape_lt, shape_st; // 张量形状
        std::vector<Ort::Value> input_tensors;  // ONNX 输入张量

    public:
        // 构造函数：加载 ONNX 模型并初始化输入张量
        ModelEngine(const wchar_t* model_path):
            env(ORT_LOGGING_LEVEL_WARNING, "DeepLOB_Pipeline"),
            mem_info(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)){
                
            Ort::SessionOptions session_options;
            session_options.SetIntraOpNumThreads(1);
            session = Ort::Session(env, model_path, session_options);

            // 分配输入缓冲区
            mem_momentum.resize(BATCH_SIZE * TIME_STEPS * C_MOMENTUM);
            mem_pic.resize(BATCH_SIZE * TIME_STEPS * C_PIC);
            mem_lt.resize(BATCH_SIZE * TIME_STEPS * C_LT);
            mem_st.resize(BATCH_SIZE * TIME_STEPS * C_ST);

            shape_momentum = {BATCH_SIZE, TIME_STEPS, C_MOMENTUM};
            shape_pic = {BATCH_SIZE, TIME_STEPS, C_PIC};
            shape_lt = {BATCH_SIZE, TIME_STEPS, C_LT};
            shape_st = {BATCH_SIZE, TIME_STEPS, C_ST};

            // 创建 ONNX 输入张量
            input_tensors.push_back(Ort::Value::CreateTensor<float>(mem_info, mem_momentum.data(), mem_momentum.size(), shape_momentum.data(), shape_momentum.size()));
            input_tensors.push_back(Ort::Value::CreateTensor<float>(mem_info, mem_pic.data(), mem_pic.size(), shape_pic.data(), shape_pic.size()));
            input_tensors.push_back(Ort::Value::CreateTensor<float>(mem_info, mem_lt.data(), mem_lt.size(), shape_lt.data(), shape_lt.size()));
            input_tensors.push_back(Ort::Value::CreateTensor<float>(mem_info, mem_st.data(), mem_st.size(), shape_st.data(), shape_st.size()));

            std::cout << "--- [Engine] Loaded Successfully ---" << std::endl;
            }
        // 前向推理接口：输入四个特征，返回三个概率值
        std::vector<float> forward(
            const std::vector<float>& in_momentum, 
            const std::vector<float>& in_pic,
            const std::vector<float>& in_lt,
            const std::vector<float>& in_st){
                // 拷贝输入数据到缓冲区
                std::copy(in_momentum.begin(), in_momentum.end(), mem_momentum.begin());
                std::copy(in_pic.begin(), in_pic.end(), mem_pic.begin());
                std::copy(in_lt.begin(), in_lt.end(), mem_lt.begin());
                std::copy(in_st.begin(), in_st.end(), mem_st.begin());

                // 执行 ONNX 模型推理
                const char* input_names[] = {"momentum", "pic", "lt_sensor", "st_sensor"};
                const char* output_names[] = {"logits"};
                auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names, input_tensors.data(), 4, output_names, 1);

                // 提取输出结果
                float* result_ptr = output_tensors.front().GetTensorMutableData<float>();
                return std::vector<float>{result_ptr[0], result_ptr[1], result_ptr[2]};
                }
    };

bool load_csv_data(
    const std::string& file_path,
    std::vector<float>& out_momentum,
    std::vector<float>& out_pic, 
    std::vector<float>& out_lt,
    std::vector<float>& out_st
) {
    std::ifstream file(file_path);
    if (!file.is_open()){
        std::cerr << "Fail to Open file" << file_path << std::endl;
        return false;
    }
    std::string line;
    int current_time_step = 0;

    while (std::getline(file, line) && current_time_step < 150) {
        std::stringstream ss(line);
        std::string cell;
        
        // Proprietary core algorithm redacted for confidentiality.
    }
    if (current_time_step < 150) {
        std::cerr << "Lacking of Data" << std::endl;
        return false;
    }
    return true;
}

int main(){
    try {
        // 初始化推理引擎
        ModelEngine Inference_engine(L"models/deeplob_v2.onnx");

        // 从 CSV 加载特征数据
        std::vector<float> real_momentum(BATCH_SIZE * TIME_STEPS * C_MOMENTUM);
        std::vector<float> real_pic(BATCH_SIZE * TIME_STEPS * C_PIC);
        std::vector<float> real_lt(BATCH_SIZE * TIME_STEPS * C_LT);
        std::vector<float> real_st(BATCH_SIZE * TIME_STEPS * C_ST);
        if (!load_csv_data("data/test_data.csv", real_momentum, real_pic, real_lt, real_st)) {
            return -1;
        }

        // 预热：防止首次推理的冷启动开销
        std::cout << "Warmup..." << std::endl;
        for (int i = 0; i < 50; ++i) {
            Inference_engine.forward(real_momentum, real_pic, real_lt, real_st);
        }

        // 压力测试：连续执行一万次推理
        const int NUM_ITERATIONS = 10000;
        std::cout << "Push to Limit: " << NUM_ITERATIONS << std::endl;

        auto start_time = std::chrono::high_resolution_clock::now();  // 开始计时

        for (int i = 0; i < NUM_ITERATIONS; ++i) {
            Inference_engine.forward(real_momentum, real_pic, real_lt, real_st);
        }

        auto end_time = std::chrono::high_resolution_clock::now();  // 结束计时

        // 计算总耗时与平均延迟
        auto total_microseconds = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
        double avg_latency = static_cast<double>(total_microseconds) / NUM_ITERATIONS;

        std::cout << "---------------------------------" << std::endl;
        std::cout << "Total Time Consumed: " << total_microseconds / 1000.0 << " ms" << std::endl;
        std::cout << "Single Calculation: " << avg_latency << " us" << std::endl;
        std::cout << "---------------------------------" << std::endl;

    } catch (const Ort::Exception& e){
        std::cerr << "Loading Wrong:" << e.what() << std::endl;
        return -1;
    }
    return 0; 
};
