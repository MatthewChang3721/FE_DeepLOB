import os
from datetime import datetime

import torch
import onnx
from FE_DeepLOB import FE_DeepLOB
import config


def export_to_onnx(pt_filename: str):
    print("1. Loading original model...")
    model = FE_DeepLOB(num_features=5, num_classes=3)
    
    pt_path = os.path.join("Model/torch", pt_filename)
    state_dict = torch.load(pt_path, map_location="cpu")
    model.load_state_dict(state_dict)
    
    model.eval()

    print("2. Generating Dummy Data ...")

    batch_size = 1
    T = 150
    
    # 按照 train_engine 中的顺序构造 4 个输入
    # 注意：这里的 C_m, C_p, C_l, C_s 需要替换为你真实数据各模态的通道数
    # dummy_inputs

    date_str = datetime.now().strftime("%Y%m%d")
    alpha_bps = config.alpha * 1e4
    alpha_str = f"{alpha_bps:g}bps"
    onnx_file_name = f"{date_str}_{config.label_window}_{alpha_str}_{config.window_size}.onnx"
    onnx_file_path = os.path.join("Model/onnx", onnx_file_name)
    os.makedirs("Model/onnx", exist_ok=True)

    print("3. Export ONNX Calculation Picture...")
    
    torch.onnx.export(
        model,                      # 要转换的模型
        dummy_inputs,               # 假数据元组
        onnx_file_path,             # 导出的文件名
        export_params=True,         # 将训练好的权重一并保存在文件内
        opset_version=19,           # ONNX 算子集版本
        do_constant_folding=True,   # 开启常量折叠优化（把能提前算好的常数直接算好，提升 C++ 速度）
        
        # 定义输入和输出的节点名称 (C++ 代码中需要通过这些字符串名字来填入数据)
        input_names=[],
        output_names=['logits'],
    )
    print(f"Done! Onnx Model Saved to -> {onnx_file_path}")

    print("4. Validating File Integrity ...")
    onnx_model = onnx.load(onnx_file_path)
    onnx.checker.check_model(onnx_model)
    print("Done! Integrity -- Passed -- ")

if __name__ == "__main__":
    pt_filename = "20260629_10_1bps_150.pt"  # ← 修改为实际的 .pt 文件名
    export_to_onnx(pt_filename)
