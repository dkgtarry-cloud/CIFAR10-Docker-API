## 项目概览
本项目是一个基于 Docker 的 CIFAR-10 图像分类推理 API。使用 Flask 构建推理服务，调用 PyTorch 训练的 CNN 模型进行图像分类，并通过 Docker 实现容器化部署，能够快速完成端到端的模型推理流程。
这是我完成的第二个完整 MLOps 实践项目，涵盖了从 模型训练、权重导出、API 构建、镜像制作到容器部署与测试 的完整流程。进一步深化了对 CNN 架构与模型推理部署流程 的理解。

## 项目文件和依赖
Python  
Flask  
PyTorch  
Docker

## 安装与构建
构建 Docker 镜像：
```bash
docker build -t cifar-api:v4 .
```
运行 Docker 容器：
```bash
docker run -d --gpus all -p 8000:5000 cifar-api:v4
```
API 使用
使用 curl 命令发送图片进行预测：
```bash
curl.exe -X POST -F "file=@test_image.png" http://localhost:8000/predict
```
你将收到一个 JSON 响应，包含预测结果：
```bash
{"prediction": "bird"}
```
## 测试截图
<img width="865" height="391" alt="image" src="https://github.com/user-attachments/assets/885b6eeb-6fbb-4ce8-8ed1-5f8ac917ba4d" />
<img width="865" height="129" alt="image" src="https://github.com/user-attachments/assets/51c61058-63e9-4b79-9ef7-a86ac7865575" />
<img width="865" height="66" alt="image" src="https://github.com/user-attachments/assets/ac0e35f6-2586-4119-9b5b-d526097e2102" />


## 遇到的问题与解决
问题 1：DataLoader 报错 - RuntimeError: An attempt has been made to start a new process...
原因： Windows 下使用多进程加载数据时未添加主函数保护。
解决：
```bash
if __name__ == '__main__':
    main()
```
问题 2：Docker 构建镜像失败（网络被远程主机强制关闭）
报错：
```bash
failed to fetch anonymous token: read tcp ... wsarecv: An existing connection was forcibly closed by the remote host.
```
原因： 拉取 pytorch/pytorch 镜像时网络中断。
解决：
```bash
docker pull pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime
docker build -t cifar-api:v1 .
```
问题 3：RuntimeError: Attempting to deserialize object on a CUDA device but torch.cuda.is_available() is False

原因： 模型在 GPU 环境中训练，而容器运行时默认仅启用 CPU，导致加载模型权重时设备不匹配。
解决方案：
通过在启动容器时显式指定 GPU 资源，确保容器具备 CUDA 环境支持：
```bash
docker run -d --gpus all -p 8000:5000 cifar-api:v4
```

