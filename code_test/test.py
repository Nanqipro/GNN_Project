# import torch
# import torch_geometric
# from torch_geometric.data import Data
# import os

# desired_gpu_id = 1  # 设置为所需的显卡编号，例如0

# # 指定使用的显卡，例如使用第0块显卡
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# print(f"可用的显卡: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
# print(f"当前使用的设备: {torch.cuda.current_device()}")
# print(f"设备名称: {torch.cuda.get_device_name(torch.cuda.current_device())}")

# device = torch.device(f'cuda:{desired_gpu_id}' if torch.cuda.is_available() else 'cpu')


# # 创建一个简单的图数据实例
# edge_index = torch.tensor([[0, 1, 1, 2],
#                            [1, 0, 2, 1]], dtype=torch.long)  # 边列表
# x = torch.tensor([[-1], [0], [1]], dtype=torch.float)  # 节点特征

# # 创建图数据对象
# data = Data(x=x, edge_index=edge_index)

# # 打印图数据
# print(data)

# # 检查是否能够使用CUDA
# if torch.cuda.is_available():
#     print("CUDA is available!")
#     data = data.to('cuda')
# else:
#     print("CUDA is not available!")



import torch

# 检查显卡是否可用并指定显卡
desired_gpu_id = 0  # 设置为所需的显卡编号，例如0
device = torch.device(f'cuda:{desired_gpu_id}' if torch.cuda.is_available() else 'cpu')


print(f"是否支持CUDA: {torch.cuda.is_available()}")
print(f"总显卡数: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"显卡{i}名称: {torch.cuda.get_device_name(i)}")

# 打印设备信息
print(f"可用的显卡: {desired_gpu_id}")
print(f"当前使用的设备: {torch.cuda.current_device()}")
print(f"设备名称: {torch.cuda.get_device_name(torch.cuda.current_device())}")