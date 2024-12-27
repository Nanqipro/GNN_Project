import torch
import torch_geometric
from torch_geometric.data import Data

# 创建一个简单的图数据实例
edge_index = torch.tensor([[0, 1, 1, 2],
                           [1, 0, 2, 1]], dtype=torch.long)  # 边列表
x = torch.tensor([[-1], [0], [1]], dtype=torch.float)  # 节点特征

# 创建图数据对象
data = Data(x=x, edge_index=edge_index)

# 打印图数据
print(data)

# 检查是否能够使用CUDA
if torch.cuda.is_available():
    print("CUDA is available!")
    data = data.to('cuda')
else:
    print("CUDA is not available!")

