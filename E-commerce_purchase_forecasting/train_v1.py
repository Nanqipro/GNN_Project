# 导入必要的库
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch_geometric.data import InMemoryDataset, Data, DataLoader
from torch_geometric.nn import TopKPooling, SAGEConv, global_mean_pool

# 数据准备
def load_and_preprocess_data(clicks_path, buys_path):
    # 读取点击数据
    df_clicks = pd.read_csv(clicks_path, header=None)
    df_clicks.columns = ['session_id', 'timestamp', 'item_id', 'category']

    # 读取购买数据
    df_buys = pd.read_csv(buys_path, header=None)
    df_buys.columns = ['session_id', 'timestamp', 'item_id', 'price', 'quantity']

    # 对item_id进行编码
    item_encoder = LabelEncoder()
    df_clicks['item_id'] = item_encoder.fit_transform(df_clicks['item_id'])
    df_buys['item_id'] = item_encoder.transform(df_buys['item_id'])

    # 标记是否有购买
    df_clicks['label'] = df_clicks['session_id'].isin(df_buys['session_id']).astype(int)

    return df_clicks, df_buys, item_encoder

# 定义自定义数据集
class YooChooseBinaryDataset(InMemoryDataset):
    def __init__(self, root, df, transform=None, pre_transform=None):
        self.df = df
        super(YooChooseBinaryDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['yoochoose_full_dataset.dataset']

    def download(self):
        # 如果需要下载数据，可以在这里实现
        pass

    def process(self):
        data_list = []
        grouped = self.df.groupby('session_id')

        for session_id, group in tqdm(grouped, desc="Processing sessions"):
            sess_item_id = LabelEncoder().fit_transform(group['item_id'])
            group = group.reset_index(drop=True)
            group['sess_item_id'] = sess_item_id

            # 节点特征
            node_features = group[['sess_item_id', 'item_id']].drop_duplicates().sort_values('sess_item_id')['item_id'].values
            x = torch.tensor(node_features, dtype=torch.long).unsqueeze(1)

            # 边索引
            target_nodes = group['sess_item_id'].values[1:]
            source_nodes = group['sess_item_id'].values[:-1]
            edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)

            # 标签
            y = torch.tensor([group['label'].values[0]], dtype=torch.float)

            # 创建图数据
            data = Data(x=x, edge_index=edge_index, y=y)
            data_list.append(data)

        # 保存处理后的数据
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

# 定义图神经网络模型
class GNNModel(torch.nn.Module):
    def __init__(self, num_items, embed_dim=128):
        super(GNNModel, self).__init__()
        self.embed_dim = embed_dim
        self.item_embedding = torch.nn.Embedding(num_embeddings=num_items + 10, embedding_dim=embed_dim)

        self.conv1 = SAGEConv(embed_dim, 128)
        self.pool1 = TopKPooling(128, ratio=0.8)
        self.conv2 = SAGEConv(128, 128)
        self.pool2 = TopKPooling(128, ratio=0.8)
        self.conv3 = SAGEConv(128, 128)
        self.pool3 = TopKPooling(128, ratio=0.8)

        self.lin1 = torch.nn.Linear(128, 128)
        self.lin2 = torch.nn.Linear(128, 64)
        self.lin3 = torch.nn.Linear(64, 1)

        self.bn1 = torch.nn.BatchNorm1d(128)
        self.bn2 = torch.nn.BatchNorm1d(64)

        self.act1 = torch.nn.ReLU()
        self.act2 = torch.nn.ReLU()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # 嵌入层
        x = self.item_embedding(x).squeeze(1)  # [n, embed_dim]

        # 第一层卷积和池化
        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        x1 = global_mean_pool(x, batch)

        # 第二层卷积和池化
        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        x2 = global_mean_pool(x, batch)

        # 第三层卷积和池化
        x = F.relu(self.conv3(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
        x3 = global_mean_pool(x, batch)

        # 合并不同层的全局特征
        x = x1 + x2 + x3

        # 全连接层
        x = self.lin1(x)
        x = self.act1(x)
        x = self.lin2(x)
        x = self.act2(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = torch.sigmoid(self.lin3(x)).squeeze(1)

        return x

# 训练函数
def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        label = data.y.to(device)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs

    return total_loss / len(loader.dataset)

# 评估函数
def evaluate(model, loader, device):
    model.eval()
    predictions = []
    labels = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            pred = model(data)
            predictions.append(pred.cpu().numpy())
            labels.append(data.y.cpu().numpy())

    predictions = np.hstack(predictions)
    labels = np.hstack(labels)
    return roc_auc_score(labels, predictions)

# 主函数
def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据路径
    clicks_path = 'yoochoose-clicks.dat'
    buys_path = 'yoochoose-buys.dat'

    # 加载和预处理数据
    df_clicks, df_buys, item_encoder = load_and_preprocess_data(clicks_path, buys_path)

    # 创建数据集
    dataset = YooChooseBinaryDataset(root='./data/', df=df_clicks)
    num_items = df_clicks['item_id'].nunique()

    # 划分训练集和验证集（例如 80% 训练，20% 验证）
    torch.manual_seed(42)
    dataset = dataset.shuffle()
    train_size = int(0.8 * len(dataset))
    train_dataset = dataset[:train_size]
    val_dataset = dataset[train_size:]

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # 初始化模型、优化器和损失函数
    model = GNNModel(num_items=num_items).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCELoss()

    # 训练模型
    epochs = 10
    for epoch in range(1, epochs + 1):
        loss = train(model, train_loader, optimizer, criterion, device)
        val_auc = evaluate(model, val_loader, device)
        print(f'Epoch {epoch}/{epochs}, Loss: {loss:.4f}, Validation ROC AUC: {val_auc:.4f}')

    # 最终评估
    final_auc = evaluate(model, val_loader, device)
    print(f'最终验证集 ROC AUC: {final_auc:.4f}')

if __name__ == '__main__':
    main()