# train.py

import os
import requests
import tarfile
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch_geometric.data import InMemoryDataset, Data, DataLoader
from torch_geometric.nn import TopKPooling, SAGEConv, global_mean_pool
import matplotlib.pyplot as plt
import seaborn as sns

# 设置数据集下载链接（请根据实际情况修改）
CLICK_DATA_URL = 'https://example.com/path/to/yoochoose-clicks.dat'  # 替换为实际下载链接
BUY_DATA_URL = 'https://example.com/path/to/yoochoose-buys.dat'      # 替换为实际下载链接

def download_file(url, dest_path):
    """下载文件并保存到指定路径"""
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        total_size = int(response.headers.get('content-length', 0))
        with open(dest_path, 'wb') as f, tqdm(
            desc=dest_path,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                bar.update(size)
    else:
        raise Exception(f"无法下载文件，状态码: {response.status_code}")

# 数据准备
def load_and_preprocess_data(clicks_path, buys_path, download=False):
    """加载并预处理点击和购买数据"""
    # 检查数据文件是否存在，如果不存在且允许下载，则下载数据
    if not os.path.exists(clicks_path) or not os.path.exists(buys_path):
        if download:
            print("数据文件不存在，开始下载数据集...")
            # 创建存储目录
            os.makedirs(os.path.dirname(clicks_path), exist_ok=True)
            # 下载点击数据
            print(f"下载点击数据到 {clicks_path}...")
            download_file(CLICK_DATA_URL, clicks_path)
            # 下载购买数据
            print(f"下载购买数据到 {buys_path}...")
            download_file(BUY_DATA_URL, buys_path)
            print("数据下载完成。")
        else:
            raise FileNotFoundError(
                f"数据文件未找到: {clicks_path} 或 {buys_path}。请设置 download=True 以自动下载。"
            )
    
    # 读取点击数据
    print("读取点击数据...")
    df_clicks = pd.read_csv(clicks_path, header=None)
    df_clicks.columns = ['session_id', 'timestamp', 'item_id', 'category']

    # 读取购买数据
    print("读取购买数据...")
    df_buys = pd.read_csv(buys_path, header=None)
    df_buys.columns = ['session_id', 'timestamp', 'item_id', 'price', 'quantity']

    # 对item_id进行编码
    print("编码 item_id...")
    item_encoder = LabelEncoder()
    df_clicks['item_id'] = item_encoder.fit_transform(df_clicks['item_id'])
    df_buys['item_id'] = item_encoder.transform(df_buys['item_id'])

    # 标记是否有购买
    print("标记是否有购买...")
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
        # 不需要额外的原始文件
        return []

    @property
    def processed_file_names(self):
        return ['yoochoose_full_dataset.dataset']

    def download(self):
        # 下载逻辑已在 load_and_preprocess_data 中处理
        pass

    def process(self):
        data_list = []
        grouped = self.df.groupby('session_id')

        print("处理会话数据并构建图结构...")
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
        print("保存处理后的数据...")
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
    all_preds = []
    all_labels = []

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        label = data.y.to(device)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs

        # 收集预测和标签用于计算训练准确率
        preds = (output > 0.5).float()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(label.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)

    return avg_loss, accuracy

# 评估函数
def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    total_loss = 0
    criterion = torch.nn.BCELoss()

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            label = data.y.to(device)
            loss = criterion(output, label)
            total_loss += loss.item() * data.num_graphs

            probs = output.cpu().numpy()
            preds = (output > 0.5).float().cpu().numpy()

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(label.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    roc_auc = roc_auc_score(all_labels, all_probs)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    metrics = {
        'loss': avg_loss,
        'roc_auc': roc_auc,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm
    }

    return metrics, all_labels, all_probs, all_preds

# 主函数
def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 创建结果保存目录
    results_dir = './results'
    os.makedirs(results_dir, exist_ok=True)

    # 数据路径
    clicks_path = './yoochoose-clicks.dat'  # 请根据实际情况修改路径
    buys_path = './yoochoose-buys.dat'      # 请根据实际情况修改路径

    # 加载和预处理数据，设置 download=True 以自动下载数据
    try:
        df_clicks, df_buys, item_encoder = load_and_preprocess_data(clicks_path, buys_path, download=True)
    except Exception as e:
        print(f"数据加载失败: {e}")
        return

    # 创建数据集
    print("创建自定义数据集...")
    dataset = YooChooseBinaryDataset(root='data/', df=df_clicks)
    num_items = df_clicks['item_id'].nunique()
    print(f"唯一 item 数量: {num_items}")

    # 划分训练集和验证集（例如 80% 训练，20% 验证）
    torch.manual_seed(42)
    print("打乱数据集并划分训练集和验证集...")
    dataset = dataset.shuffle()
    train_size = int(0.8 * len(dataset))
    train_dataset = dataset[:train_size]
    val_dataset = dataset[train_size:]
    print(f"训练集大小: {len(train_dataset)}，验证集大小: {len(val_dataset)}")

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # 初始化模型、优化器和损失函数
    print("初始化模型、优化器和损失函数...")
    model = GNNModel(num_items=num_items).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCELoss()

    # 训练模型
    epochs = 10
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1_score': [],
        'val_roc_auc': []
    }
    all_val_labels = []
    all_val_probs = []
    all_val_preds = []

    print("开始训练模型...")
    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = train(model, train_loader, optimizer, criterion, device)
        val_metrics, val_labels, val_probs, val_preds = evaluate(model, val_loader, device)

        history['train_loss'].append(train_loss)
        history['train_accuracy'].append(train_accuracy)
        history['val_loss'].append(val_metrics['loss'])
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['val_precision'].append(val_metrics['precision'])
        history['val_recall'].append(val_metrics['recall'])
        history['val_f1_score'].append(val_metrics['f1_score'])
        history['val_roc_auc'].append(val_metrics['roc_auc'])

        all_val_labels.extend(val_labels)
        all_val_probs.extend(val_probs)
        all_val_preds.extend(val_preds)

        print(f'Epoch {epoch}/{epochs}, '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, '
              f'Val Loss: {val_metrics["loss"]:.4f}, Val Acc: {val_metrics["accuracy"]:.4f}, '
              f'Val ROC AUC: {val_metrics["roc_auc"]:.4f}, Val Precision: {val_metrics["precision"]:.4f}, '
              f'Val Recall: {val_metrics["recall"]:.4f}, Val F1 Score: {val_metrics["f1_score"]:.4f}')

    # 最终评估
    final_metrics, final_labels, final_probs, final_preds = evaluate(model, val_loader, device)
    print(f'最终验证集 ROC AUC: {final_metrics["roc_auc"]:.4f}')

    # 可视化
    print("开始可视化训练过程...")

    # 1. 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), history['train_loss'], label='Train Loss')
    plt.plot(range(1, epochs + 1), history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, 'loss_curve.png'))
    plt.close()

    # 2. 绘制准确率曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), history['train_accuracy'], label='Train Accuracy')
    plt.plot(range(1, epochs + 1), history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, 'accuracy_curve.png'))
    plt.close()

    # 3. 绘制ROC曲线
    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(all_val_labels, all_val_probs)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {final_metrics["roc_auc"]:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, 'roc_curve.png'))
    plt.close()

    # 4. 绘制混淆矩阵
    cm = confusion_matrix(all_val_labels, all_val_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Buy', 'Buy'], yticklabels=['Not Buy', 'Buy'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(results_dir, 'confusion_matrix.png'))
    plt.close()

    print(f"所有可视化图像已保存在 '{results_dir}' 文件夹下。")

if __name__ == '__main__':
    main()
