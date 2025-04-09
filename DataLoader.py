import ast
import math
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import pandas as pd


# 加载问题数据
problem_df = pd.read_csv('problem.csv')
# 加载问题标签数据
problem_tag_df = pd.read_csv('problem_tag.csv')
# 加载标签数据
tags_df = pd.read_csv('problem_tags.csv')
# 提取问题的特征，例如难度级别、描述等
problem_features = problem_df[['difficulty', 'time_limit', 'memory_limit']].values

# 将问题与标签关联，提取每个问题的知识点标签
problem_tags = pd.merge(problem_df, problem_tag_df, left_on='id', right_on='problem_id')
problem_tags = pd.merge(problem_tags, tags_df, left_on='problemtag_id', right_on='id')
# 为每个问题创建一个包含所有知识点标签的特征向量
# 假设每个问题最多有5个知识点标签
max_tags_per_problem = 5
tag_features = np.zeros((len(problem_df), max_tags_per_problem), dtype=np.longlong)

for i, problem_id in enumerate(problem_df['id']):
    tags = problem_tags[problem_tags['problem_id'] == problem_id]['name'].values[:max_tags_per_problem]
    tag_features[i, :len(tags)] = tags

# 将知识点标签特征与问题特征合并
problem_features = np.hstack([problem_features, tag_features])

# 保存为numpy文件
np.save('node_features.npy', problem_features)

# 加载提交数据
submission_df = pd.read_csv('submission.csv')
# 提取边的特征，例如结果、时间戳等
submission_features = submission_df[['result', 'create_time']].values
# 保存为numpy文件
np.save('edge_features.npy', submission_features)


# 这个类用来封装一个索引列表，它继承自Pytorch的Dataset类
class CustomizedDataset(Dataset):
    def __init__(self, indices_list: list):
        """
        Customized dataset.
        :param indices_list: list, list of indices
        """
        super(CustomizedDataset, self).__init__()

        self.indices_list = indices_list

    def __getitem__(self, idx: int):
        """
        get item at the index in self.indices_list
        :param idx: int, the index
        :return:
        """
        return self.indices_list[idx]

    def __len__(self):
        return len(self.indices_list)

# 生成一个数据加载器，批量加载数据  indices_list表示索引列表
def get_idx_data_loader(indices_list: list, batch_size: int, shuffle: bool):
    """
    get data loader that iterates over indices
    :param indices_list: list, list of indices
    :param batch_size: int, batch size
    :param shuffle: boolean, whether to shuffle the data
    :return: data_loader, DataLoader
    """
    dataset = CustomizedDataset(indices_list=indices_list)

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             drop_last=False)
    return data_loader



class Data:
# 存储节点的交互信息
    def __init__(self,src_node_ids: np.ndarray, dst_node_ids: np.ndarray, node_interact_times: np.ndarray, edge_ids: np.ndarray, labels: np.ndarray):
        """
        Data object to store the nodes interaction information.
        :param src_node_ids: ndarray
        :param dst_node_ids: ndarray
        :param node_interact_times: ndarray
        :param edge_ids: ndarray
        :param labels: ndarray
        """
        self.src_node_ids = src_node_ids
        self.dst_node_ids = dst_node_ids
        self.node_interact_times = node_interact_times
        self.edge_ids = edge_ids
        self.labels = labels
        self.num_interactions = len(src_node_ids)
        self.unique_node_ids = set(src_node_ids) | set(dst_node_ids)
        self.num_unique_nodes = len(self.unique_node_ids)
# 生成链接预测任务的数据
def get_link_prediction_data(dataset_name: str, val_ratio: float, test_ratio: float):
    """
    生成链接预测任务的数据（归纳和演绎设置）。
    :param dataset_name: str, 数据集名称
    :param val_ratio: float, 验证数据比例
    :param test_ratio: float, 测试数据比例
    :return: 节点原始特征、边原始特征、完整数据、训练数据、验证数据、测试数据、新节点验证数据、新节点测试数据
    """
    # 加载submission.csv文件，包含用户提交记录
    submission_df = pd.read_csv('submission.csv')

    # 提取交互信息，包括用户ID、问题ID、交互时间戳、结果（作为标签）
    # 生成边ID（假设submission.csv中没有边ID，这里使用enumerate生成唯一ID）
    submission_df['edge_id'] = np.arange(len(submission_df))

    src_node_ids = submission_df['user_id'].values.astype(np.longlong)  # 源节点ID（用户ID）
    dst_node_ids = submission_df['problem_id'].values.astype(np.longlong)  # 目标节点ID（问题ID）
    node_interact_times = pd.to_datetime(submission_df['create_time']).astype(np.float64) // 10**9  # 交互时间戳
    edge_ids = submission_df['edge_id'].values.astype(np.longlong)  # 边ID
    labels = submission_df['result'].values  # 标签（结果）

    # 创建完整数据对象
    full_data = Data(src_node_ids=src_node_ids, dst_node_ids=dst_node_ids, node_interact_times=node_interact_times, edge_ids=edge_ids, labels=labels)

    # 获取验证集和测试集的时间戳分位数
    val_time, test_time = list(np.quantile(node_interact_times, [(1 - val_ratio - test_ratio), (1 - test_ratio)]))

    # 设置随机种子以确保结果可复现
    random.seed(2020)

    # 获取所有唯一节点的集合
    node_set = set(src_node_ids) | set(dst_node_ids)
    num_total_unique_node_ids = len(node_set)

    # 计算测试时间之后出现的节点
    test_node_set = set(src_node_ids[node_interact_times > val_time]).union(set(dst_node_ids[node_interact_times > val_time]))
    # 从测试节点中随机选择10%的节点作为新测试节点
    new_test_node_set = set(random.sample(sorted(test_node_set), int(0.1 * num_total_unique_node_ids)))

    # 生成源节点和目标节点是否为新测试节点的掩码
    new_test_source_mask = np.array([x in new_test_node_set for x in src_node_ids])
    new_test_destination_mask = np.array([x in new_test_node_set for x in dst_node_ids])

    # 生成掩码，表示源节点和目标节点都不是新测试节点的边
    observed_edges_mask = np.logical_and(~new_test_source_mask, ~new_test_destination_mask)

    # 训练数据掩码：交互时间在验证时间之前且不涉及新测试节点的边
    train_mask = np.logical_and(node_interact_times <= val_time, observed_edges_mask)

    # 创建训练数据对象
    train_data = Data(src_node_ids=src_node_ids[train_mask], dst_node_ids=dst_node_ids[train_mask],
                      node_interact_times=node_interact_times[train_mask],
                      edge_ids=edge_ids[train_mask], labels=labels[train_mask])

    # 训练节点集合
    train_node_set = set(train_data.src_node_ids).union(train_data.dst_node_ids)
    # 确保新测试节点不在训练集中
    assert len(train_node_set & new_test_node_set) == 0
    # 新节点集合（不在训练集中）
    new_node_set = node_set - train_node_set

    # 验证集和测试集掩码
    val_mask = np.logical_and(node_interact_times <= test_time, node_interact_times > val_time)
    test_mask = node_interact_times > test_time

    # 新节点验证集和测试集掩码
    edge_contains_new_node_mask = np.array([(src_node_id in new_node_set or dst_node_id in new_node_set)
                                           for src_node_id, dst_node_id in zip(src_node_ids, dst_node_ids)])
    new_node_val_mask = np.logical_and(val_mask, edge_contains_new_node_mask)
    new_node_test_mask = np.logical_and(test_mask, edge_contains_new_node_mask)

    # 创建验证集和测试集数据对象
    val_data = Data(src_node_ids=src_node_ids[val_mask], dst_node_ids=dst_node_ids[val_mask],
                    node_interact_times=node_interact_times[val_mask], edge_ids=edge_ids[val_mask], labels=labels[val_mask])

    test_data = Data(src_node_ids=src_node_ids[test_mask], dst_node_ids=dst_node_ids[test_mask],
                     node_interact_times=node_interact_times[test_mask], edge_ids=edge_ids[test_mask], labels=labels[test_mask])

    # 创建包含新节点的验证集和测试集数据对象
    new_node_val_data = Data(src_node_ids=src_node_ids[new_node_val_mask], dst_node_ids=dst_node_ids[new_node_val_mask],
                             node_interact_times=node_interact_times[new_node_val_mask],
                             edge_ids=edge_ids[new_node_val_mask], labels=labels[new_node_val_mask])

    new_node_test_data = Data(src_node_ids=src_node_ids[new_node_test_mask], dst_node_ids=dst_node_ids[new_node_test_mask],
                              node_interact_times=node_interact_times[new_node_test_mask],
                              edge_ids=edge_ids[new_node_test_mask], labels=labels[new_node_test_mask])

    # 打印数据集统计信息
    print("数据集包含 {} 交互，涉及 {} 个不同节点".format(full_data.num_interactions, full_data.num_unique_nodes))
    print("训练数据集包含 {} 交互，涉及 {} 个不同节点".format(
        train_data.num_interactions, train_data.num_unique_nodes))
    print("验证数据集包含 {} 交互，涉及 {} 个不同节点".format(
        val_data.num_interactions, val_data.num_unique_nodes))
    print("测试数据集包含 {} 交互，涉及 {} 个不同节点".format(
        test_data.num_interactions, test_data.num_unique_nodes))
    print("新节点验证数据集包含 {} 交互，涉及 {} 个不同节点".format(
        new_node_val_data.num_interactions, new_node_val_data.num_unique_nodes))
    print("新节点测试数据集包含 {} 交互，涉及 {} 个不同节点".format(
        new_node_test_data.num_interactions, new_node_test_data.num_unique_nodes))
    print("用于归纳测试的节点数量：{}".format(len(new_test_node_set)))

    # 假设节点和边的特征已经预处理并保存
    node_raw_features = np.load('node_features.npy', allow_pickle=True)
    edge_raw_features = np.load('edge_features.npy', allow_pickle=True)

    return node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data

# 数据集划分（节点分类任务）
def get_node_classification_data(dataset_name: str, val_ratio: float, test_ratio: float):
    """
    生成节点分类任务的数据。
    :param dataset_name: str, 数据集名称
    :param val_ratio: float, 验证数据比例
    :param test_ratio: float, 测试数据比例
    :return: 节点特征、边特征、完整数据、训练数据、验证数据、测试数据
    """
    # 加载问题数据
    problem_df = pd.read_csv('problem.csv')
    # 加载问题标签数据
    problem_tag_df = pd.read_csv('problem_tag.csv')
    # 加载标签数据
    tags_df = pd.read_csv('problem_tags.csv')

    # 提取问题的特征，包括难度级别、描述、时间限制、内存限制和知识点标签
    problem_features = problem_df[['difficulty', 'time_limit', 'memory_limit']].values

    # 将问题与标签关联，提取每个问题的知识点标签
    problem_tags = pd.merge(problem_df, problem_tag_df, left_on='id', right_on='problem_id')
    problem_tags = pd.merge(problem_tags, tags_df, left_on='problemtag_id', right_on='id')

    # 为每个问题创建一个包含所有知识点标签的特征向量
    max_tags_per_problem = 5
    tag_features = np.zeros((len(problem_df), max_tags_per_problem), dtype=np.longlong)

    for i, problem_id in enumerate(problem_df['id']):
        tags = problem_tags[problem_tags['problem_id'] == problem_id]['name'].values[:max_tags_per_problem]
        tag_features[i, :len(tags)] = tags

    # 将知识点标签特征与问题特征合并
    node_features = np.hstack([problem_features, tag_features])

    # 加载提交数据
    submission_df = pd.read_csv('submission.csv')
    # 提取边的特征，例如结果、时间戳等
    submission_features = submission_df[['result', 'create_time']].values

    # 获取所有唯一节点的集合
    node_set = set(submission_df['user_id']) | set(submission_df['problem_id'])
    num_total_unique_node_ids = len(node_set)

    # 获取验证集和测试集的时间戳分位数
    node_interact_times = pd.to_datetime(submission_df['create_time']).astype(np.float64) // 10**9
    val_time, test_time = list(np.quantile(node_interact_times, [(1 - val_ratio - test_ratio), (1 - test_ratio)]))

    # 设置随机种子以确保结果可复现
    random.seed(2020)

    # 训练数据掩码：交互时间在验证时间之前
    train_mask = node_interact_times <= val_time
    val_mask = np.logical_and(node_interact_times <= test_time, node_interact_times > val_time)
    test_mask = node_interact_times > test_time

    # 创建数据对象
    full_data = Data(
        src_node_ids=submission_df['user_id'].values.astype(np.longlong),
        dst_node_ids=submission_df['problem_id'].values.astype(np.longlong),
        node_interact_times=node_interact_times,
        edge_ids=submission_df['edge_id'].values.astype(np.longlong),
        labels=submission_df['result'].values
    )

    train_data = Data(
        src_node_ids=submission_df['user_id'].values.astype(np.longlong)[train_mask],
        dst_node_ids=submission_df['problem_id'].values.astype(np.longlong)[train_mask],
        node_interact_times=node_interact_times[train_mask],
        edge_ids=submission_df['edge_id'].values.astype(np.longlong)[train_mask],
        labels=submission_df['result'].values[train_mask]
    )

    val_data = Data(
        src_node_ids=submission_df['user_id'].values.astype(np.longlong)[val_mask],
        dst_node_ids=submission_df['problem_id'].values.astype(np.longlong)[val_mask],
        node_interact_times=node_interact_times[val_mask],
        edge_ids=submission_df['edge_id'].values.astype(np.longlong)[val_mask],
        labels=submission_df['result'].values[val_mask]
    )

    test_data = Data(
        src_node_ids=submission_df['user_id'].values.astype(np.longlong)[test_mask],
        dst_node_ids=submission_df['problem_id'].values.astype(np.longlong)[test_mask],
        node_interact_times=node_interact_times[test_mask],
        edge_ids=submission_df['edge_id'].values.astype(np.longlong)[test_mask],
        labels=submission_df['result'].values[test_mask]
    )

    return node_features, submission_features, full_data, train_data, val_data, test_data


# 数据集划分（连接分类任务）
def get_link_classification_data(dataset_name: str, val_ratio: float, test_ratio: float):
    """
    生成链接分类任务的数据。
    :param dataset_name: str, 数据集名称
    :param val_ratio: float, 验证数据比例
    :param test_ratio: float, 测试数据比例
    :return: 节点特征、边特征、完整数据、训练数据、验证数据、测试数据、新节点验证数据、新节点测试数据
    """
    # 加载问题数据
    problem_df = pd.read_csv('problem.csv')
    # 加载问题标签数据
    problem_tag_df = pd.read_csv('problem_tag.csv')
    # 加载标签数据
    tags_df = pd.read_csv('problem_tags.csv')

    # 提取问题的特征，包括难度级别、描述、时间限制、内存限制和知识点标签
    problem_features = problem_df[['difficulty', 'time_limit', 'memory_limit']].values

    # 将问题与标签关联，提取每个问题的知识点标签
    problem_tags = pd.merge(problem_df, problem_tag_df, left_on='id', right_on='problem_id')
    problem_tags = pd.merge(problem_tags, tags_df, left_on='problemtag_id', right_on='id')

    # 为每个问题创建一个包含所有知识点标签的特征向量
    max_tags_per_problem = 5
    tag_features = np.zeros((len(problem_df), max_tags_per_problem), dtype=np.longlong)

    for i, problem_id in enumerate(problem_df['id']):
        tags = problem_tags[problem_tags['problem_id'] == problem_id]['name'].values[:max_tags_per_problem]
        tag_features[i, :len(tags)] = tags

    # 将知识点标签特征与问题特征合并
    node_features = np.hstack([problem_features, tag_features])

    # 加载提交数据
    submission_df = pd.read_csv('submission.csv')
    # 提取边的特征，例如结果、时间戳等
    submission_features = submission_df[['result', 'create_time']].values

    # 获取所有唯一节点的集合
    node_set = set(submission_df['user_id']) | set(submission_df['problem_id'])
    num_total_unique_node_ids = len(node_set)

    # 获取验证集和测试集的时间戳分位数
    node_interact_times = pd.to_datetime(submission_df['create_time']).astype(np.float64) // 10**9
    val_time, test_time = list(np.quantile(node_interact_times, [(1 - val_ratio - test_ratio), (1 - test_ratio)]))

    # 设置随机种子以确保结果可复现
    random.seed(2020)

    # 计算测试时间之后出现的节点
    test_node_set = set(submission_df['user_id'][node_interact_times > val_time]).union(set(submission_df['problem_id'][node_interact_times > val_time]))
    # 从测试节点中随机选择10%的节点作为新测试节点
    new_test_node_set = set(random.sample(sorted(test_node_set), int(0.1 * num_total_unique_node_ids)))

    # 生成源节点和目标节点是否为新测试节点的掩码
    new_test_source_mask = np.array([x in new_test_node_set for x in submission_df['user_id']])
    new_test_destination_mask = np.array([x in new_test_node_set for x in submission_df['problem_id']])

    # 生成掩码，表示源节点和目标节点都不是新测试节点的边
    observed_edges_mask = np.logical_and(~new_test_source_mask, ~new_test_destination_mask)

    # 训练数据掩码：交互时间在验证时间之前且不涉及新测试节点的边
    train_mask = np.logical_and(node_interact_times <= val_time, observed_edges_mask)

    # 创建数据对象
    full_data = Data(
        src_node_ids=submission_df['user_id'].values.astype(np.longlong),
        dst_node_ids=submission_df['problem_id'].values.astype(np.longlong),
        node_interact_times=node_interact_times,
        edge_ids=submission_df['edge_id'].values.astype(np.longlong),
        labels=submission_df['result'].values
    )

    train_data = Data(
        src_node_ids=submission_df['user_id'].values.astype(np.longlong)[train_mask],
        dst_node_ids=submission_df['problem_id'].values.astype(np.longlong)[train_mask],
        node_interact_times=node_interact_times[train_mask],
        edge_ids=submission_df['edge_id'].values.astype(np.longlong)[train_mask],
        labels=submission_df['result'].values[train_mask]
    )

    # 训练节点集合
    train_node_set = set(train_data.src_node_ids).union(train_data.dst_node_ids)
    # 确保新测试节点不在训练集中
    assert len(train_node_set & new_test_node_set) == 0
    # 新节点集合（不在训练集中）
    new_node_set = node_set - train_node_set

    # 验证集和测试集掩码
    val_mask = np.logical_and(node_interact_times <= test_time, node_interact_times > val_time)
    test_mask = node_interact_times > test_time

    # 新节点验证集和测试集掩码
    edge_contains_new_node_mask = np.array([(src_node_id in new_node_set or dst_node_id in new_node_set)
                                           for src_node_id, dst_node_id in zip(submission_df['user_id'], submission_df['problem_id'])])
    new_node_val_mask = np.logical_and(val_mask, edge_contains_new_node_mask)
    new_node_test_mask = np.logical_and(test_mask, edge_contains_new_node_mask)

    # 创建验证集和测试集数据对象
    val_data = Data(
        src_node_ids=submission_df['user_id'].values.astype(np.longlong)[val_mask],
        dst_node_ids=submission_df['problem_id'].values.astype(np.longlong)[val_mask],
        node_interact_times=node_interact_times[val_mask],
        edge_ids=submission_df['edge_id'].values.astype(np.longlong)[val_mask],
        labels=submission_df['result'].values[val_mask]
    )

    test_data = Data(
        src_node_ids=submission_df['user_id'].values.astype(np.longlong)[test_mask],
        dst_node_ids=submission_df['problem_id'].values.astype(np.longlong)[test_mask],
        node_interact_times=node_interact_times[test_mask],
        edge_ids=submission_df['edge_id'].values.astype(np.longlong)[test_mask],
        labels=submission_df['result'].values[test_mask]
    )

    # 创建包含新节点的验证集和测试集数据对象
    new_node_val_data = Data(
        src_node_ids=submission_df['user_id'].values.astype(np.longlong)[new_node_val_mask],
        dst_node_ids=submission_df['problem_id'].values.astype(np.longlong)[new_node_val_mask],
        node_interact_times=node_interact_times[new_node_val_mask],
        edge_ids=submission_df['edge_id'].values.astype(np.longlong)[new_node_val_mask],
        labels=submission_df['result'].values[new_node_val_mask]
    )

    new_node_test_data = Data(
        src_node_ids=submission_df['user_id'].values.astype(np.longlong)[new_node_test_mask],
        dst_node_ids=submission_df['problem_id'].values.astype(np.longlong)[new_node_test_mask],
        node_interact_times=node_interact_times[new_node_test_mask],
        edge_ids=submission_df['edge_id'].values.astype(np.longlong)[new_node_test_mask],
        labels=submission_df['result'].values[new_node_test_mask]
    )

    return node_features, submission_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data
