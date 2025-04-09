import argparse
import sys
import torch

def get_link_classification_args(is_evaluation: bool = False):
    """
    获取链接分类任务的参数
    :param is_evaluation: 是否处于评估过程
    :return: 参数对象
    """
    parser = argparse.ArgumentParser('链接预测任务的接口')
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--dataset_name', type=str, help='使用的数据集', default='BePKT')
    parser.add_argument('--batch_size', type=int, default=2000, help='批次大小')
    parser.add_argument('--model_name', type=str, default='DyGKT', help='模型名称',
                        choices=['DyGKT', 'QIKT', 'IEKT', 'IPKT', 'DIMKT', 'simpleKT', 'CTNCM', 'AKT', 'DKT', 'TGAT', 'TGN', 'DyGFormer'])
    parser.add_argument('--gpu', type=int, default=0, help='使用的GPU编号')

    parser.add_argument('--ablation', type=str, default='-1', choices=['-1', 'q_qid', 'q_kid', 'counter', 'dual', 'embed', 'skill', 'time'], help='如何采样历史邻居')

    parser.add_argument('--predict_loss', type=float, default=0.0, help='使用的损失函数')

    parser.add_argument('--state_dim', type=int, default=256, help='学生状态嵌入的维度')
    parser.add_argument('--num_neighbors', type=int, default=50, help='每个节点采样的邻居数量')
    parser.add_argument('--dropout', type=float, default=0.1, help='丢弃率')
    parser.add_argument('--sample_neighbor_strategy', type=str, default='recent', choices=['uniform', 'recent', 'time_interval_aware'], help='如何采样历史邻居')
    parser.add_argument('--time_scaling_factor', default=1e-6, type=float, help='控制时间间隔采样偏好的超参数')
    parser.add_argument('--num_walk_heads', type=int, default=8, help='行走编码器中使用的注意力头数')
    parser.add_argument('--num_heads', type=int, default=2, help='注意力层中使用的头数')
    parser.add_argument('--num_layers', type=int, default=2, help='模型层数')
    parser.add_argument('--walk_length', type=int, default=1, help='每次随机行走的长度')
    parser.add_argument('--time_gap', type=int, default=2000, help='计算节点特征的时间间隔')
    parser.add_argument('--time_feat_dim', type=int, default=16, help='时间嵌入的维度')
    parser.add_argument('--position_feat_dim', type=int, default=172, help='位置嵌入的维度')
    parser.add_argument('--time_window_mode', type=str, default='fixed_proportion', help='如何选择时间窗口大小')
    parser.add_argument('--patch_size', type=int, default=1, help='补丁大小')
    parser.add_argument('--channel_embedding_dim', type=int, default=50, help='每个通道嵌入的维度')
    parser.add_argument('--max_input_sequence_length', type=int, default=20, help='每个节点输入序列的最大长度')
    parser.add_argument('--learning_rate', type=float, default=0.0005, help='学习率')
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['SGD', 'Adam', 'RMSprop'], help='优化器名称')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='权重衰减')
    parser.add_argument('--patience', type=int, default=10, help='早停的耐心值')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='验证集比例')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='测试集比例')
    parser.add_argument('--num_runs', type=int, default=3, help='运行次数')
    parser.add_argument('--test_interval_epochs', type=int, default=10, help='每多少个周期进行一次测试')
    parser.add_argument('--negative_sample_strategy', type=str, default='random', choices=['random', 'historical', 'inductive'], help='负样本采样策略')

    try:
        args = parser.parse_args()
        args.device = f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu'
    except:
        parser.print_help()
        sys.exit()

    return args