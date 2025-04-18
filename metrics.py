import torch
from sklearn.metrics import average_precision_score, roc_auc_score
# 这个函数使用来计算连接分类任务的评估指标，通过计算AP和AUC-ROC
def get_link_classification_metrics(predicts: torch.Tensor, labels: torch.Tensor):
    """
    get metrics for the link classification task
    :param predicts: Tensor, shape (num_samples, )
    :param labels: Tensor, shape (num_samples, )
    :return:
        dictionary of metrics {'metric_name_1': metric_1, ...}
    """
    predicts = predicts.cpu().detach().numpy()
    labels = labels.cpu().numpy()
    average_precision = average_precision_score(y_true=labels, y_score=predicts)
    roc_auc = roc_auc_score(y_true=labels, y_score=predicts)

    return {'average_precision': average_precision, 'roc_auc': roc_auc}

