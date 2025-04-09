import logging
import time
import sys
import os
from tqdm import tqdm
import numpy as np
import warnings
import shutil
import json
import torch
import torch.nn as nn
from models.TGAT import TGAT
from models.DyGKT import DyGKT
from models.MemoryModel import MemoryModel, compute_src_dst_node_time_shifts
from models.simpleKT import SimpleKT
from models.DyGFormer import DyGFormer
from models.CTNCM import CTNCM
from models.DKT import DKT
from models.DIMKT import DIMKT
from models.QIKT import QIKT
from models.IPKT import IPKT
from models.IEKT import IEKT
from models.AKT import AKT
from models.modules import MergeLayer, MLPClassifier
from utils.utils import set_random_seed, convert_to_gpu, get_parameter_sizes, create_optimizer
from utils.utils import get_neighbor_sampler, NegativeEdgeSampler
from evaluate_models_utils import evaluate_model_link_classification
from utils.metrics import get_link_classification_metrics
from utils.DataLoader import get_idx_data_loader, get_link_classification_data
from utils.EarlyStopping import EarlyStopping
from utils.load_configs import get_link_classification_args

if __name__ == "__main__":
    warnings.filterwarnings('ignore')

    # 获取参数
    args = get_link_classification_args(is_evaluation=False)

    # 加载数据
    node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data = \
        get_link_classification_data(dataset_name=args.dataset_name, val_ratio=args.val_ratio,
                                     test_ratio=args.test_ratio)

    # 初始化训练邻居采样器
    train_neighbor_sampler = get_neighbor_sampler(data=train_data,
                                                  sample_neighbor_strategy=args.sample_neighbor_strategy,
                                                  time_scaling_factor=args.time_scaling_factor, seed=0)

    # 初始化验证和测试邻居采样器
    full_neighbor_sampler = get_neighbor_sampler(data=full_data, sample_neighbor_strategy=args.sample_neighbor_strategy,
                                                 time_scaling_factor=args.time_scaling_factor, seed=1)

    # 初始化负样本采样器
    train_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=train_data.src_node_ids,
                                                 dst_node_ids=train_data.dst_node_ids)
    val_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=full_data.src_node_ids, dst_node_ids=full_data.dst_node_ids,
                                               seed=0)
    new_node_val_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=new_node_val_data.src_node_ids,
                                                        dst_node_ids=new_node_val_data.dst_node_ids, seed=1)
    test_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=full_data.src_node_ids,
                                                dst_node_ids=full_data.dst_node_ids, seed=2)
    new_node_test_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=new_node_test_data.src_node_ids,
                                                         dst_node_ids=new_node_test_data.dst_node_ids, seed=3)

    # 获取数据加载器
    train_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(train_data.src_node_ids))),
                                                batch_size=args.batch_size, shuffle=False)
    val_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(val_data.src_node_ids))),
                                              batch_size=args.batch_size, shuffle=False)
    new_node_val_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(new_node_val_data.src_node_ids))),
                                                       batch_size=args.batch_size, shuffle=False)
    test_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(test_data.src_node_ids))),
                                               batch_size=args.batch_size, shuffle=False)
    new_node_test_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(new_node_test_data.src_node_ids))),
                                                        batch_size=args.batch_size, shuffle=False)

    val_metric_all_runs, new_node_val_metric_all_runs, test_metric_all_runs, new_node_test_metric_all_runs = [], [], [], []

    for run in range(args.num_runs):
        set_random_seed(seed=run)

        args.seed = run
        args.save_model_name = f'{args.model_name}_seed{args.seed}'

        # 设置日志
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        os.makedirs(f"./logs/{args.model_name}/{args.dataset_name}/{args.save_model_name}/", exist_ok=True)
        inf = 'paper'
        fh = logging.FileHandler(
            f"./logs/{args.model_name}/{args.dataset_name}/{args.save_model_name}/{inf + str(time.time())}.log")
        fh.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(ch)

        run_start_time = time.time()
        start_time = time.time()
        initial_memory = torch.cuda.memory_allocated()
        logger.info(f"********** Run {run + 1} starts. **********")

        logger.info(f'configuration is {args}')

        # 创建模型
        if args.model_name == 'DKT':
            dynamic_backbone = DKT(node_raw_features=node_raw_features,
                                   edge_raw_features=edge_raw_features,
                                   dropout=args.dropout,
                                   num_neighbors=args.num_neighbors,
                                   device=args.device)
        elif args.model_name == 'DyGKT':
            dynamic_backbone = DyGKT(node_raw_features=node_raw_features,
                                     edge_raw_features=edge_raw_features,
                                     dropout=args.dropout,
                                     num_neighbors=args.num_neighbors,
                                     device=args.device,
                                     ablation=args.ablation)

        elif args.model_name == 'CTNCM':
            dynamic_backbone = CTNCM(node_raw_features=node_raw_features,
                                     edge_raw_features=edge_raw_features,
                                     dropout=args.dropout,
                                     num_neighbors=args.num_neighbors,
                                     device=args.device)

        elif args.model_name == 'AKT':
            dynamic_backbone = AKT(node_raw_features=node_raw_features,
                                   edge_raw_features=edge_raw_features,
                                   dropout=args.dropout,
                                   num_neighbors=args.num_neighbors,
                                   device=args.device)
        elif args.model_name == 'DIMKT':
            dynamic_backbone = DIMKT(node_raw_features=node_raw_features,
                                     edge_raw_features=edge_raw_features,
                                     dropout=args.dropout,
                                     dataset_name=args.dataset_name,
                                     device=args.device)
        elif args.model_name == 'IPKT':
            dynamic_backbone = IPKT(node_raw_features=node_raw_features,
                                    edge_raw_features=edge_raw_features,
                                    dropout=args.dropout,
                                    device=args.device)

        elif args.model_name == 'IEKT':
            dynamic_backbone = IEKT(node_raw_features=node_raw_features,
                                    edge_raw_features=edge_raw_features,
                                    dropout=args.dropout,
                                    device=args.device)
        elif args.model_name == 'QIKT':
            dynamic_backbone = QIKT(node_raw_features=node_raw_features,
                                    edge_raw_features=edge_raw_features,
                                    dropout=args.dropout,
                                    device=args.device)
        elif args.model_name == 'simpleKT':
            dynamic_backbone = SimpleKT(node_raw_features=node_raw_features,
                                        edge_raw_features=edge_raw_features,
                                        dropout=args.dropout,
                                        num_neighbors=args.num_neighbors,
                                        device=args.device)

        elif args.model_name == 'TGAT':
            dynamic_backbone = TGAT(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features,
                                    neighbor_sampler=train_neighbor_sampler,
                                    time_feat_dim=args.time_feat_dim, num_layers=args.num_layers,
                                    num_heads=args.num_heads, dropout=args.dropout, device=args.device)

        elif args.model_name in ['TGN']:
            src_node_mean_time_shift, src_node_std_time_shift, dst_node_mean_time_shift_dst, dst_node_std_time_shift = \
                compute_src_dst_node_time_shifts(train_data.src_node_ids, train_data.dst_node_ids,
                                                 train_data.node_interact_times)
            dynamic_backbone = MemoryModel(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features,
                                           neighbor_sampler=train_neighbor_sampler,
                                           time_feat_dim=args.time_feat_dim, model_name=args.model_name,
                                           num_layers=args.num_layers, num_heads=args.num_heads,
                                           dropout=args.dropout, src_node_mean_time_shift=src_node_mean_time_shift,
                                           src_node_std_time_shift=src_node_std_time_shift,
                                           dst_node_mean_time_shift_dst=dst_node_mean_time_shift_dst,
                                           dst_node_std_time_shift=dst_node_std_time_shift, device=args.device)

        elif args.model_name == 'DyGFormer':
            dynamic_backbone = DyGFormer(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features,
                                         neighbor_sampler=train_neighbor_sampler,
                                         time_feat_dim=args.time_feat_dim,
                                         channel_embedding_dim=args.channel_embedding_dim, patch_size=args.patch_size,
                                         num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout,
                                         max_input_sequence_length=args.max_input_sequence_length, device=args.device)
        else:
            raise ValueError(f"Wrong value for model_name {args.model_name}!")

        if args.model_name == 'DyGKT':
            link_predictor = MergeLayer(input_dim1=64, input_dim2=64, hidden_dim=64, output_dim=1)
        else:
            link_predictor = MergeLayer(node_raw_features.shape[1], input_dim2=node_raw_features.shape[1],
                                        hidden_dim=node_raw_features.shape[1], output_dim=1)

        model = nn.Sequential(dynamic_backbone, link_predictor)
        logger.info(f'model -> {model}')
        logger.info(f'model name: {args.model_name}, #parameters: {get_parameter_sizes(model) * 4} B, '
                    f'{get_parameter_sizes(model) * 4 / 1024} KB, {get_parameter_sizes(model) * 4 / 1024 / 1024} MB.')

        optimizer = create_optimizer(model=model, optimizer_name=args.optimizer, learning_rate=args.learning_rate,
                                     weight_decay=args.weight_decay)

        model = convert_to_gpu(model, device=args.device)

        save_model_folder = f"./saved_models/{args.model_name}/{args.dataset_name}/{args.save_model_name}/"
        shutil.rmtree(save_model_folder, ignore_errors=True)
        os.makedirs(save_model_folder, exist_ok=True)

        early_stopping = EarlyStopping(patience=args.patience, save_model_folder=save_model_folder,
                                       save_model_name=args.save_model_name, logger=logger, model_name=args.model_name)

        loss_func = nn.BCELoss()
        torch.autograd.set_detect_anomaly(True)
        final_memory = None
        for epoch in range(args.num_epochs):
            if args.test and epoch > 0:
                end_time = time.time()
                memory_used = final_memory - initial_memory
                logger.info(f'USE_TIME: {end_time - start_time}')
                logger.info(f"USE_GPU: {memory_used / (1024 ** 2):.3f} MB")
                logger.info(f'MODEL_PARA: {get_parameter_sizes(model) * 4 / 1024 / 1024:.3f} MB')
                print(args.model_name)
                sys.exit()
            model.train()
            if args.model_name in ['DyGKT', 'QIKT', 'IEKT', 'IPKT', 'DIMKT', 'TGAT', 'TGN', 'DyGFormer', 'DKT', 'AKT',
                                   'CTNCM', 'simpleKT']:
                model[0].set_neighbor_sampler(train_neighbor_sampler)
            if args.model_name in ['TGN']:
                model[0].memory_bank.__init_memory_bank__()
                model[0].last_node_id = None

            train_losses, train_metrics = [], []
            train_predicts, train_labels = [], []
            train_idx_data_loader_tqdm = tqdm(train_idx_data_loader, ncols=120)
            for batch_idx, train_data_indices in enumerate(train_idx_data_loader_tqdm):
                train_data_indices = train_data_indices.numpy()
                batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids, batch_edge_labels = \
                    train_data.src_node_ids[train_data_indices], train_data.dst_node_ids[train_data_indices], \
                    train_data.node_interact_times[train_data_indices], train_data.edge_ids[train_data_indices], \
                    train_data.labels[train_data_indices]

                if args.model_name in ['DyGKT', 'DKT', 'AKT', 'CTNCM', 'simpleKT']:
                    batch_src_node_embeddings, batch_dst_node_embeddings = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                          edge_ids=batch_edge_ids,
                                                                          node_interact_times=batch_node_interact_times,
                                                                          dst_node_ids=batch_dst_node_ids)

                elif args.model_name in ['QIKT', 'IEKT', 'IPKT', 'DIMKT']:
                    batch_src_node_embeddings, batch_dst_node_embeddings = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                          dst_node_ids=batch_dst_node_ids,
                                                                          node_interact_times=batch_node_interact_times,
                                                                          edge_ids=batch_edge_ids)

                elif args.model_name in ['TGAT']:
                    batch_src_node_embeddings, batch_dst_node_embeddings = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                          dst_node_ids=batch_dst_node_ids,
                                                                          node_interact_times=batch_node_interact_times,
                                                                          num_neighbors=args.num_neighbors)

                elif args.model_name in ['TGN']:
                    batch_src_node_embeddings, batch_dst_node_embeddings = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                          dst_node_ids=batch_dst_node_ids,
                                                                          node_interact_times=batch_node_interact_times,
                                                                          edge_ids=batch_edge_ids,
                                                                          edges_are_positive=True,
                                                                          num_neighbors=args.num_neighbors)

                elif args.model_name in ['DyGFormer']:
                    batch_src_node_embeddings, batch_dst_node_embeddings = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                          dst_node_ids=batch_dst_node_ids,
                                                                          node_interact_times=batch_node_interact_times)
                else:
                    raise ValueError(f"Wrong value for model_name {args.model_name}!")

                predicts = model[1](batch_src_node_embeddings, batch_dst_node_embeddings).squeeze(dim=-1).sigmoid()
                labels = torch.tensor(batch_edge_labels, dtype=torch.float32, device=args.device)
                loss = loss_func(input=predicts, target=labels)
                final_memory = torch.cuda.memory_allocated()
                train_losses.append(loss.item())

                train_predicts.append(predicts)
                train_labels.append(labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_idx_data_loader_tqdm.set_description(
                    f'Epoch: {epoch + 1}, train for the {batch_idx + 1}-th batch, train loss: {loss.item()}')

                if args.model_name in ['TGN']:
                    model[0].memory_bank.detach_memory_bank()

            if args.model_name in ['TGN']:
                train_backup_memory_bank = model[0].memory_bank.backup_memory_bank()

            val_losses, val_metrics = evaluate_model_link_classification(model_name=args.model_name,
                                                                         model=model,
                                                                         neighbor_sampler=full_neighbor_sampler,
                                                                         evaluate_idx_data_loader=val_idx_data_loader,
                                                                         evaluate_neg_edge_sampler=val_neg_edge_sampler,
                                                                         evaluate_data=val_data,
                                                                         loss_func=loss_func,
                                                                         num_neighbors=args.num_neighbors,
                                                                         time_gap=args.time_gap)

            if args.model_name in ['TGN']:
                val_backup_memory_bank = model[0].memory_bank.backup_memory_bank()
                model[0].memory_bank.reload_memory_bank(train_backup_memory_bank)

            new_node_val_losses, new_node_val_metrics = evaluate_model_link_classification(model_name=args.model_name,
                                                                                           model=model,
                                                                                           neighbor_sampler=full_neighbor_sampler,
                                                                                           evaluate_idx_data_loader=new_node_val_idx_data_loader,
                                                                                           evaluate_neg_edge_sampler=new_node_val_neg_edge_sampler,
                                                                                           evaluate_data=new_node_val_data,
                                                                                           loss_func=loss_func,
                                                                                           num_neighbors=args.num_neighbors,
                                                                                           time_gap=args.time_gap)

            if args.model_name in ['TGN']:
                model[0].memory_bank.reload_memory_bank(val_backup_memory_bank)

            train_predict = torch.cat(train_predicts, dim=0)
            train_label = torch.cat(train_labels, dim=0)

            train_metrics.append(get_link_classification_metrics(predicts=train_predict, labels=train_label))

            logger.info(
                f'Epoch: {epoch + 1}, learning rate: {optimizer.param_groups[0]["lr"]}, train loss: {np.mean(train_losses):.4f}')
            for metric_name in train_metrics[0].keys():
                logger.info(
                    f'train {metric_name}, {np.mean([train_metric[metric_name] for train_metric in train_metrics]):.4f}')
            logger.info(f'validate loss: {np.mean(val_losses):.4f}')
            for metric_name in val_metrics[0].keys():
                logger.info(
                    f'validate {metric_name}, {np.mean([val_metric[metric_name] for val_metric in val_metrics]):.4f}')
            logger.info(f'new node validate loss: {np.mean(new_node_val_losses):.4f}')
            for metric_name in new_node_val_metrics[0].keys():
                logger.info(
                    f'new node validate {metric_name}, {np.mean([new_node_val_metric[metric_name] for new_node_val_metric in new_node_val_metrics]):.4f}')

            if (epoch + 1) % args.test_interval_epochs == 0:
                test_losses, test_metrics = evaluate_model_link_classification(model_name=args.model_name,
                                                                               model=model,
                                                                               neighbor_sampler=full_neighbor_sampler,
                                                                               evaluate_idx_data_loader=test_idx_data_loader,
                                                                               evaluate_neg_edge_sampler=test_neg_edge_sampler,
                                                                               evaluate_data=test_data,
                                                                               loss_func=loss_func,
                                                                               num_neighbors=args.num_neighbors,
                                                                               time_gap=args.time_gap)

                if args.model_name in ['TGN']:
                    model[0].memory_bank.reload_memory_bank(val_backup_memory_bank)

                new_node_test_losses, new_node_test_metrics = evaluate_model_link_classification(
                    model_name=args.model_name,
                    model=model,
                    neighbor_sampler=full_neighbor_sampler,
                    evaluate_idx_data_loader=new_node_test_idx_data_loader,
                    evaluate_neg_edge_sampler=new_node_test_neg_edge_sampler,
                    evaluate_data=new_node_test_data,
                    loss_func=loss_func,
                    num_neighbors=args.num_neighbors,
                    time_gap=args.time_gap)

                if args.model_name in ['TGN']:
                    model[0].memory_bank.reload_memory_bank(val_backup_memory_bank)

                logger.info(f'test loss: {np.mean(test_losses):.4f}')
                for metric_name in test_metrics[0].keys():
                    logger.info(
                        f'test {metric_name}, {np.mean([test_metric[metric_name] for test_metric in test_metrics]):.4f}')
                logger.info(f'new node test loss: {np.mean(new_node_test_losses):.4f}')
                for metric_name in new_node_test_metrics[0].keys():
                    logger.info(
                        f'new node test {metric_name}, {np.mean([new_node_test_metric[metric_name] for new_node_test_metric in new_node_test_metrics]):.4f}')

            val_metric_indicator = []
            for metric_name in val_metrics[0].keys():
                val_metric_indicator.append(
                    (metric_name, np.mean([val_metric[metric_name] for val_metric in val_metrics]), True))
            early_stop = early_stopping.step(val_metric_indicator, model)

            if early_stop:
                break

        early_stopping.load_checkpoint(model)

        logger.info(f'get final performance on dataset {args.dataset_name}...')

        if args.model_name not in ['TGN']:
            val_losses, val_metrics = evaluate_model_link_classification(model_name=args.model_name,
                                                                         model=model,
                                                                         neighbor_sampler=full_neighbor_sampler,
                                                                         evaluate_idx_data_loader=val_idx_data_loader,
                                                                         evaluate_neg_edge_sampler=val_neg_edge_sampler,
                                                                         evaluate_data=val_data,
                                                                         loss_func=loss_func,
                                                                         num_neighbors=args.num_neighbors,
                                                                         time_gap=args.time_gap)

            new_node_val_losses, new_node_val_metrics = evaluate_model_link_classification(model_name=args.model_name,
                                                                                           model=model,
                                                                                           neighbor_sampler=full_neighbor_sampler,
                                                                                           evaluate_idx_data_loader=new_node_val_idx_data_loader,
                                                                                           evaluate_neg_edge_sampler=new_node_val_neg_edge_sampler,
                                                                                           evaluate_data=new_node_val_data,
                                                                                           loss_func=loss_func,
                                                                                           num_neighbors=args.num_neighbors,
                                                                                           time_gap=args.time_gap)

        if args.model_name in ['TGN']:
            val_backup_memory_bank = model[0].memory_bank.backup_memory_bank()

        test_losses, test_metrics = evaluate_model_link_classification(model_name=args.model_name,
                                                                       model=model,
                                                                       neighbor_sampler=full_neighbor_sampler,
                                                                       evaluate_idx_data_loader=test_idx_data_loader,
                                                                       evaluate_neg_edge_sampler=test_neg_edge_sampler,
                                                                       evaluate_data=test_data,
                                                                       loss_func=loss_func,
                                                                       num_neighbors=args.num_neighbors,
                                                                       time_gap=args.time_gap)

        if args.model_name in ['TGN']:
            model[0].memory_bank.reload_memory_bank(val_backup_memory_bank)

        new_node_test_losses, new_node_test_metrics = evaluate_model_link_classification(model_name=args.model_name,
                                                                                         model=model,
                                                                                         neighbor_sampler=full_neighbor_sampler,
                                                                                         evaluate_idx_data_loader=new_node_test_idx_data_loader,
                                                                                         evaluate_neg_edge_sampler=new_node_test_neg_edge_sampler,
                                                                                         evaluate_data=new_node_test_data,
                                                                                         loss_func=loss_func,
                                                                                         num_neighbors=args.num_neighbors,
                                                                                         time_gap=args.time_gap)

        if args.model_name not in ['TGN']:
            val_metric_dict, new_node_val_metric_dict = {}, {}
            for metric_name in val_metrics[0].keys():
                average_val_metric = np.mean([val_metric[metric_name] for val_metric in val_metrics])
                val_metric_dict[metric_name] = average_val_metric
            for metric_name in new_node_val_metrics[0].keys():
                average_new_node_val_metric = np.mean(
                    [new_node_val_metric[metric_name] for new_node_val_metric in new_node_val_metrics])
                new_node_val_metric_dict[metric_name] = average_new_node_val_metric

        test_metric_dict, new_node_test_metric_dict = {}, {}
        for metric_name in test_metrics[0].keys():
            average_test_metric = np.mean([test_metric[metric_name] for test_metric in test_metrics])
            test_metric_dict[metric_name] = average_test_metric
        for metric_name in new_node_test_metrics[0].keys():
            average_new_node_test_metric = np.mean(
                [new_node_test_metric[metric_name] for new_node_test_metric in new_node_test_metrics])
            new_node_test_metric_dict[metric_name] = average_new_node_test_metric

        logger.info(f'test loss: {np.mean(test_losses):.4f}')
        for metric_name in test_metrics[0].keys():
            logger.info(
                f'test {metric_name}, {np.mean([test_metric[metric_name] for test_metric in test_metrics]):.4f}')
        logger.info(f'new node test loss: {np.mean(new_node_test_losses):.4f}')
        for metric_name in new_node_test_metrics[0].keys():
            logger.info(
                f'new node test {metric_name}, {np.mean([new_node_test_metric[metric_name] for new_node_test_metric in new_node_test_metrics]):.4f}')

        single_run_time = time.time() - run_start_time
        logger.info(f'Run {run + 1} cost {single_run_time:.2f} seconds.')

        if args.model_name not in ['TGN']:
            val_metric_all_runs.append(val_metric_dict)
            new_node_val_metric_all_runs.append(new_node_val_metric_dict)
        test_metric_all_runs.append(test_metric_dict)
        new_node_test_metric_all_runs.append(new_node_test_metric_dict)

        if run < args.num_runs - 1:
            logger.removeHandler(fh)
            logger.removeHandler(ch)

        save_result_folder = f"./saved_results/{args.model_name}/{args.dataset_name}"
        os.makedirs(save_result_folder, exist_ok=True)
        save_result_path = os.path.join(save_result_folder, f"{args.save_model_name}.json")

        with open(save_result_path, 'w') as file:
            file.write(json.dumps({
                "validate metrics": val_metric_dict if args.model_name not in ['TGN'] else {},
                "new node validate metrics": new_node_val_metric_dict if args.model_name not in ['TGN'] else {},
                "test metrics": test_metric_dict,
                "new node test metrics": new_node_test_metric_dict
            }, indent=4))

    logger.info(f'metrics over {args.num_runs} runs:')
    if args.model_name not in ['TGN']:
        for metric_name in val_metric_all_runs[0].keys():
            logger.info(
                f'validate {metric_name}, {[val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs]}')
            logger.info(
                f'average validate {metric_name}, {np.mean([val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs]):.4f} ± {np.std([val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs], ddof=1):.4f}')
        for metric_name in new_node_val_metric_all_runs[0].keys():
            logger.info(
                f'new node validate {metric_name}, {[new_node_val_metric_single_run[metric_name] for new_node_val_metric_single_run in new_node_val_metric_all_runs]}')
            logger.info(
                f'average new node validate {metric_name}, {np.mean([new_node_val_metric_single_run[metric_name] for new_node_val_metric_single_run in new_node_val_metric_all_runs]):.4f} ± {np.std([new_node_val_metric_single_run[metric_name] for new_node_val_metric_single_run in new_node_val_metric_all_runs], ddof=1):.4f}')

    for metric_name in test_metric_all_runs[0].keys():
        logger.info(
            f'test {metric_name}, {[test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]}')
        logger.info(
            f'average test {metric_name}, {np.mean([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]):.4f} ± {np.std([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs], ddof=1):.4f}')

    for metric_name in new_node_test_metric_all_runs[0].keys():
        logger.info(
            f'new node test {metric_name}, {[new_node_test_metric_single_run[metric_name] for new_node_test_metric_single_run in new_node_test_metric_all_runs]}')
        logger.info(
            f'average new node test {metric_name}, {np.mean([new_node_test_metric_single_run[metric_name] for new_node_test_metric_single_run in new_node_test_metric_all_runs]):.4f} ± {np.std([new_node_test_metric_single_run[metric_name] for new_node_test_metric_single_run in new_node_test_metric_all_runs], ddof=1):.4f}')

    sys.exit()