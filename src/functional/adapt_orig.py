import copy
from fastargs.decorators import param
from fastargs import Param
import torch

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn.inits import glorot

import ast
import re
from .graph_prompt import GPF

import networkx as nx
import torch


def find_subgraph_class_distrib(data, mask):
    """
    input : data - graph dataset
    output : case_type을 key값, case_type에 해당하는 sample로 구성된 list를 value로 삼는 dictionary.
    하는 일 : data sample 중 mask로 지정된 일부 set(e.g., training set)에 포함된 sample 각각에 대하여 sample의 case_type을 판단함.
        case1 : 자기 자신의 label과 자신과 1-hop 연결된 모든 이웃의 label이 같은 경우.
        case2 : 자기 자신의 label과 모든 이웃의 label이 같지 않음. 그러나 이웃끼리는 label이 모두 같은 경우.
        case3 : 자기 자신과 label과 모든 이웃의 label이 같지 않음. 그리고 이웃끼리도 label이 같지 않은 경우.
        case_type별로 샘플 리스트를 구성하고 난 후, key를 case_type, value를 그 case에 해당하는 sample로 구성된 리스트로 된 dictionary를 반환함.
    """
    G = nx.Graph()
    # 먼저 모든 노드를 그래프에 추가
    for i in range(data.num_nodes):
        G.add_node(i)
    # 그 다음 엣지들을 추가
    edges = list(zip(data.edge_index[0].tolist(), data.edge_index[1].tolist()))
    G.add_edges_from(edges)

    train_nodes = torch.where(mask)[0].tolist()
    # Initialize result dictionary
    case1_nodes, case2_nodes, case3_nodes = [], [], []

    for node in train_nodes:
        try:
            neighbors = list(G.neighbors(node))
            # 고립 노드 무조건 case3 (올바른 처리라고는 할 수 없음.)
            if not neighbors:
                case3_nodes.append(node)
            else:
                node_class = int(data.y[node])
                neighbor_classes = [int(data.y[neighbor]) for neighbor in neighbors]

                if all(nc == node_class for nc in neighbor_classes):
                    case1_nodes.append(node)
                elif len(set(neighbor_classes)) == 1 and neighbor_classes[0] != node_class:
                    case2_nodes.append(node)
                else:
                    case3_nodes.append(node)
        except Exception as e:
            print(f"Error processing node {node}: {str(e)}")
            continue

    return {"case1": case1_nodes, "case2": case2_nodes, "case3": case3_nodes}


@param('data.name', 'dataset')
@param('adapt.batch_size')
@param('data.supervised.ratios')
@param('adapt.method')
@param('model.backbone.model_type', 'backbone_model')
@param('model.saliency.model_type', 'saliency_model')
@param('model.answering.model_type', 'answering_model')
@param('adapt.pretrained_file')
@param('general.save_dir')
@param('adapt.repeat_times')
def run(
        dataset,
        batch_size,
        ratios,
        method,
        backbone_model,
        saliency_model,
        answering_model,
        pretrained_file,
        save_dir,
        repeat_times,
):
    # load data
    from data import get_supervised_data
    from torch_geometric.loader import DataLoader, NeighborLoader
    datasets, num_classes = get_supervised_data(dataset[0], ratios=ratios)

    # NeighborLoader 설정
    train_data = datasets['train'][0]  # 첫 번째 그래프 가져오기
    train_loader = NeighborLoader(
        data=train_data,
        num_neighbors=[-1, 10],  # 1-hop은 모든 이웃, 2-hop은 10개 샘플링
        batch_size=batch_size,
        input_nodes=None,  # 모든 노드를 시작점으로 사용
        shuffle=True,
        num_workers=4
    )

    # loader dictionary 생성
    loaders = {
        'train': train_loader,
        'val': DataLoader(datasets['val'], batch_size=batch_size, shuffle=False, num_workers=4),
        'test': DataLoader(datasets['test'], batch_size=batch_size, shuffle=False, num_workers=4)
    }

    # init model
    from model import get_model
    model = get_model(
        backbone_kwargs={
            'name': backbone_model,
            'num_features': datasets['train'][0].x.size(-1),
        },
        answering_kwargs={
            'name': answering_model,
            'num_class': num_classes,
        },
        saliency_kwargs={
            'name': saliency_model,
            'feature_dim': datasets['train'][0].x.size(-1),
        } if saliency_model != 'none' else None,
    )

    model.load_state_dict(torch.load(pretrained_file, map_location=lambda storage, loc: storage.cuda(0)), strict=False)

    # train
    all_results = []
    for _ in range(repeat_times):
        if method == 'finetune':
            results = finetune(loaders, model)
        elif method == 'prog':
            from model import get_prompt_model
            # statistic the average node number of dataset
            total_graph = sum([len(v) for k, v in datasets.items()])
            train_node_num = sum([g.num_nodes for g in datasets['train']])
            val_node_num = sum([g.num_nodes for g in datasets['val']])
            test_node_num = sum([g.num_nodes for g in datasets['test']])
            prompt_node_num = int((train_node_num + val_node_num + test_node_num) / total_graph)
            prompt_model = get_prompt_model(num_features=datasets['train'][0].x.size(-1),
                                            prompt_node_num=prompt_node_num)
            results = prog(loaders, model, prompt_model, dataset)
        elif method == 'gpf':
            results = gpf(loaders=loaders, model=model, dataset=dataset, backbone_tuning=False, saliency_tuning=False,
                          prompt_lr=1e-4, prompt_weight_decay=1e-5, prompt_basis_num=5, ans_lr=1e-2,
                          ans_weight_decay=1e-5)
        else:
            raise NotImplementedError(f'Unknown method: {method}')

        results.pop('model')
        all_results.append(results)

        # print acc, auroc, f1 with std
    import numpy as np
    for k in all_results[0].keys():
        print(f'{k}: {np.mean([r[k] for r in all_results]):.4f} ± {np.std([r[k] for r in all_results]):.4f}')

    import os

    if (method != 'prog'):
        with open(os.path.join(save_dir, dataset[0] + '_results.txt'), 'a+') as f:
            f.write('-------------------------------------------------\n')
            for k in all_results[0].keys():
                f.write(
                    method + f'FT on All, Target Dataset: {dataset[0]}, {k}: {np.mean([r[k] for r in all_results]):.4f} ± {np.std([r[k] for r in all_results]):.4f}\n')
    else:
        with open(os.path.join(save_dir, dataset[0] + '_results.txt'), 'a+') as f:
            f.write('-------------------------------------------------\n')
            for k in all_results[0].keys():
                f.write(
                    method + f' on All, Target Dataset: {dataset[0]}, {k}: {np.mean([r[k] for r in all_results]):.4f} ± {np.std([r[k] for r in all_results]):.4f}\n')

                # save
    # torch.save(results, os.path.join(save_dir, dataset[0]+'_results.pt'))


@param('adapt.finetune.backbone_tuning')
@param('adapt.finetune.saliency_tuning')
@param('adapt.finetune.learning_rate')
@param('adapt.finetune.weight_decay')
@param('adapt.epoch')
def finetune(
        loaders,
        model,
        backbone_tuning,
        saliency_tuning,
        learning_rate,
        weight_decay,
        epoch,
):
    model.backbone.requires_grad_(backbone_tuning)
    model.saliency.requires_grad_(saliency_tuning)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    from torchmetrics import MeanMetric, Accuracy, F1Score, AUROC
    from tqdm import tqdm
    from copy import deepcopy

    loss_metric = MeanMetric().to(device)
    acc_metric = Accuracy(task='multiclass', num_classes=model.answering.num_class).to(device)
    f1_metric = F1Score(task='multiclass', num_classes=model.answering.num_class, average='macro').to(device)
    auroc_metric = AUROC(task="multiclass", num_classes=model.answering.num_class).to(device)

    best_acc = 0.
    best_model = None

    for e in range(epoch):
        model.train()

        loss_metric.reset()
        acc_metric.reset()
        f1_metric.reset()
        auroc_metric.reset()

        pbar = tqdm(loaders['train'], total=len(loaders['train']), ncols=100, desc=f'Epoch {e} Training, Loss: inf')

        for batch in pbar:
            optimizer.zero_grad()
            batch = batch.to(device)
            pred = model(batch)
            loss = torch.nn.functional.cross_entropy(pred, batch.y)
            loss.backward()
            optimizer.step()

            loss_metric.update(loss.detach(), batch.size(0))
            pbar.set_description(f'Epoch {e} Training Loss: {loss_metric.compute():.4f}', refresh=True)
        pbar.close()

        model.eval()

        pbar = tqdm(loaders['val'], total=len(loaders['val']), ncols=100, desc=f'Epoch {e} Validation, Acc: 0., F1: 0.')
        with torch.no_grad():
            for batch in pbar:
                batch = batch.to(device)
                pred = model(batch).argmax(dim=-1)

                acc_metric.update(pred, batch.y)
                f1_metric.update(pred, batch.y)
                auroc_metric.update(model(batch), batch.y)
                pbar.set_description(
                    f'Epoch {e} Validation Acc: {acc_metric.compute():.4f}, AUROC: {auroc_metric.compute():.4f}, F1: {f1_metric.compute():.4f}',
                    refresh=True)
            pbar.close()

        if acc_metric.compute() > best_acc:
            best_acc = acc_metric.compute()
            best_model = deepcopy(model)

    model = best_model if best_model is not None else model

    # test
    model.eval()

    acc_metric.reset()
    f1_metric.reset()
    auroc_metric.reset()

    pbar = tqdm(loaders['test'], total=len(loaders['test']), ncols=100, desc=f'Testing, Acc: 0., F1: 0.')
    with torch.no_grad():
        for batch in pbar:
            batch = batch.to(device)
            pred = model(batch).argmax(dim=-1)

            acc_metric.update(pred, batch.y)
            f1_metric.update(pred, batch.y)
            auroc_metric.update(model(batch), batch.y)
            pbar.set_description(
                f'Testing Acc: {acc_metric.compute():.4f}, AUROC: {auroc_metric.compute():.4f}, F1: {f1_metric.compute():.4f}',
                refresh=True)
        pbar.close()

    return {
        'acc': acc_metric.compute().item(),
        'auroc': auroc_metric.compute().item(),
        'f1': f1_metric.compute().item(),
        'model': model.state_dict(),
    }


@param('adapt.epoch')
@param('adapt.prog.prompt_lr')
@param('adapt.prog.prompt_weight_decay')
@param('adapt.prog.ans_lr')
@param('adapt.prog.ans_weight_decay')
@param('adapt.prog.backbone_tuning')
@param('adapt.prog.saliency_tuning')
def prog(
        loaders,
        model,
        prompt_model,
        dataset,
        epoch,
        backbone_tuning,
        saliency_tuning,
        prompt_lr,
        prompt_weight_decay,
        ans_lr,
        ans_weight_decay,
):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.backbone.to(device)
    model.answering.to(device)
    prompt_model.to(device)

    model.backbone.requires_grad_(backbone_tuning)
    model.saliency.requires_grad_(saliency_tuning)

    opi_pg = torch.optim.Adam(
        prompt_model.parameters(),
        lr=prompt_lr,
        weight_decay=prompt_weight_decay,
    )

    opi_answer = torch.optim.Adam(
        model.answering.parameters(),
        lr=ans_lr,
        weight_decay=ans_weight_decay,
    )

    from torchmetrics import MeanMetric, Accuracy, F1Score, AUROC
    from tqdm import tqdm
    from copy import deepcopy

    loss_metric = MeanMetric().to(device)
    acc_metric = Accuracy(task='multiclass', num_classes=model.answering.num_class).to(device)
    f1_metric = F1Score(task='multiclass', num_classes=model.answering.num_class, average='macro').to(device)
    auroc_metric = AUROC(task="multiclass", num_classes=model.answering.num_class).to(device)

    # load prompting data

    from torch_geometric.loader import DataLoader

    best_acc = 0.
    best_backbone = None
    best_prompt_model = None
    best_answering = None

    for e in range(epoch):

        loss_metric.reset()
        acc_metric.reset()
        f1_metric.reset()
        auroc_metric.reset()

        print(("{}/{} frozen gnn | *tune prompt and tune answering function...".format(e, epoch)))
        prompt_model.train()
        model.backbone.eval()
        model.answering.train()

        from tqdm import tqdm

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        running_loss = 0.

        ans_pbar = tqdm(loaders['train'], total=len(loaders['train']), ncols=100,
                        desc=f'Epoch {e} / Total Epoch {epoch} Training, Loss: inf')

        for batch_id, train_batch in enumerate(ans_pbar):  # bar2

            train_batch = train_batch.to(device)
            prompted_graph = prompt_model(train_batch)

            graph_emb = model.backbone(prompted_graph)

            # print(graph_emb)
            pred = model.answering(graph_emb)
            # print(pre)
            train_loss = torch.nn.functional.cross_entropy(pred, train_batch.y)

            opi_answer.zero_grad()
            opi_pg.zero_grad()
            train_loss.backward()
            opi_answer.step()
            opi_pg.step()
            running_loss += train_loss.item()

            current_avg_last_loss = running_loss / (batch_id + 1)  # loss per batch

            ans_pbar.set_description(
                'Epoch {} / Total Epoch {} | avg loss: {:.8f}'.format(e, epoch, current_avg_last_loss), refresh=True)

        ans_pbar.close()

        model.backbone.eval()
        prompt_model.eval()
        model.answering.eval()

        pbar = tqdm(loaders['val'], total=len(loaders['val']), ncols=100, desc=f'Epoch {e} Validation, Acc: 0., F1: 0.')
        with torch.no_grad():
            for batch in pbar:
                batch = batch.to(device)
                prompted_graph = prompt_model(batch)
                z = model.backbone(prompted_graph)
                pred = model.answering(z).argmax(dim=-1)

                acc_metric.update(pred, batch.y)
                f1_metric.update(pred, batch.y)
                auroc_metric.update(model(prompted_graph), batch.y)
                pbar.set_description(
                    f'Epoch {e} Validation Acc: {acc_metric.compute():.4f}, AUROC: {auroc_metric.compute():.4f}, F1: {f1_metric.compute():.4f}',
                    refresh=True)
            pbar.close()

        if acc_metric.compute() > best_acc:
            best_acc = acc_metric.compute()
            best_backbone = deepcopy(model.backbone)
            best_answering = deepcopy(model.answering)
            best_prompt_model = deepcopy(prompt_model)

    model.backbone = best_backbone if best_backbone is not None else model.backbone
    model.answering = best_answering if best_answering is not None else model.answering
    prompt_model = best_prompt_model if best_prompt_model is not None else prompt_model

    # test
    model.backbone.eval()
    model.answering.eval()
    prompt_model.eval()

    acc_metric.reset()
    f1_metric.reset()
    auroc_metric.reset()

    pbar = tqdm(loaders['test'], total=len(loaders['test']), ncols=100, desc=f'Testing, Acc: 0., F1: 0.')
    with torch.no_grad():
        for batch in pbar:
            batch = batch.to(device)
            prompted_graph = prompt_model(batch)
            z = model.backbone(prompted_graph)
            pred = model.answering(z).argmax(dim=-1)

            acc_metric.update(pred, batch.y)
            f1_metric.update(pred, batch.y)
            auroc_metric.update(model(prompted_graph), batch.y)
            pbar.set_description(
                f'Testing Acc: {acc_metric.compute():.4f}, AUROC: {auroc_metric.compute():.4f}, F1: {f1_metric.compute():.4f}',
                refresh=True)
        pbar.close()

    return {
        'acc': acc_metric.compute().item(),
        'auroc': auroc_metric.compute().item(),
        'f1': f1_metric.compute().item(),
        'model': model.state_dict(),
    }


@param('adapt.epoch')
@param('adapt.gpf.prompt_lr')
@param('adapt.gpf.prompt_weight_decay')
@param('adapt.gpf.prompt_basis_num')
@param('adapt.gpf.ans_lr')
@param('adapt.gpf.ans_weight_decay')
@param('adapt.gpf.backbone_tuning')
@param('adapt.gpf.saliency_tuning')
def gpf(
        loaders,
        model,
        dataset,
        epoch,
        backbone_tuning,
        saliency_tuning,
        prompt_lr,
        prompt_weight_decay,
        prompt_basis_num,
        ans_lr,
        ans_weight_decay,
):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.backbone.to(device)
    model.answering.to(device)
    model.backbone.requires_grad_(backbone_tuning)
    model.saliency.requires_grad_(saliency_tuning)

    # GPF Prompt 초기화
    gpf_prompt = GPF(in_channels=loaders['train'].data.x.size(-1), p_num=prompt_basis_num).to(device)

    opi_pg = torch.optim.Adam(
        gpf_prompt.parameters(),
        lr=prompt_lr,
        weight_decay=prompt_weight_decay,
    )
    opi_answer = torch.optim.Adam(
        model.answering.parameters(),
        lr=ans_lr,
        weight_decay=ans_weight_decay,
    )

    from torchmetrics import MeanMetric, Accuracy, F1Score, AUROC
    from tqdm import tqdm
    from copy import deepcopy
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    import numpy as np

    loss_metric = MeanMetric().to(device)
    acc_metric = Accuracy(task='multiclass', num_classes=model.answering.num_class).to(device)
    f1_metric = F1Score(task='multiclass', num_classes=model.answering.num_class, average='macro').to(device)
    auroc_metric = AUROC(task="multiclass", num_classes=model.answering.num_class).to(device)

    # 모니터링할 노드 선택
    # train_data = loaders['train'].dataset[0]  # 첫 번째 그래프 데이터
    train_data = loaders['train'].data
    train_mask = torch.ones(train_data.num_nodes, dtype=torch.bool)  # 모든 노드를 대상으로
    case_dict = find_subgraph_class_distrib(train_data, train_mask)

    monitored_nodes = {
        'case1': case_dict['case1'][0] if case_dict['case1'] else None,
        'case2': case_dict['case2'][0] if case_dict['case2'] else None,
        'case3': case_dict['case3'][0] if case_dict['case3'] else None
    }

    # 모니터링 결과 저장
    monitoring_results = {
        'case1': {'original_emb': [], 'prompted_emb': [], 'distances': []},
        'case2': {'original_emb': [], 'prompted_emb': [], 'distances': []},
        'case3': {'original_emb': [], 'prompted_emb': [], 'distances': []}
    }

    best_acc = 0.
    best_prompt_model = None
    best_answering = None
    best_backbone = None

    for e in range(epoch):
        loss_metric.reset()
        acc_metric.reset()
        f1_metric.reset()
        auroc_metric.reset()

        print(f"{e}/{epoch} frozen gnn | *tune prompt and tune answering function...")
        gpf_prompt.train()
        model.backbone.eval()
        model.answering.train()

        running_loss = 0.
        ans_pbar = tqdm(loaders['train'], total=len(loaders['train']), ncols=100,
                        desc=f'Epoch {e} / Total Epoch {epoch} Training, Loss: inf')

        for batch_id, train_batch in enumerate(ans_pbar):
            train_batch = train_batch.to(device)

            # 모니터링: 원본 임베딩 저장
            with torch.no_grad():
                original_emb = model.backbone(train_batch)

                # 각 case type별 노드의 임베딩 저장
                for case, node_idx in monitored_nodes.items():
                    if node_idx is not None and node_idx < original_emb.size(0):
                        orig_emb = original_emb[node_idx].detach().cpu().numpy()
                        monitoring_results[case]['original_emb'].append(orig_emb)

            # Prompt 적용
            train_batch.x = gpf_prompt.add(train_batch.x)

            # 모니터링: prompted 임베딩 저장
            with torch.no_grad():
                prompted_emb = model.backbone(train_batch)

                # 각 case type별 노드의 prompted 임베딩 저장
                for case, node_idx in monitored_nodes.items():
                    if node_idx is not None and node_idx < prompted_emb.size(0):
                        prom_emb = prompted_emb[node_idx].detach().cpu().numpy()
                        monitoring_results[case]['prompted_emb'].append(prom_emb)

                        # Euclidean distance 계산
                        distance = np.linalg.norm(prom_emb - orig_emb)
                        monitoring_results[case]['distances'].append(distance)

            # 일반적인 training 과정
            pred = model.answering(prompted_emb)
            target = torch.tensor([train_batch.y], device=device)  # 또는
            # target = train_batch.y.clone().detach().to(device) if isinstance(train_batch.y, torch.Tensor) else torch.tensor([train_batch.y], device=device)
            train_loss = torch.nn.functional.cross_entropy(pred, target)

            opi_answer.zero_grad()
            opi_pg.zero_grad()
            train_loss.backward()
            opi_answer.step()
            opi_pg.step()

            loss_metric.update(train_loss.detach(), train_batch.num_graphs)
            ans_pbar.set_description(
                f'Epoch {e} / Total Epoch {epoch} | avg loss: {loss_metric.compute():.8f}', refresh=True)

        ans_pbar.close()

        # Validation
        model.backbone.eval()
        gpf_prompt.eval()
        model.answering.eval()

        pbar = tqdm(loaders['val'], total=len(loaders['val']), ncols=100, desc=f'Epoch {e} Validation, Acc: 0., F1: 0.')

        with torch.no_grad():
            for batch in pbar:
                batch = batch.to(device)
                batch.x = gpf_prompt.add(batch.x)
                graph_emb = model.backbone(batch)
                pred = model.answering(graph_emb)
                pred_class = pred.argmax(dim=-1)

                acc_metric.update(pred_class, batch.y)
                f1_metric.update(pred_class, batch.y)
                auroc_metric.update(pred, batch.y)

                pbar.set_description(
                    f'Epoch {e} Validation Acc: {acc_metric.compute():.4f}, AUROC: {auroc_metric.compute():.4f}, F1: {f1_metric.compute():.4f}',
                    refresh=True)
            pbar.close()

        # Best model 저장
        if acc_metric.compute() > best_acc:
            best_acc = acc_metric.compute()
            best_answering = deepcopy(model.answering)
            best_prompt_model = deepcopy(gpf_prompt)
            best_backbone = deepcopy(model.backbone)

        # 에포크마다 임베딩 변화 시각화
        plt.figure(figsize=(15, 5))
        for i, case in enumerate(['case1', 'case2', 'case3']):
            if monitored_nodes[case] is not None:
                plt.subplot(1, 3, i + 1)

                # 현재 에포크의 original과 prompted 임베딩
                orig = monitoring_results[case]['original_emb'][-1]
                prom = monitoring_results[case]['prompted_emb'][-1]

                # PCA로 차원 축소
                pca = PCA(n_components=2)
                combined = np.vstack([orig.reshape(1, -1), prom.reshape(1, -1)])
                reduced = pca.fit_transform(combined)

                # 플롯
                plt.scatter(reduced[0, 0], reduced[0, 1], label='Original', c='blue')
                plt.scatter(reduced[1, 0], reduced[1, 1], label='Prompted', c='red')
                plt.title(f'{case} - Epoch {e}')
                plt.legend()

        plt.tight_layout()
        plt.savefig(f'embedding_vis_epoch_{e}.png')
        plt.close()

        # Distance 변화 플롯
        plt.figure(figsize=(10, 6))
        for case in ['case1', 'case2', 'case3']:
            if monitored_nodes[case] is not None:
                distances = monitoring_results[case]['distances']
                plt.plot(distances, label=case)
        plt.xlabel('Training Steps')
        plt.ylabel('Original-Prompted Distance')
        plt.legend()
        plt.title('Distance Evolution During Training')
        plt.savefig(f'distance_evolution_epoch_{e}.png')
        plt.close()

    # Best model 복원
    model.backbone = best_backbone if best_backbone is not None else model.backbone
    model.answering = best_answering if best_answering is not None else model.answering
    gpf_prompt = best_prompt_model if best_prompt_model is not None else gpf_prompt

    # Test phase
    model.backbone.eval()
    model.answering.eval()
    gpf_prompt.eval()

    acc_metric.reset()
    f1_metric.reset()
    auroc_metric.reset()

    pbar = tqdm(loaders['test'], total=len(loaders['test']), ncols=100, desc=f'Testing, Acc: 0., F1: 0.')
    with torch.no_grad():
        for batch in pbar:
            batch = batch.to(device)
            batch.x = gpf_prompt.add(batch.x)
            graph_emb = model.backbone(batch)
            pred = model.answering(graph_emb)
            pred_class = pred.argmax(dim=-1)

            acc_metric.update(pred_class, batch.y)
            f1_metric.update(pred_class, batch.y)
            auroc_metric.update(pred, batch.y)

            pbar.set_description(
                f'Testing Acc: {acc_metric.compute():.4f}, AUROC: {auroc_metric.compute():.4f}, F1: {f1_metric.compute():.4f}',
                refresh=True)
        pbar.close()

    return {
        'acc': acc_metric.compute().item(),
        'auroc': auroc_metric.compute().item(),
        'f1': f1_metric.compute().item(),
        'model': model.state_dict(),
        'monitoring_results': monitoring_results  # 모니터링 결과도 반환
    }