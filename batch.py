import os
import subprocess

def pretrain(target_datasets, datasets): 
        # 외부 루프: 타겟 데이터셋 반복
    for target_dataset in target_datasets:
        source_dataset_str = ""

        # 내부 루프: 데이터셋 배열에서 타겟 데이터셋을 제외한 소스 데이터셋 문자열 생성
        source_datasets = [ds for ds in datasets if ds != target_dataset]
        source_dataset_str = ",".join(source_datasets)

        # 쉼표로 구분된 문자열에서 마지막 쉼표 제거는 필요 없음 (Python에서는 자동 처리)
        print(f"Source datasets: {source_dataset_str}")
        print(f"Target dataset: {target_dataset}")
        pretrained_model_path = f"storage/reconstruct/{source_dataset_str.replace(',', '_')}_pretrained_model.pt"
        print(pretrained_model_path)

        # 첫 번째 Python 명령어 실행 (사전 학습)
        pretrain_cmd = [
            'python', 'src/exec.py', '--config-file', 'pretrain.json',
            '--general.save_dir', f'storage/{backbone}/reconstruct',
            '--general.reconstruct', '0.2',
            '--data.name', source_dataset_str,
            '--pretrain.split_method', split_method,
            '--model.backbone.model_type', backbone
        ]

        try:
            subprocess.run(pretrain_cmd, check=True)  # check=True로 설정하여 실패 시 예외 발생
        except subprocess.CalledProcessError as e:
            print(f"Error during pre-training: {e}")
            continue  # 에러가 발생해도 다음 작업으로 넘어감

def adapt(target_datasets, datasets): 
        # 외부 루프: 타겟 데이터셋 반복
    for target_dataset in target_datasets:
        source_dataset_str = ""

        # 내부 루프: 데이터셋 배열에서 타겟 데이터셋을 제외한 소스 데이터셋 문자열 생성
        #source_datasets = [ds for ds in datasets if ds != target_dataset]
        source_datasets = [ds for ds in datasets]
        source_dataset_str = ",".join(source_datasets)

        # 쉼표로 구분된 문자열에서 마지막 쉼표 제거는 필요 없음 (Python에서는 자동 처리)
        print(f"Source datasets: {source_dataset_str}")
        print(f"Target dataset: {target_dataset}")

        source_dataset_str = source_dataset_str.replace(',', '_')
        pretrained_model_path = f"storage/{backbone}/reconstruct/{source_dataset_str}_pretrained_model.pt"
        print(pretrained_model_path)

        # 내부 루프: 학습률과 배치 크기 조합에 따라 미세 조정 실행
        for lr in learning_rates:
            for batch_size in batch_sizes:
                """
                fine_tune_cmd = [
                    'python', 'src/exec.py', '--general.func', 'adapt',
                    '--general.save_dir', f'storage/{backbone}/balanced_few_shot_fine_tune_backbone_with_rec',
                    '--general.few_shot', str(few_shot),
                    '--general.reconstruct', '0.0',
                    '--data.node_feature_dim', '100',
                    '--data.name', target_dataset,
                    '--adapt.method', 'finetune',
                    '--model.backbone.model_type', backbone,
                    '--model.saliency.model_type', 'none',
                    '--adapt.pretrained_file', pretrained_model_path,
                    '--adapt.finetune.learning_rate', lr,
                    '--adapt.batch_size', str(batch_size),
                    '--adapt.finetune.backbone_tuning', str(backbone_tuning)
                ]
                """
                fine_tune_cmd = [
                    'python', 'src/exec.py', '--general.func', 'adapt',
                    '--general.save_dir', f'storage/{backbone}/balanced_few_shot_fine_tune_backbone_with_rec',
                    '--general.few_shot', str(few_shot),
                    '--general.reconstruct', '0.0',
                    '--data.node_feature_dim', '100',
                    '--data.name', target_dataset,
                    '--adapt.method', 'gpf',
                    '--model.backbone.model_type', backbone,
                    '--model.saliency.model_type', 'none',
                    '--adapt.pretrained_file', pretrained_model_path,
                    '--adapt.batch_size', str(batch_size),
                ]
                try:
                    subprocess.run(fine_tune_cmd, check=True)
                except subprocess.CalledProcessError as e:
                    print(f"Error during fine-tuning: {e}")
                    continue 

def ete(target_datasets, datasets): 
        # 외부 루프: 타겟 데이터셋 반복
    for target_dataset in target_datasets:
        source_dataset_str = ""

        # 내부 루프: 데이터셋 배열에서 타겟 데이터셋을 제외한 소스 데이터셋 문자열 생성
        source_datasets = [ds for ds in datasets if ds != target_dataset]
        source_dataset_str = ",".join(source_datasets)

        # 쉼표로 구분된 문자열에서 마지막 쉼표 제거는 필요 없음 (Python에서는 자동 처리)
        print(f"Source datasets: {source_dataset_str}")
        print(f"Target dataset: {target_dataset}")
        pretrained_model_path = f"storage/reconstruct/{source_dataset_str.replace(',', '_')}_pretrained_model.pt"
        print(pretrained_model_path)

        # 첫 번째 Python 명령어 실행 (사전 학습)
        pretrain_cmd = [
            'python', 'src/exec.py', '--config-file', 'pretrain.json',
            '--general.save_dir', f'storage/{backbone}/reconstruct',
            '--general.reconstruct', '0.2',
            '--data.name', source_dataset_str,
            '--pretrain.split_method', split_method,
            '--model.backbone.model_type', backbone
        ]

        try:
            subprocess.run(pretrain_cmd, check=True)  # check=True로 설정하여 실패 시 예외 발생
        except subprocess.CalledProcessError as e:
            print(f"Error during pre-training: {e}")
            continue  # 에러가 발생해도 다음 작업으로 넘어감

        source_dataset_str = source_dataset_str.replace(',', '_')
        pretrained_model_path = f"storage/reconstruct/{source_dataset_str}_pretrained_model.pt"
        print(pretrained_model_path)

        # 내부 루프: 학습률과 배치 크기 조합에 따라 미세 조정 실행
        for lr in learning_rates:
            for batch_size in batch_sizes:
                fine_tune_cmd = [
                    'python', 'src/exec.py', '--general.func', 'adapt',
                    '--general.save_dir', f'storage/{backbone}/balanced_few_shot_fine_tune_backbone_with_rec',
                    '--general.few_shot', str(few_shot),
                    '--general.reconstruct', '0.0',
                    '--data.node_feature_dim', '100',
                    '--data.name', target_dataset,
                    '--adapt.method', 'finetune',
                    '--model.backbone.model_type', backbone,
                    '--model.saliency.model_type', 'none',
                    '--adapt.pretrained_file', pretrained_model_path,
                    '--adapt.finetune.learning_rate', lr,
                    '--adapt.batch_size', str(batch_size),
                    '--adapt.finetune.backbone_tuning', str(backbone_tuning)
                ]
                try:
                    subprocess.run(fine_tune_cmd, check=True)
                except subprocess.CalledProcessError as e:
                    print(f"Error during fine-tuning: {e}")
                    continue 

# 설정된 파라미터 정의
backbone = 'fagcn'
backbone_tuning = 1
split_method = 'RandomWalk'
few_shot = 1

# 학습률 배열 설정
learning_rates = ['1e-2']

# 배치 크기 배열 설정
batch_sizes = [100]

# ['cora', 'citeseer', 'pubmed', 'computers', 'photo', 'wisconsin', 'texas', 'cornell', 'chameleon', 'squirrel']

# 타겟 데이터셋 설정
target_datasets_list = [
    #['cora', 'citeseer', 'cornell', 'chameleon', 'squirrel']
    ['chameleon']
]

# 데이터셋 배열 설정
datasets_list = [
    #['cora', 'citeseer', 'pubmed', 'computers', 'photo', 'wisconsin', 'texas', 'cornell', 'chameleon', 'squirrel']
    ['cora', 'citeseer', 'pubmed', 'computers', 'photo']
]

for i in range(len(datasets_list)):
    target_datasets = target_datasets_list[0]
    datasets = datasets_list[i]

    #pretrain(target_datasets=target_datasets, datasets=datasets)
    adapt(target_datasets=target_datasets, datasets=datasets)
