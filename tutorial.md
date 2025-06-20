## 문서의 목적
이 문서는 텔레옵시스템 등을 통해 학습데이터를 수집한 후 slurm 클러스터에서 학습을 진행하고 서빙 장비에서 테스트/추론을 진행하는 방법을 설명합니다. 
더 자세한 내용은 [Isaac-GR00T 프로젝트 페이지](https://github.com/RLWRLD/Isaac-GR00T)를 참고해주세요.

## 작업 절차
1. Slurm 클러스트에 Isaac-GR00T 설치 
```sh
ssh -i ~/.ssh/id_rsa storm@210.109.59.241
git clone https://github.com/RLWRLD/Isaac-GR00T.git
```


2. 학습데이터 복사 - Robosuite를 통해 생성한 학습데이터를 slurm 클러스트로 복사합니다. (RTX4090 장비에 학습데이터 저장되어 있는 것 가정)
```sh
scp -r allex-cube-dataset storm@61.109.237.73:/virtual_lab/rlwrld/storm/Isaa-GR00t/demo_data/
```

3. 학습 준비 - 신규 데이터 적용을 위한 modality 설정 파일 추가 및 필요한 코드 수정
```sh
ssh -i ~/.ssh/id_rsa storm@210.109.59.241
cd Isaac-GR00T
# 샘플 파일 복사후 meta/info.json 참고하여 적절히 수정
cp ./demo_data/modality_sample.json ./demo_data/sim_pick_place_v1/meta/modality.json
# 필요한 코드 수정
# gr00t/experiment/data_config.py 에 신규 데이터 설정 추가 (aloha_put_cube, franka_hook 등을 참고하여 코드 작성)
```

4. Slurm 클러스트에서 학습 진행
```sh
# 환경 설정
python3 -m venv venv
source venv/bin/activate
pip install --upgrade setuptools
pip install -e .
pip install --no-build-isolation flash-attn==2.7.1.post4
# wandb 계정 없으면 생성
wandb login 
# slurm batch job 스크립트 작성 - 아래처럼 복사 후 변수명 적절히 변경
cp ./scripts/slurm_ft_sample.batch ./slurm_ft_sim_pick_place_v1.batch 
# 학습 진행 및 잘 되는지 확인
sbatch slurm_ft_sim_pick_place_v1.batch
squeue
tail -f tmp/slurm-xxxx*.log
# 학습 완료 후 학습된 결과를 RTX4090 장비에 복사 (체크포인트는 모두 삭제하고 진행하는 것이 빠름)
```

5. 서빙 장비에서 테스트/추론 진행 - RTX4090 장비에서 진행하는 것을 가정
```sh
ssh storm@172.30.1.102
git clone https://github.com/RLWRLD/Isaac-GR00T.git
cd Isaac-GR00T
# 환경 설정
conda create -n gr00t python=3.10
conda activate gr00t
pip install --upgrade setuptools
pip install -e .
pip install --no-build-isolation flash-attn==2.7.1.post4
# 터미널 1에서 서버 실행
python scripts/inference_service.py --server \
    --model_path finetuned_models/sim_pick_place_v1 \
    --embodiment_tag new_embodiment
    --data_config sim_pick_place_v1
# 터미널 2에서 offline 평가 진행
python scripts/eval_policy.py --plot \
    --dataset_path demo_data/sim_pick_place_v1 \
    --embodiment_tag new_embodiment \
    --data_config sim_pick_place_v1
```
