# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import warnings

import numpy as np

from gr00t.data.dataset import LeRobotSingleDataset
from scripts.inference_service_with_websocket import RobotWebsocketInferenceClient
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import BasePolicy, Gr00tPolicy
from gr00t.utils.eval import calc_mse_for_single_trajectory

warnings.simplefilter("ignore", category=FutureWarning)

"""
Example command:

python scripts/eval_policy_with_websocket.py --host localhost --port 8000 --plot
    --modality_keys right_arm right_hand
    --steps 250
    --trajs 1000
    --action_horizon 16
    --video_backend decord
    --dataset_path demo_data/robot_sim.PickNPlace/
    --embodiment_tag gr1
    --data_config gr1_arms_waist
provide --model_path to load up the model checkpoint in this script.
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost", help="호스트")
    parser.add_argument("--port", type=int, default=8000, help="포트")
    parser.add_argument("--plot", action="store_true", help="이미지 플롯")
    parser.add_argument("--modality_keys", nargs="+", type=str, default=["right_arm", "right_hand"])
    parser.add_argument(
        "--data_config",
        type=str,
        default="gr1_arms_waist",
        choices=list(DATA_CONFIG_MAP.keys()),
        help="데이터 구성 이름",
    )
    parser.add_argument("--steps", type=int, default=150, help="실행할 단계 수")
    parser.add_argument("--trajs", type=int, default=1, help="실행할 궤적 수")
    parser.add_argument("--action_horizon", type=int, default=16)
    parser.add_argument("--video_backend", type=str, default="decord")
    parser.add_argument("--dataset_path", type=str, default="demo_data/robot_sim.PickNPlace/")
    parser.add_argument(
        "--embodiment_tag",
        type=str,
        help="모델의 구현 태그",
        default="gr1",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="[선택사항] 모델 체크포인트 디렉토리 경로, 이 옵션을 사용하면 클라이언트-서버 모드가 비활성화됩니다.",
    )
    args = parser.parse_args()

    data_config = DATA_CONFIG_MAP[args.data_config]
    if args.model_path is not None:
        import torch

        modality_config = data_config.modality_config()
        modality_transform = data_config.transform()

        policy: BasePolicy = Gr00tPolicy(
            model_path=args.model_path,
            modality_config=modality_config,
            modality_transform=modality_transform,
            embodiment_tag=args.embodiment_tag,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
    else:
        # WebSocket 클라이언트 사용
        policy: BasePolicy = RobotWebsocketInferenceClient(host=args.host, port=args.port)

    all_gt_actions = []
    all_pred_actions = []

    # 정책에서 지원하는 모달리티 가져오기
    modality = policy.get_modality_config()

    # WebSocket 클라이언트를 사용하는 경우, 모달리티 구성을 직접 생성
    if args.model_path is None:
        # 서버에서 받은 모달리티 구성이 문자열이거나 직렬화된 형태인 경우
        # 데이터 구성에서 직접 모달리티 구성을 가져옵니다
        modality = data_config.modality_config()

    print(f'DBG11 {modality}')

    # 데이터셋 생성
    dataset = LeRobotSingleDataset(
        dataset_path=args.dataset_path,
        modality_configs=modality,
        video_backend=args.video_backend,
        video_backend_kwargs=None,
        transforms=None,  # 정책을 통해 변환을 별도로 처리합니다
        embodiment_tag=args.embodiment_tag,
    )

    print(len(dataset))
    # 예측 수행
    obs = dataset[0]

    for k, v in obs.items():
        if isinstance(v, np.ndarray):
            print(k, v.shape)
        else:
            print(k, v)

    for k, v in dataset.get_step_data(0, 0).items():
        if isinstance(v, np.ndarray):
            print(k, v.shape)
        else:
            print(k, v)

    print("총 궤적 수:", len(dataset.trajectory_lengths))
    print("모든 궤적:", dataset.trajectory_lengths)
    print("모달리티 키로 모든 궤적 실행 중:", args.modality_keys)

    all_mse = []
    for traj_id in range(args.trajs):
        print("궤적 실행 중:", traj_id)
        mse = calc_mse_for_single_trajectory(
            policy,
            dataset,
            traj_id,
            modality_keys=args.modality_keys,
            steps=args.steps,
            action_horizon=args.action_horizon,
            plot=args.plot,
        )
        print("MSE:", mse)
        all_mse.append(mse)
    print("모든 궤적의 평균 MSE:", np.mean(all_mse))
    print("완료")
    
    # 클라이언트 연결 닫기
    if args.model_path is None:
        policy.close()
        
    exit() 