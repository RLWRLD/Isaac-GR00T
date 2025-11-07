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
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

from gr00t.eval.robot import RobotInferenceServer
from gr00t.eval.simulation import (
    MultiStepConfig,
    SimulationConfig,
    SimulationInferenceClient,
    VideoConfig,
)
from gr00t.model.policy import Gr00tPolicy

# 24개 GR1 환경 리스트
GR1_TASK_LIST = [
    "gr1_unified/PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_Env",
    "gr1_unified/PnPPotatoToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env",
    "gr1_unified/PnPMilkToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env",
    "gr1_unified/PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_Env",
    "gr1_unified/PnPWineToCabinetClose_GR1ArmsAndWaistFourierHands_Env",
    "gr1_unified/PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_Env",
    "gr1_unified/PosttrainPnPNovelFromCuttingboardToBasketSplitA_GR1ArmsAndWaistFourierHands_Env",
    "gr1_unified/PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env",
    "gr1_unified/PosttrainPnPNovelFromCuttingboardToPanSplitA_GR1ArmsAndWaistFourierHands_Env",
    "gr1_unified/PosttrainPnPNovelFromCuttingboardToPotSplitA_GR1ArmsAndWaistFourierHands_Env",
    "gr1_unified/PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env",

    "gr1_unified/PosttrainPnPNovelFromPlacematToBasketSplitA_GR1ArmsAndWaistFourierHands_Env",
    "gr1_unified/PosttrainPnPNovelFromPlacematToBowlSplitA_GR1ArmsAndWaistFourierHands_Env",
    "gr1_unified/PosttrainPnPNovelFromPlacematToPlateSplitA_GR1ArmsAndWaistFourierHands_Env",
    "gr1_unified/PosttrainPnPNovelFromPlacematToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env",
    "gr1_unified/PosttrainPnPNovelFromPlateToBowlSplitA_GR1ArmsAndWaistFourierHands_Env",
    "gr1_unified/PosttrainPnPNovelFromPlateToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env",
    "gr1_unified/PosttrainPnPNovelFromPlateToPanSplitA_GR1ArmsAndWaistFourierHands_Env",
    "gr1_unified/PosttrainPnPNovelFromPlateToPlateSplitA_GR1ArmsAndWaistFourierHands_Env",
    "gr1_unified/PosttrainPnPNovelFromTrayToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env",
    "gr1_unified/PosttrainPnPNovelFromTrayToPlateSplitA_GR1ArmsAndWaistFourierHands_Env",
    "gr1_unified/PosttrainPnPNovelFromTrayToPotSplitA_GR1ArmsAndWaistFourierHands_Env",
    "gr1_unified/PosttrainPnPNovelFromTrayToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env",
    "gr1_unified/PosttrainPnPNovelFromTrayToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env",
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to the model checkpoint directory.",
        default="<PATH_TO_YOUR_MODEL>",  # change this to your model path
    )
    parser.add_argument(
        "--embodiment_tag",
        type=str,
        help="The embodiment tag for the model.",
        default="<EMBODIMENT_TAG>",  # change this to your embodiment tag
    )
    parser.add_argument(
        "--env_name",
        type=str,
        help="Name of the environment to run.",
        default="<ENV_NAME>",  # change this to your environment name
    )
    parser.add_argument("--port", type=int, help="Port number for the server.", default=7777)
    parser.add_argument(
        "--host", type=str, help="Host address for the server.", default="localhost"
    )
    parser.add_argument("--output_dir", type=str, help="Directory to save results and videos.", default="./outputs/gr1_tabletop/debug")
    parser.add_argument("--n_episodes", type=int, help="Number of episodes to run.", default=2)
    parser.add_argument("--n_envs", type=int, help="Number of parallel environments.", default=1)
    parser.add_argument(
        "--n_action_steps",
        type=int,
        help="Number of action steps per environment step.",
        default=16,
    )
    parser.add_argument(
        "--max_episode_steps", type=int, help="Maximum number of steps per episode.", default=1440
    )
    # server mode
    parser.add_argument("--server", action="store_true", help="Run the server.")
    # client mode
    parser.add_argument("--client", action="store_true", help="Run the client")
    # multitask evaluation
    parser.add_argument("--all_tasks", action="store_true", help="Run evaluation on all 24 GR1 tasks.")
    args = parser.parse_args()

    if args.server:
        # Create a policy
        policy = Gr00tPolicy(
            model_path=args.model_path,
            embodiment_tag=args.embodiment_tag,
        )

        # Start the server
        server = RobotInferenceServer(policy, port=args.port)
        server.run()

    elif args.client:
        # Create a simulation client
        simulation_client = SimulationInferenceClient(host=args.host, port=args.port)

        print("Available modality configs:")
        modality_config = simulation_client.get_modality_config()
        print(modality_config.keys())

        if args.all_tasks:
            # Run evaluation on all 24 GR1 tasks
            print(f"Starting evaluation on all {len(GR1_TASK_LIST)} GR1 tasks...")
            
            # Create output directory
            os.makedirs(args.output_dir, exist_ok=True)
            print(f"Output directory: {args.output_dir}")
            
            results = []
            start_time = time.time()
            
            for i, env_name in enumerate(tqdm(GR1_TASK_LIST, desc="Evaluating GR1 tasks")):
                try:
                    print(f"\n[{i+1}/{len(GR1_TASK_LIST)}] Running simulation for {env_name}...")
                    
                    # Create task-specific directory for videos
                    task_name = env_name.split('/')[-1]  # Get the last part of the env name
                    task_video_dir = os.path.join(args.output_dir, task_name)
                    os.makedirs(task_video_dir, exist_ok=True)
                    
                    # Create simulation configuration for this environment
                    config = SimulationConfig(
                        env_name=env_name,
                        n_episodes=args.n_episodes,
                        n_envs=args.n_envs,
                        video=VideoConfig(video_dir=task_video_dir),
                        multistep=MultiStepConfig(
                            n_action_steps=args.n_action_steps, max_episode_steps=args.max_episode_steps
                        ),
                    )
                    
                    # Run the simulation
                    task_start_time = time.time()
                    _, episode_successes = simulation_client.run_simulation(config)
                    task_end_time = time.time()
                    
                    # Calculate metrics
                    success_rate = np.mean(episode_successes)
                    n_successes = np.sum(episode_successes)
                    evaluation_time = task_end_time - task_start_time
                    
                    # Store results
                    result = {
                        'env_name': env_name,
                        'success_rate': success_rate,
                        'n_episodes': args.n_episodes,
                        'n_successes': n_successes,
                        'evaluation_time': evaluation_time,
                        'status': 'success'
                    }
                    results.append(result)
                    
                    print(f"  Success rate: {success_rate:.3f} ({n_successes}/{args.n_episodes})")
                    print(f"  Evaluation time: {evaluation_time:.1f}s")
                    
                except Exception as e:
                    print(f"  ERROR: Failed to evaluate {env_name}: {str(e)}")
                    result = {
                        'env_name': env_name,
                        'success_rate': 0.0,
                        'n_episodes': args.n_episodes,
                        'n_successes': 0,
                        'evaluation_time': 0.0,
                        'status': f'error: {str(e)}'
                    }
                    results.append(result)
            
            # Save results to CSV in output directory
            csv_path = os.path.join(args.output_dir, "results.csv")
            df = pd.DataFrame(results)
            df.to_csv(csv_path, index=False)
            
            # Print summary
            total_time = time.time() - start_time
            successful_evaluations = len([r for r in results if r['status'] == 'success'])
            avg_success_rate = np.mean([r['success_rate'] for r in results if r['status'] == 'success'])
            
            print(f"\n{'='*60}")
            print(f"EVALUATION SUMMARY")
            print(f"{'='*60}")
            print(f"Total tasks: {len(GR1_TASK_LIST)}")
            print(f"Successful evaluations: {successful_evaluations}")
            print(f"Failed evaluations: {len(GR1_TASK_LIST) - successful_evaluations}")
            print(f"Average success rate: {avg_success_rate:.3f}")
            print(f"Total evaluation time: {total_time:.1f}s")
            print(f"Results saved to: {csv_path}")
            print(f"{'='*60}")
            
        else:
            # Single environment evaluation (original functionality)
            # Create simulation configuration
            config = SimulationConfig(
                env_name=args.env_name,
                n_episodes=args.n_episodes,
                n_envs=args.n_envs,
                video=VideoConfig(video_dir=args.output_dir),
                multistep=MultiStepConfig(
                    n_action_steps=args.n_action_steps, max_episode_steps=args.max_episode_steps
                ),
            )

            # Run the simulation
            print(f"Running simulation for {args.env_name}...")
            env_name, episode_successes = simulation_client.run_simulation(config)

            # Print results
            print(f"Results for {env_name}:")
            print(f"Success rate: {np.mean(episode_successes):.2f}")

    else:
        raise ValueError("Please specify either --server or --client")
