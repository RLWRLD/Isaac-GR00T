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
import asyncio
import logging
import traceback
import json

import numpy as np
import websockets.asyncio.server
import websockets.frames

from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import Gr00tPolicy

import functools

import msgpack
import numpy as np


def pack_array(obj):
    if (isinstance(obj, (np.ndarray, np.generic))) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")

    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }

    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }

    return obj


def unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])

    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])

    return obj


Packer = functools.partial(msgpack.Packer, default=pack_array)
packb = functools.partial(msgpack.packb, default=pack_array)

Unpacker = functools.partial(msgpack.Unpacker, object_hook=unpack_array)
unpackb = functools.partial(msgpack.unpackb, object_hook=unpack_array)


class RobotWebsocketInferenceServer:
    """
    WebSocket 서버로 로봇 추론 서비스를 제공합니다.
    """

    def __init__(
        self,
        policy,
        host: str = "0.0.0.0",
        port: int = 8000,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def run(self) -> None:
        """서버를 시작하고 영원히 실행합니다."""
        asyncio.run(self._run())

    async def _run(self):
        """WebSocket 서버를 비동기적으로 실행합니다."""
        async with websockets.asyncio.server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
        ) as server:
            print(f"서버가 {self._host}:{self._port}에서 실행 중입니다...")
            await server.serve_forever()

    async def _handler(self, websocket: websockets.asyncio.server.ServerConnection):
        """WebSocket 연결을 처리합니다."""
        logging.info(f"연결이 {websocket.remote_address}에서 열렸습니다")
        packer = Packer()

        # 초기 메타데이터 전송 (모달리티 구성 포함)
        modality_config = self._policy.get_modality_config()

        # 직렬화 가능한 형태로 변환
        def make_serializable(obj):
            if isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(item) for item in obj]
            elif isinstance(obj, (int, float, str, bool, type(None))):
                return obj
            else:
                # 기본 타입이 아닌 객체는 문자열로 변환
                return str(obj)
    
        serializable_config = make_serializable(modality_config)
        await websocket.send(packer.pack({"modality_config": serializable_config}))

        while True:
            try:
                # 메시지 수신 및 명령 처리
                message = unpackb(await websocket.recv())
                
                # 명령 유형에 따라 처리
                if "command" in message:
                    if message["command"] == "get_modality_config":
                        response = self._policy.get_modality_config()
                        await websocket.send(packer.pack(response))
                    elif message["command"] == "get_action":
                        if "observation" in message:
                            action = self._policy.get_action(message["observation"])

                            await websocket.send(packer.pack(action))
                        else:
                            await websocket.send(packer.pack({"error": "관측값이 필요합니다"}))
                    else:
                        await websocket.send(packer.pack({"error": f"알 수 없는 명령: {message['command']}"}))
                else:
                    # 기본적으로 관측값으로 처리
                    action = self._policy.get_action(message)
                    await websocket.send(packer.pack(action))
                    
            except websockets.ConnectionClosed:
                logging.info(f"연결이 {websocket.remote_address}에서 닫혔습니다")
                break
            except Exception:
                error_traceback = traceback.format_exc()
                logging.error(f"오류 발생: {error_traceback}")
                await websocket.send(packer.pack({"error": error_traceback}))
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="내부 서버 오류가 발생했습니다.",
                )
                break


class RobotWebsocketInferenceClient:
    """
    WebSocket 클라이언트로 로봇 추론 서비스에 접속합니다.
    """

    def __init__(self, host: str = "localhost", port: int = 8000):
        self.host = host
        self.port = port
        self.websocket = None
        self.modality_config = None
        self.loop = None
        self._connect_sync()

    def _connect_sync(self):
        """서버에 동기적으로 연결합니다."""
        import asyncio
        self.loop = asyncio.new_event_loop()
        
        async def connect():
            uri = f"ws://{self.host}:{self.port}"
            self.websocket = await websockets.connect(uri, max_size=None)
            # 초기 메타데이터 수신
            initial_data = unpackb(await self.websocket.recv())
            self.modality_config = initial_data.get("modality_config", {})
            return self.modality_config
            
        self.modality_config = self.loop.run_until_complete(connect())

    def _send_command_sync(self, command, data=None):
        """명령을 서버로 동기적으로 전송합니다."""
        if not self.websocket:
            raise RuntimeError("WebSocket 연결이 없습니다")
            
        async def send_command():
            message = {"command": command}
            if data is not None:
                message.update(data)
                
            await self.websocket.send(packb(message))
            response = unpackb(await self.websocket.recv())
            return response
            
        return self.loop.run_until_complete(send_command())

    def get_modality_config(self):
        """모달리티 구성을 가져옵니다."""
        return self.modality_config

    def get_action(self, observation):
        """관측값에 대한 액션을 가져옵니다."""
        response = self._send_command_sync("get_action", {"observation": observation})
        
        # 문자열을 NumPy 배열로 변환
        for key, value in response.items():
            if isinstance(value, str) and '[' in value and ']' in value:
                try:
                    # 문자열을 NumPy 배열로 변환
                    array_str = value.replace('\n', '')
                    response[key] = np.array(eval(array_str))
                    print(f"문자열을 NumPy 배열로 변환: {key}: {response[key].shape}")
                except Exception as e:
                    print(f"배열 변환 오류 ({key}): {e}")
        
        return response

    def close(self):
        """연결을 닫습니다."""
        if self.websocket:
            async def close_connection():
                await self.websocket.close()
                
            self.loop.run_until_complete(close_connection())
            self.loop.close()
            self.websocket = None
            self.loop = None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        help="모델 체크포인트 디렉토리 경로.",
        default="nvidia/GR00T-N1-2B",
    )
    parser.add_argument(
        "--embodiment_tag",
        type=str,
        help="모델의 구현 태그.",
        default="gr1",
    )
    parser.add_argument(
        "--data_config",
        type=str,
        help="사용할 데이터 구성의 이름.",
        choices=list(DATA_CONFIG_MAP.keys()),
        default="gr1_arms_waist",
    )

    parser.add_argument("--port", type=int, help="서버의 포트 번호.", default=8000)
    parser.add_argument(
        "--host", type=str, help="서버의 호스트 주소.", default="0.0.0.0"
    )
    # 서버 모드
    parser.add_argument("--server", action="store_true", help="서버를 실행합니다.")
    # 클라이언트 모드
    parser.add_argument("--client", action="store_true", help="클라이언트를 실행합니다.")
    parser.add_argument("--denoising_steps", type=int, help="디노이징 단계 수.", default=4)
    args = parser.parse_args()

    if args.server:
        # 데이터 구성을 사용하여 모달리티 구성 및 변환 생성
        data_config = DATA_CONFIG_MAP[args.data_config]
        modality_config = data_config.modality_config()
        modality_transform = data_config.transform()

        # 정책 생성
        policy = Gr00tPolicy(
            model_path=args.model_path,
            modality_config=modality_config,
            modality_transform=modality_transform,
            embodiment_tag=args.embodiment_tag,
            denoising_steps=args.denoising_steps,
        )

        # 서버 시작
        server = RobotWebsocketInferenceServer(policy, host=args.host, port=args.port)
        server.run()

    elif args.client:
        # 클라이언트 모드에서는 서버에 랜덤 관측값을 보내고 액션을 받습니다.
        # 서버 및 클라이언트 연결을 테스트하는 데 유용합니다.

        # 정책 래퍼 생성
        policy_client = RobotWebsocketInferenceClient(host=args.host, port=args.port)

        print("사용 가능한 모달리티 구성:")
        modality_configs = policy_client.get_modality_config()
        print(modality_configs.keys())

        # 예측 수행...
        # - obs: video.ego_view: (1, 256, 256, 3)
        # - obs: state.left_arm: (1, 7)
        # - obs: state.right_arm: (1, 7)
        # - obs: state.left_hand: (1, 6)
        # - obs: state.right_hand: (1, 6)
        # - obs: state.waist: (1, 3)

        # - action: action.left_arm: (16, 7)
        # - action: action.right_arm: (16, 7)
        # - action: action.left_hand: (16, 6)
        # - action: action.right_hand: (16, 6)
        # - action: action.waist: (16, 3)
        obs = {
            "video.ego_view": np.random.randint(
                0, 256, (1, 256, 256, 3), dtype=np.uint8
            ),
            "state.left_arm": np.random.rand(1, 7),
            "state.right_arm": np.random.rand(1, 7),
            "state.left_hand": np.random.rand(1, 6),
            "state.right_hand": np.random.rand(1, 6),
            "state.waist": np.random.rand(1, 3),
            "annotation.human.action.task_description": ["do your thing!"],
        }
        action = policy_client.get_action(obs)

        for key, value in action.items():
            print(f"액션: {key}: {value.shape}")

        # 연결 닫기
        policy_client.close()

    else:
        raise ValueError("--server 또는 --client를 지정해주세요.") 