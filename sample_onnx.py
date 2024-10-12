#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import time
import argparse

import cv2
import numpy as np
import onnxruntime  # type:ignore
from typing import List, Tuple


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--movie", type=str, default=None)
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument(
        "--model",
        type=str,
        default='model/rtdetrv2_r18vd_120e_coco_rerun_48.onnx',
    )
    parser.add_argument(
        '--score_th',
        type=float,
        default=0.6,
        help='Class confidence',
    )
    parser.add_argument(
        "--unuse_gpu",
        action="store_true",
    )

    args = parser.parse_args()

    return args


def main() -> None:
    # 引数解析
    args = get_args()
    cap_device: int = args.device
    cap_width: int = args.width
    cap_height: int = args.height

    if args.movie is not None:
        cap_device = args.movie
    image_path: str = args.image

    model_path: str = args.model
    score_th: float = args.score_th

    unuse_gpu: bool = args.unuse_gpu

    # カメラ準備
    if image_path is None:
        cap = cv2.VideoCapture(cap_device)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)

    # モデルロード
    providers: List[str]
    if unuse_gpu:
        providers = ['CPUExecutionProvider']
    else:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

    onnx_session: onnxruntime.InferenceSession = onnxruntime.InferenceSession(
        model_path,
        providers=providers,
    )
    input_size: List[int] = onnx_session.get_inputs()[0].shape
    input_width: int = input_size[3]
    input_height: int = input_size[2]

    print(onnx_session.get_providers())
    print('input size:', input_size)

    if image_path is not None:
        image: np.ndarray = cv2.imread(image_path)
        image_height, image_width, _ = image.shape
        original_size: np.ndarray = np.array([[image_width, image_height]],
                                             dtype=np.int64)

        # 前処理：BGR->RGB、リサイズ、正規化、NCHW
        input_image: np.ndarray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_image = cv2.resize(input_image, (input_width, input_height))
        input_image = input_image.astype(np.float32) / 255.0
        input_image = np.transpose(input_image, (2, 0, 1))
        input_image = np.expand_dims(input_image, axis=0)

        # ウォームアップ
        _ = onnx_session.run(
            None,
            input_feed={
                'images': input_image,
                "orig_target_sizes": original_size
            },
        )

        # 推論実施
        start_time: float = time.time()
        output: List[np.ndarray] = onnx_session.run(
            None,
            input_feed={
                'images': input_image,
                "orig_target_sizes": original_size
            },
        )
        labels, bboxes, scores = output
        elapsed_time: float = time.time() - start_time

        # 描画
        image = draw_debug(
            image,
            elapsed_time,
            score_th,
            labels,
            bboxes,
            scores,
        )

        cv2.imshow('RT-DETR Sample', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        while True:
            start_time = time.time()

            # カメラキャプチャ
            ret: bool
            ret, frame = cap.read()
            if not ret:
                break
            debug_image: np.ndarray = copy.deepcopy(frame)
            image_height, image_width, _ = frame.shape
            original_size = np.array(
                [[image_width, image_height]],
                dtype=np.int64,
            )

            # 前処理：BGR->RGB、リサイズ、正規化、NCHW
            input_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_image = cv2.resize(input_image, (input_width, input_height))
            input_image = input_image.astype(np.float32) / 255.0
            input_image = np.transpose(input_image, (2, 0, 1))
            input_image = np.expand_dims(input_image, axis=0)

            # 推論実施
            start_time = time.time()
            output = onnx_session.run(
                None,
                input_feed={
                    'images': input_image,
                    "orig_target_sizes": original_size
                },
            )
            labels, bboxes, scores = output
            elapsed_time = time.time() - start_time

            # 描画
            image = draw_debug(
                debug_image,
                elapsed_time,
                score_th,
                labels,
                bboxes,
                scores,
            )

            # キー処理(ESC：終了)
            key: int = cv2.waitKey(1)
            if key == 27:  # ESC
                break

            # 画面反映
            cv2.imshow('RT-DETR Sample', image)

        cap.release()
        cv2.destroyAllWindows()


def draw_debug(
    image: np.ndarray,
    elapsed_time: float,
    score_th: float,
    labels: np.ndarray,
    bboxes: np.ndarray,
    scores: np.ndarray,
) -> np.ndarray:
    debug_image: np.ndarray = copy.deepcopy(image)

    for label, bbox, score in zip(labels[0], bboxes[0], scores[0]):
        if score < score_th:
            continue

        cv2.rectangle(
            debug_image,
            (int(bbox[0]), int(bbox[1])),
            (int(bbox[2]), int(bbox[3])),
            (0, 255, 0),
            2,
        )
        cv2.putText(
            debug_image,
            str(label),
            (int(bbox[0]), int(bbox[1] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

    # 推論時間
    text = 'Elapsed time:' + '%.0f' % (elapsed_time * 1000)
    text = text + 'ms'
    debug_image = cv2.putText(
        debug_image,
        text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        thickness=2,
    )

    return debug_image


if __name__ == '__main__':
    main()
