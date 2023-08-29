# 基本ライブラリ
import streamlit as st
import av
from streamlit_webrtc import WebRtcMode, webrtc_streamer
import numpy as np
from turn import get_ice_servers
from yolox.tracker.byte_tracker import BYTETracker, STrack
from onemetric.cv.utils.iou import box_iou_batch
from dataclasses import dataclass
from supervision.draw.color import ColorPalette
from supervision.geometry.dataclasses import Point
from supervision.tools.detections import Detections, BoxAnnotator
from supervision.tools.line_counter import LineCounter, LineCounterAnnotator
from typing import List
import numpy as np

import detection_model


# 利用するモデ  ルをラジオボタンで選択
radio_model = st.sidebar.radio('利用するモデルを選んでください', detection_model.set_st_radio())

# 利用するモデルをセット
model= detection_model.select_model(radio_model)

# モデルのlayer、parameterをテーブルでサイドバー表示
st.sidebar.table(data=detection_model.df_set())

st.title('柿ピー検出')

st.title("Real-time video streaming")
st.caption("リアルタイムのカメラ画像を表示します")

# converts Detections into format that can be consumed by match_detections_with_tracks function
def detections2boxes(detections: Detections) -> np.ndarray:
    return np.hstack((
        detections.xyxy,
        detections.confidence[:, np.newaxis]
    ))


# converts List[STrack] into format that can be consumed by match_detections_with_tracks function
def tracks2boxes(tracks: List[STrack]) -> np.ndarray:
    return np.array([
        track.tlbr
        for track
        in tracks
    ], dtype=float)


# matches our bounding boxes with predictions
def match_detections_with_tracks(
    detections: Detections,
    tracks: List[STrack]
) -> Detections:
    if not np.any(detections.xyxy) or len(tracks) == 0:
        return np.empty((0,))

    tracks_boxes = tracks2boxes(tracks=tracks)
    iou = box_iou_batch(tracks_boxes, detections.xyxy)
    track2detection = np.argmax(iou, axis=1)

    tracker_ids = [None] * len(detections)

    for tracker_index, detection_index in enumerate(track2detection):
        if iou[tracker_index, detection_index] != 0:
            tracker_ids[detection_index] = tracks[tracker_index].track_id

    return tracker_ids


@dataclass(frozen=True)
class BYTETrackerArgs:
    track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False

# モデルの辞書を出力
CLASS_NAMES_DICT = model.model.names
# 対象のクラスID - 柿の種、ピーナッツ
CLASS_ID = [0, 1]

# セッティング
LINE_START = Point(5, 300)
LINE_END = Point(640-5, 300)

# BYTETrackerインスタンス化
byte_tracker = BYTETracker(BYTETrackerArgs())
# LineCounterインスタンス化
line_counter = LineCounter(start=LINE_START, end=LINE_END)
# BoxAnnotatorインスタンス化
box_annotator = BoxAnnotator(color=ColorPalette(), thickness=2, text_thickness=2, text_scale=1)
# LineCounterAnnotatorインスタンス化
line_annotator = LineCounterAnnotator(thickness=2, text_thickness=2, text_scale=1)

# 推論と描写
def video_frame_callback(frame):
    #av.video.frame.VideoFrameからndarray型に変換
    frame = frame.to_ndarray(format="bgr24")

    results = model(frame, conf=0.7)

    detections = Detections(
        xyxy=results[0].boxes.xyxy.cpu().numpy(),
        confidence=results[0].boxes.conf.cpu().numpy(),
        class_id=results[0].boxes.cls.cpu().numpy().astype(int)
    )
    # tracking detections
    tracks = byte_tracker.update(
        output_results=detections2boxes(detections=detections),
        img_info=frame.shape,
        img_size=frame.shape
    )
    tracker_id = match_detections_with_tracks(detections=detections, tracks=tracks)
    detections.tracker_id = np.array(tracker_id)
    # トラッカーなしの検出を除外する
    mask = np.array([tracker_id is not None for tracker_id in detections.tracker_id], dtype=bool)
    detections.filter(mask=mask, inplace=True)
    # ラベルの書式設定
    labels = [
        f"#{tracker_id} {CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
        for _, confidence, class_id, tracker_id
        in detections
    ]
    # ラインカウンターの更新
    line_counter.update(detections=detections)

    # フレームを表示する
    frame = box_annotator.annotate(frame=frame, detections=detections, labels=labels)
    line_annotator.annotate(frame=frame, line_counter=line_counter)

    return av.VideoFrame.from_ndarray(np.array(frame), format="bgr24")


webrtc_streamer(
    key="object-detection",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={
        "iceServers": get_ice_servers(),
        "iceTransportPolicy": "relay",
    },
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

