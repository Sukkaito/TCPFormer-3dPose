import os
import cv2
import re
import json
import uuid
import base64
import asyncio
import requests
import numpy as np
from typing import Dict, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from fastdtw import fastdtw
import mediapipe as mp

import torch
from tcpformer_model import MemoryInducedTransformer

# =========================
# CONFIG
# =========================
VLM_API_URL = os.getenv("VLM_API_URL", "https://stung-ceremony-charity.ngrok-free.dev/analyze-pair").strip() 
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

ANGLE_KEYS = [
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_hip", "right_hip",
    "left_knee", "right_knee"
]

ANGLE_LABELS_VI = {
    "left_shoulder": "Vai trái",
    "right_shoulder": "Vai phải",
    "left_elbow": "Khuỷu tay trái",
    "right_elbow": "Khuỷu tay phải",
    "left_hip": "Hông trái",
    "right_hip": "Hông phải",
    "left_knee": "Đầu gối trái",
    "right_knee": "Đầu gối phải",
}

MP_IDX = {
    "left_shoulder": 11, "right_shoulder": 12,
    "left_elbow": 13, "right_elbow": 14,
    "left_wrist": 15, "right_wrist": 16,
    "left_hip": 23, "right_hip": 24,
    "left_knee": 25, "right_knee": 26,
    "left_ankle": 27, "right_ankle": 28,
}

app = FastAPI(title="Local Pose 3D Server", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
mp_pose = mp.solutions.pose

# =========================
# TCPFormer Initialization
# =========================
TCPFORMER_FRAMES = 81
tcpformer_model = None

try:
    print("Loading TCPFormer model...")
    # Initialize model
    tcpformer_model = MemoryInducedTransformer(
        n_layers=16, dim_in=3, dim_feat=128, mlp_ratio=4, hierarchical=False,
        use_tcn=False, graph_only=False, n_frames=TCPFORMER_FRAMES
    ).to(DEVICE)
    
    ckpt_path = 'TCPFormer_ap3d_81.pth.tr'
    if os.path.exists(ckpt_path):
        state_dict = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        model_state_dict = state_dict.get('model', state_dict)
        for k, v in model_state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        tcpformer_model.load_state_dict(new_state_dict, strict=False)
        tcpformer_model.eval()
        print("TCPFormer loaded successfully.")
    else:
        print(f"Warning: Model weights {ckpt_path} not found! TCPFormer will output random predictions.")
except Exception as e:
    print(f"Failed to load TCPFormer: {e}")
    tcpformer_model = None

# =========================
# DTO
# =========================
class ExtractRequest(BaseModel):
    videoUrl: str

class ComparePoseRequest(BaseModel):
    standardData: Dict[str, Any]
    studentData: Dict[str, Any]

class EvaluatePairwiseRequest(BaseModel):
    standardData: Dict[str, Any]
    studentData: Dict[str, Any]
    scores: Dict[str, float]
    maxPairs: int = 6

class CleanupRequest(BaseModel):
    paths: List[str]

# =========================
# HELPERS
# =========================
def download_video(video_url: str) -> str:
    local_path = f"{uuid.uuid4()}.mp4"
    r = requests.get(video_url, stream=True, timeout=60)
    if r.status_code != 200:
        raise Exception(f"Không tải được video: {video_url}")
    with open(local_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    return local_path

def calculate_angle_3d(a, b, c) -> float:
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    c = np.array(c, dtype=np.float32)
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom < 1e-6:
        return 0.0
    cosang = np.dot(ba, bc) / denom
    cosang = np.clip(cosang, -1.0, 1.0)
    return round(float(np.degrees(np.arccos(cosang))), 2)

def mp_to_h36m(lms, w, h):
    def norm_pt(landmark):
        x = landmark.x * 2 - 1
        y = (landmark.y * h) / w * 2 - h / w
        conf = landmark.visibility if hasattr(landmark, 'visibility') else 1.0
        return [x, y, conf]

    def midpoint(lm1, lm2):
        class LM: pass
        m = LM()
        m.x = (lm1.x + lm2.x) / 2
        m.y = (lm1.y + lm2.y) / 2
        m.visibility = min(lm1.visibility, lm2.visibility)
        return m

    pelvis = midpoint(lms[23], lms[24])
    neck = midpoint(lms[11], lms[12])
    spine = midpoint(pelvis, neck)
    
    # head approx
    head_top = midpoint(lms[0], lms[0])
    head_top.y -= (neck.y - lms[0].y) 

    pts = [
        pelvis, lms[24], lms[26], lms[28],
        lms[23], lms[25], lms[27],
        spine, neck, lms[0], head_top,
        lms[11], lms[13], lms[15],
        lms[12], lms[14], lms[16]
    ]
    return np.array([norm_pt(p) for p in pts], dtype=np.float32)

def h36m_to_kp3d(h36m_3d_frame):
    return {
        "left_shoulder": h36m_3d_frame[11].tolist(),
        "right_shoulder": h36m_3d_frame[14].tolist(),
        "left_elbow": h36m_3d_frame[12].tolist(),
        "right_elbow": h36m_3d_frame[15].tolist(),
        "left_wrist": h36m_3d_frame[13].tolist(),
        "right_wrist": h36m_3d_frame[16].tolist(),
        "left_hip": h36m_3d_frame[4].tolist(),
        "right_hip": h36m_3d_frame[1].tolist(),
        "left_knee": h36m_3d_frame[5].tolist(),
        "right_knee": h36m_3d_frame[2].tolist(),
        "left_ankle": h36m_3d_frame[6].tolist(),
        "right_ankle": h36m_3d_frame[3].tolist()
    }

def extract_pose_data_3d(video_url: str) -> Dict[str, Any]:
    video_path = download_video(video_url)
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception("Không mở được video")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        sample_rate = max(1, int(fps / 5))  # 5 fps
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        frame_index = 0
        valid_frames_meta = []
        h36m_2d_sequence = []

        with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as pose:
            while cap.isOpened():
                ok, frame = cap.read()
                if not ok:
                    break

                if frame_index % sample_rate != 0:
                    frame_index += 1
                    continue

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = pose.process(rgb)

                if res.pose_landmarks:
                    lms = res.pose_landmarks.landmark
                    # Check visibility
                    visible = True
                    for name, idx in MP_IDX.items():
                        if lms[idx].visibility < 0.5:
                            visible = False
                            break

                    if visible:
                        h36m_2d = mp_to_h36m(lms, width, height)
                        h36m_2d_sequence.append(h36m_2d)
                        valid_frames_meta.append({
                            "frame_index": int(frame_index),
                            "timestamp": round(frame_index / fps, 3)
                        })

                frame_index += 1

        cap.release()

        frames_data = []

        if len(h36m_2d_sequence) > 0 and tcpformer_model is not None:
            seq = np.array(h36m_2d_sequence) # [N, 17, 3]
            num_frames = len(seq)

            # Batch into 81-frame chunks
            chunk_size = TCPFORMER_FRAMES
            chunks = []
            for i in range(0, num_frames, chunk_size):
                chunk = seq[i:i+chunk_size]
                if len(chunk) < chunk_size:
                    # pad with last frame
                    pad_size = chunk_size - len(chunk)
                    pad = np.repeat(chunk[-1:], pad_size, axis=0)
                    chunk = np.concatenate([chunk, pad], axis=0)
                chunks.append(chunk)

            with torch.no_grad():
                tcpformer_model.eval()
                output_3d_seq = []
                for chunk in chunks:
                    # chunk is [81, 17, 3]
                    input_tensor = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0).to(DEVICE) # [1, 81, 17, 3]
                    out = tcpformer_model(input_tensor) # [1, 81, 17, 3]
                    output_3d_seq.extend(out[0].cpu().numpy()[:])

            # reconstruct frames
            output_3d_seq = output_3d_seq[:num_frames] # truncate padding

            for i in range(num_frames):
                kp3d = h36m_to_kp3d(output_3d_seq[i])
                angles = {
                    "left_shoulder":  calculate_angle_3d(kp3d["left_hip"], kp3d["left_shoulder"], kp3d["left_elbow"]),
                    "right_shoulder": calculate_angle_3d(kp3d["right_hip"], kp3d["right_shoulder"], kp3d["right_elbow"]),
                    "left_elbow":     calculate_angle_3d(kp3d["left_shoulder"], kp3d["left_elbow"], kp3d["left_wrist"]),
                    "right_elbow":    calculate_angle_3d(kp3d["right_shoulder"], kp3d["right_elbow"], kp3d["right_wrist"]),
                    "left_hip":       calculate_angle_3d(kp3d["left_shoulder"], kp3d["left_hip"], kp3d["left_knee"]),
                    "right_hip":      calculate_angle_3d(kp3d["right_shoulder"], kp3d["right_hip"], kp3d["right_knee"]),
                    "left_knee":      calculate_angle_3d(kp3d["left_hip"], kp3d["left_knee"], kp3d["left_ankle"]),
                    "right_knee":     calculate_angle_3d(kp3d["right_hip"], kp3d["right_knee"], kp3d["right_ankle"]),
                }
                frames_data.append({
                    "frame_index": valid_frames_meta[i]["frame_index"],
                    "timestamp": valid_frames_meta[i]["timestamp"],
                    "angles": angles,
                    "keypoints_3d": kp3d
                })

    finally:
        cleanup_video(video_path)

    return {
        "total_frames": len(frames_data),
        "fps": fps,
        "sample_rate": sample_rate,
        "video_url": video_url, 
        "frames": frames_data
    }

def extract_angle_sequence(data: Dict[str, Any], angle_key: str) -> List[float]:
    return [f["angles"][angle_key] for f in data.get("frames", []) if angle_key in f.get("angles", {})]

def calculate_dtw_score(std_seq: List[float], stu_seq: List[float]) -> float:
    if not std_seq or not stu_seq:
        return 0.0
    distance, path = fastdtw(std_seq, stu_seq, dist=lambda x, y: abs(x - y))
    avg_distance = distance / len(path) if path else 180.0
    score = 100.0 * (1.0 - (avg_distance / 45.0))
    return round(float(np.clip(score, 0.0, 100.0)), 2)

def build_angle_dtw_path(std_data: Dict[str, Any], stu_data: Dict[str, Any]) -> List[Tuple[int, int]]:
    std_frames = std_data.get("frames", [])
    stu_frames = stu_data.get("frames", [])
    if not std_frames or not stu_frames:
        return []

    std_vec = [[f["angles"].get(k, 0.0) for k in ANGLE_KEYS] for f in std_frames]
    stu_vec = [[f["angles"].get(k, 0.0) for k in ANGLE_KEYS] for f in stu_frames]

    def vec_dist(a, b):
        return float(np.mean(np.abs(np.array(a) - np.array(b))))

    _, path = fastdtw(std_vec, stu_vec, dist=vec_dist)
    return path or []

def pair_error(std_frame: Dict[str, Any], stu_frame: Dict[str, Any]) -> float:
    errs = [abs(std_frame["angles"].get(k, 0.0) - stu_frame["angles"].get(k, 0.0)) for k in ANGLE_KEYS]
    return float(np.mean(errs)) if errs else 0.0

def select_key_pairs(std_data: Dict[str, Any], stu_data: Dict[str, Any], max_pairs: int = 6):
    path = build_angle_dtw_path(std_data, stu_data)
    std_frames = std_data.get("frames", [])
    stu_frames = stu_data.get("frames", [])
    scored = []

    for i, j in path:
        if i < len(std_frames) and j < len(stu_frames):
            e = pair_error(std_frames[i], stu_frames[j])
            scored.append((i, j, e))

    scored.sort(key=lambda x: x[2], reverse=True)

    selected, used_i, used_j = [], set(), set()
    for i, j, e in scored:
        if len(selected) >= max_pairs:
            break
        if i in used_i or j in used_j:
            continue
        selected.append((i, j, e))
        used_i.add(i); used_j.add(j)

    return selected

def read_frame_by_index(video_path: str, frame_idx: int):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None

def render_pair_image(std_img, stu_img, std_ts: float, stu_ts: float, err: float):
    h = 360

    def fit(img):
        if img is None:
            return np.zeros((h, 320, 3), dtype=np.uint8)
        ih, iw = img.shape[:2]
        nw = max(1, int(iw * (h / max(ih, 1))))
        return cv2.resize(img, (nw, h))

    left = fit(std_img)
    right = fit(stu_img)
    gap = np.full((h, 20, 3), 30, dtype=np.uint8)
    canvas = np.concatenate([left, gap, right], axis=1)

    cv2.putText(canvas, f"STANDARD t={std_ts:.2f}s", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv2.putText(canvas, f"STUDENT  t={stu_ts:.2f}s", (left.shape[1]+30, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,255), 2)
    cv2.putText(canvas, f"mean angle error={err:.1f}", (10, h-18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20,20,255), 2)
    return canvas

def image_to_base64_jpg(img) -> str:
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        raise Exception("Encode ảnh lỗi")
    return base64.b64encode(buf.tobytes()).decode("utf-8")

def analyze_pair_remote(pair_img_b64: str, pair_meta: Dict[str, Any]) -> Dict[str, Any]:
    if not VLM_API_URL:
        return {
            "joint": "N/A",
            "issue": "Chưa cấu hình VLM_API_URL",
            "severity": 50.0,
            "fix": "Set VLM_API_URL trỏ tới Colab endpoint /analyze-pair",
            "confidence": 0.0
        }

    try:
        r = requests.post(
            VLM_API_URL,
            json={"image_base64": pair_img_b64, "meta": pair_meta},
            timeout=180
        )
        r.raise_for_status()
        data = r.json()
        return {
            "joint": str(data.get("joint", "Chưa rõ")),
            "issue": str(data.get("issue", "Chưa rõ")),
            "severity": float(np.clip(data.get("severity", 50), 0, 100)),
            "fix": str(data.get("fix", "Điều chỉnh tư thế theo mẫu")),
            "confidence": float(np.clip(data.get("confidence", 0.5), 0, 1)),
        }
    except Exception as e:
        return {
            "joint": "N/A",
            "issue": f"Gọi VLM thất bại: {str(e)}",
            "severity": 50.0,
            "fix": "Kiểm tra Colab endpoint /analyze-pair",
            "confidence": 0.0
        }

def cleanup_video(video_path: str) -> bool:
    try:
        if video_path and os.path.exists(video_path):
            os.remove(video_path)
            return True
    except Exception as e:
        print(f"Warning: Không xóa được video {video_path}: {str(e)}")
    return False

def get_rating(score: float) -> str:
    if score >= 90: return "Xuất sắc"
    if score >= 75: return "Khá"
    if score >= 60: return "Trung bình"
    return "Cần cải thiện"

# =========================
# ROUTES
# =========================
@app.get("/")
async def root():
    return {
        "status": "running",
        "service": "local_pose_3d",
        "vlm_api_url": VLM_API_URL or "(not set)",
        "tcpformer_loaded": tcpformer_model is not None
    }

@app.post("/api/ai/extract-template")
async def extract_template(req: ExtractRequest):
    try:
        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(executor, extract_pose_data_3d, req.videoUrl)
        return {"status": "success", "standardData": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ai/extract-student")
async def extract_student(req: ExtractRequest):
    try:
        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(executor, extract_pose_data_3d, req.videoUrl)
        return {"status": "success", "studentData": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ai/compare-pose")
async def compare_pose(req: ComparePoseRequest):
    try:
        std, stu = req.standardData, req.studentData
        scores = {k: calculate_dtw_score(extract_angle_sequence(std, k), extract_angle_sequence(stu, k)) for k in ANGLE_KEYS}
        scores["overall"] = round(sum(scores.values()) / len(ANGLE_KEYS), 2) if ANGLE_KEYS else 0.0
        return {"status": "success", "scores": scores}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ai/evaluate-pairwise-vlm")
def evaluate_pairwise_vlm(req: EvaluatePairwiseRequest):
    std_data, stu_data, scores = req.standardData, req.studentData, req.scores
    std_video_url = std_data.get("video_url", "")
    stu_video_url = stu_data.get("video_url", "")
    
    std_video = ""
    stu_video = ""
    
    try:
        if not std_video_url:
            raise Exception("standardData.video_url không hợp lệ")
        if not stu_video_url:
            raise Exception("studentData.video_url không hợp lệ")

        std_video = download_video(std_video_url)
        stu_video = download_video(stu_video_url)

        pairs = select_key_pairs(std_data, stu_data, max_pairs=max(1, min(12, req.maxPairs)))
        pair_findings = []

        for i, j, err in pairs:
            std_f = std_data["frames"][i]
            stu_f = stu_data["frames"][j]

            std_img = read_frame_by_index(std_video, std_f["frame_index"])
            stu_img = read_frame_by_index(stu_video, stu_f["frame_index"])
            pair_img = render_pair_image(std_img, stu_img, std_f["timestamp"], stu_f["timestamp"], err)
            pair_b64 = image_to_base64_jpg(pair_img)

            pair_meta = {
                "std_timestamp": std_f["timestamp"],
                "stu_timestamp": stu_f["timestamp"],
                "mean_angle_error": round(err, 2),
                "angle_delta": {k: round(abs(std_f["angles"].get(k, 0.0) - stu_f["angles"].get(k, 0.0)), 2) for k in ANGLE_KEYS}
            }

            vlm = analyze_pair_remote(pair_b64, pair_meta)
            pair_findings.append({
                "std_index": i,
                "stu_index": j,
                "std_timestamp": std_f["timestamp"],
                "stu_timestamp": stu_f["timestamp"],
                "mean_angle_error": round(err, 2),
                "vlm": vlm
            })

        if pair_findings:
            vlm_visual = round(max(0.0, 100.0 - float(np.mean([x["vlm"]["severity"] for x in pair_findings]))), 2)
            priority = max(pair_findings, key=lambda x: x["vlm"]["severity"])["vlm"]["joint"]
        else:
            vlm_visual, priority = 50.0, ""

        angle_overall = float(scores.get("overall", 0.0))
        final_overall = round(0.7 * angle_overall + 0.3 * vlm_visual, 2)

        suggestions = sorted(
            [{
                "joint": x["vlm"]["joint"],
                "issue": x["vlm"]["issue"],
                "fix": x["vlm"]["fix"],
                "severity": x["vlm"]["severity"],
                "confidence": x["vlm"]["confidence"],
                "std_timestamp": x["std_timestamp"],
                "stu_timestamp": x["stu_timestamp"],
            } for x in pair_findings],
            key=lambda z: z["severity"], reverse=True
        )[:5]

        return {
            "status": "success",
            "evaluation": {
                "rating": get_rating(final_overall),
                "priority": priority,
                "comment": f"Angle score={angle_overall}, visual score={vlm_visual}, final={final_overall}",
                "suggestions": suggestions,
            },
            "scores": {
                **scores,
                "vlm_visual": vlm_visual,
                "final_overall": final_overall
            },
            "pair_findings": pair_findings
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cleanup_video(std_video)
        cleanup_video(stu_video)

@app.post("/api/ai/cleanup")
async def cleanup(req: CleanupRequest):
    removed = []
    for p in req.paths:
        try:
            if p and os.path.exists(p):
                os.remove(p)
                removed.append(p)
        except:
            pass
    return {"status": "success", "removed": removed}

@app.get("/ping")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("local_pose_3d_server:app", host="0.0.0.0", port=8000, reload=False)