# final_pipeline_hsv_ready_to_run.py
# Full pipeline: MediaPipe hands + HSV object detection (color masks)
# Diagonal person split, robust grasp/touch/reach detection, timeouts, merges
# Outputs: annotated video, JSON log, per-person feature CSV, anomaly summary, debug log

import cv2
import numpy as np
import mediapipe as mp
import json, time, math, csv, os
from collections import defaultdict, deque, Counter

# ---------------- CONFIG ----------------
input_path = "/kaggle/input/videoss/VIDEO/p_17/webcam_output.mp4"   # change if needed
output_path = "annotated_output_hsv_final_ready_p_17.mp4"
log_json_path = "interaction_log_hsv_final_ready_p17.json"
feature_csv_path = "feature_matrix_hsv_final_ready_p17.csv"
anomaly_json_path = "anomaly_summary_hsv_final_ready_p17.json"
debug_log_path = "debug_log_hsv_final_ready_p17.txt"

FPS = 30
LOG_EVERY_N_FRAMES = 100
SHOW_PREVIEW = False

# thresholds and hyperparams
MIN_OBJ_AREA = 80.0
GRASP_FRAMES = 30
GRASP_HOLD_TOLERANCE_FRAMES = 10
GRASP_TIMEOUT_FRAMES = 45
MERGE_GAP_FRAMES = 30
START_COOLDOWN_FRAMES = 10
REACH_THRESHOLD = 100
MAX_HANDS = 4
SMOOTH_HISTORY = 15
MAX_HANDS_PER_PERSON = 2
HAND_OBJECT_IOU_THRESH = 0.7

# diagonal constants (locked scalars)
M_val = float(-0.5)
C_val = float(500.0)

# HSV color ranges
HSV_RANGES = {
    "red": [ (np.array([0,120,120]), np.array([10,255,255])),
             (np.array([160,120,120]), np.array([180,255,255])) ],
    "green": [(np.array([40,70,70]), np.array([80,255,255]))],
    "yellow": [(np.array([20,120,150]), np.array([35,255,255]))],
    "white": [(np.array([0,0,180]), np.array([180,50,255]))],
    "gray": [(np.array([0,0,50]), np.array([180,30,180]))]
}

DRAW_COLOR = {
    "red": (0,0,255), "green": (0,255,0), "yellow": (0,255,255),
    "gray": (128,128,128), "white": (255,255,255), "unknown": (200,200,200),
    "diag": (255,0,0), "hand_bbox": (120,120,120), "text": (255,255,255)
}

# ---------------- logging helper ----------------
def log_write(msg, level="INFO"):
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    line = f"{ts} [{level}] {msg}"
    # write only to file (no continuous console printing)
    with open(debug_log_path, "a") as f:
        f.write(line + "\n")

def print_progress(frame_idx):
    if frame_idx % LOG_EVERY_N_FRAMES == 0:
        print(f"Processed {frame_idx} frames...")

if os.path.exists(debug_log_path): os.remove(debug_log_path)
log_write("Starting FINAL HSV pipeline (ready to run)", "INFO")

# ---------------- helpers ----------------
def rect_center(rect):
    x1,y1,x2,y2 = rect
    return ((x1+x2)/2.0, (y1+y2)/2.0)

def iou(a,b):
    xA = max(a[0], b[0]); yA = max(a[1], b[1])
    xB = min(a[2], b[2]); yB = min(a[3], b[3])
    interW = max(0, xB-xA); interH = max(0, yB-yA)
    inter = interW * interH
    union = max(1, (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)
    return inter / union

def object_overlap_with_hand(obj_bb, hand_bb):
    xA = max(obj_bb[0], hand_bb[0]); yA = max(obj_bb[1], hand_bb[1])
    xB = min(obj_bb[2], hand_bb[2]); yB = min(obj_bb[3], hand_bb[3])
    interW = max(0, xB-xA); interH = max(0, yB-yA)
    inter = interW * interH
    obj_area = max(1, (obj_bb[2]-obj_bb[0])*(obj_bb[3]-obj_bb[1]))
    return inter / obj_area

def dist(a,b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

# ---------------- CentroidTracker ----------------
class CentroidTracker:
    def __init__(self, max_disappeared=40):
        self.next_id = 1
        self.tracks = {}
        self.max_disappeared = max_disappeared
    def register(self, bbox, frame_idx):
        tid = self.next_id; self.next_id += 1
        self.tracks[tid] = {'bbox': bbox, 'disappeared': 0, 'last_seen': frame_idx}
        return tid
    def deregister(self, tid):
        if tid in self.tracks: del self.tracks[tid]
    def update(self, rects, frame_idx):
        if len(rects) == 0:
            for tid in list(self.tracks.keys()):
                self.tracks[tid]['disappeared'] += 1
                if self.tracks[tid]['disappeared'] > self.max_disappeared:
                    self.deregister(tid)
            return {tid: self.tracks[tid]['bbox'] for tid in self.tracks}
        if len(self.tracks) == 0:
            for r in rects: self.register(r, frame_idx)
            return {tid: v['bbox'] for tid,v in self.tracks.items()}
        tids = list(self.tracks.keys())
        existing = [rect_center(self.tracks[t]['bbox']) for t in tids]
        inputs = [rect_center(r) for r in rects]
        cost = np.zeros((len(existing), len(inputs)), dtype=np.float32)
        for i,e in enumerate(existing):
            for j,inp in enumerate(inputs):
                cost[i,j] = dist(e, inp)
        pairs = sorted([(i,j,cost[i,j]) for i in range(cost.shape[0]) for j in range(cost.shape[1])], key=lambda x: x[2])
        used_r, used_c = set(), set()
        for i,j,_ in pairs:
            if i in used_r or j in used_c: continue
            used_r.add(i); used_c.add(j)
            tid = tids[i]
            self.tracks[tid]['bbox'] = rects[j]
            self.tracks[tid]['disappeared'] = 0
            self.tracks[tid]['last_seen'] = frame_idx
        unmatched_inputs = [j for j in range(len(inputs)) if j not in used_c]
        unmatched_existing = [i for i in range(len(existing)) if i not in used_r]
        for j in unmatched_inputs: self.register(rects[j], frame_idx)
        for i in unmatched_existing:
            tid = tids[i]; self.tracks[tid]['disappeared'] += 1
            if self.tracks[tid]['disappeared'] > self.max_disappeared: self.deregister(tid)
        return {tid: self.tracks[tid]['bbox'] for tid in self.tracks}

# ---------------- MediaPipe & trackers ----------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=MAX_HANDS,
                       model_complexity=1,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

obj_tracker = CentroidTracker(max_disappeared=40)
hand_tracker = CentroidTracker(max_disappeared=40)

# state
hand_palm_smooth = defaultdict(lambda: deque(maxlen=SMOOTH_HISTORY))
grasp_counters = defaultdict(int)
interaction_log = []
active_interactions = {}
interaction_miss_counts = defaultdict(int)
holding_state = {}
object_shared_state = defaultdict(lambda: False)
object_pass_count = defaultdict(int)

hand_positions_by_person = defaultdict(list)
hand_prev_pos_by_tid = {}
hand_speed_samples = defaultdict(list)
hand_distance_sums = defaultdict(float)
hand_distance_counts = defaultdict(int)

frames_with_any_interaction_by_person = defaultdict(int)
frames_with_both_hands_active_by_person = defaultdict(int)
frames_present_by_person = defaultdict(int)
object_shared_frames = defaultdict(int)
color_counts_by_person = defaultdict(lambda: defaultdict(int))

interaction_sessions = []
hand_person_history = defaultdict(lambda: deque(maxlen=SMOOTH_HISTORY))
hand_slot_history = {0: {1: None, 2: None}, 1: {1: None, 2: None}}
last_known_hand_label = {}
start_cooldown = defaultdict(lambda: -9999)

# ---------------- Video IO ----------------
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {input_path}")
fps = FPS
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (W, H))
frame_idx = 0
log_write(f"Video opened: {input_path} ({W}x{H}) fps={fps}", "INFO")

# ---------------- Utilities reused ----------------
def get_smoothed_person_id_for_hand(htid, palm_xy):
    x = float(palm_xy[0]); y = float(palm_xy[1])
    raw_pid = 0 if (y > (M_val * x + C_val)) else 1
    hand_person_history[htid].append(raw_pid)
    counts = Counter(hand_person_history[htid])
    most_common = counts.most_common()
    if not most_common: return raw_pid
    top_count = most_common[0][1]
    top_items = [item for item,count in most_common if count==top_count]
    if len(top_items) == 1: return top_items[0]
    else: return hand_person_history[htid][-1]

def distance_to_diagonal(point):
    x,y = float(point[0]), float(point[1])
    return abs(M_val * x - y + C_val) / math.sqrt(M_val * M_val + 1.0)

def rebalance_hands_by_diagonal(hand_to_person, smoothed_palm):
    per_pid = {0: [], 1: []}
    for tid, pid in hand_to_person.items():
        if pid in (0,1): per_pid[pid].append(tid)
    changed = True
    while changed:
        changed = False
        for pid in (0,1):
            tids = per_pid[pid]
            if len(tids) > MAX_HANDS_PER_PERSON:
                def get_pos(tid):
                    if tid in smoothed_palm and smoothed_palm[tid] is not None: return smoothed_palm[tid]
                    return np.array([0.0,0.0], dtype=np.float32)
                move_tid = min(tids, key=lambda t: distance_to_diagonal(get_pos(t)))
                other = 1 - pid
                if len(per_pid[other]) < MAX_HANDS_PER_PERSON:
                    hand_to_person[move_tid] = other
                    tids.remove(move_tid)
                    per_pid[other].append(move_tid)
                else:
                    hand_to_person[move_tid] = -1
                    tids.remove(move_tid)
                changed = True
    return hand_to_person

def assign_hand_slots(hand_to_person, smoothed_palm):
    per_pid = {0: [], 1: []}
    for tid,pid in hand_to_person.items():
        if pid in (0,1): per_pid[pid].append(tid)
    hand_slot_labels = {}
    for pid in (0,1):
        tids = per_pid[pid]
        sorted_by_x = sorted(tids, key=lambda t: float(smoothed_palm.get(t, (0,0))[0]))
        prev_map = hand_slot_history[pid]
        used_slots = set()
        for slot_idx in (1,2):
            prev_tid = prev_map.get(slot_idx)
            if prev_tid in sorted_by_x:
                hand_slot_labels[prev_tid] = f"H{slot_idx}"
                used_slots.add(slot_idx)
                sorted_by_x.remove(prev_tid)
        slot_candidates = [s for s in (1,2) if s not in used_slots]
        for tid, slot_idx in zip(sorted_by_x, slot_candidates):
            hand_slot_labels[tid] = f"H{slot_idx}"; used_slots.add(slot_idx)
        for slot_idx in (1,2):
            assigned_tid = None
            for tid, lab in hand_slot_labels.items():
                if lab == f"H{slot_idx}" and hand_to_person.get(tid)==pid:
                    assigned_tid = tid
            hand_slot_history[pid][slot_idx] = assigned_tid
    return hand_slot_labels

# ---------------- HSV detection ----------------
def detect_hsv_objects(frame, min_area_local=MIN_OBJ_AREA):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    kernel = np.ones((3,3), np.uint8)
    detected = []
    for cname, ranges in HSV_RANGES.items():
        mask = None
        for lo, hi in ranges:
            msk = cv2.inRange(hsv, lo, hi)
            mask = msk if mask is None else cv2.bitwise_or(mask, msk)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area_local: continue
            x,y,w,h = cv2.boundingRect(c)
            detected.append({"bbox":[x,y,x+w,y+h], "color":cname, "area":float(area)})
    return detected

# ---------------- main loop ----------------
try:
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1
        time_s = frame_idx / fps

        if frame_idx % LOG_EVERY_N_FRAMES == 0:
            print_progress(frame_idx)
            log_write(f"Frame {frame_idx} | active_interactions={len(active_interactions)} holding={len(holding_state)} objects={len(obj_tracker.tracks)}", "INFO")

        # 1) hands via MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)
        hand_rects = []; hand_landmarks_list = []; hand_labels_list = []
        if res.multi_hand_landmarks:
            handedness_info = res.multi_handedness or []
            for i, lm in enumerate(res.multi_hand_landmarks):
                xs = [p.x * W for p in lm.landmark]
                ys = [p.y * H for p in lm.landmark]
                bb = [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]
                hand_rects.append(bb)
                hand_landmarks_list.append(lm)
                lab = handedness_info[i].classification[0].label if i < len(handedness_info) else "Unknown"
                hand_labels_list.append(lab)

        # 2) update hand tracker
        tracked_hands = hand_tracker.update(hand_rects, frame_idx)

        # 3) map landmarks -> hand_info_by_tid and smooth palms
        hand_info_by_tid = {}
        for htid, hb in tracked_hands.items():
            best_iou = 0; best_idx = None
            for idx, hr in enumerate(hand_rects):
                v = iou(hb, hr)
                if v > best_iou:
                    best_iou = v; best_idx = idx
            if best_idx is not None and best_iou > 0:
                lm = hand_landmarks_list[best_idx]
                palm = np.array([lm.landmark[0].x * W, lm.landmark[0].y * H])
                tips = [np.array([lm.landmark[i].x * W, lm.landmark[i].y * H]) for i in [4,8,12,16,20]]
                label = hand_labels_list[best_idx] if best_idx < len(hand_labels_list) else "Unknown"
                if label == "Unknown" and htid in last_known_hand_label:
                    label = last_known_hand_label[htid]
                else:
                    last_known_hand_label[htid] = label
                hand_info_by_tid[htid] = {'bbox': hb, 'palm': palm, 'tips': tips, 'label': label}
                hand_palm_smooth[htid].append(palm)
            else:
                cx, cy = rect_center(hb)
                palm = np.array([cx, cy])
                label = last_known_hand_label.get(htid, "Unknown")
                hand_info_by_tid[htid] = {'bbox': hb, 'palm': palm, 'tips': [], 'label': label}
                hand_palm_smooth[htid].append(palm)

        smoothed_palm = {}
        for tid, dq in hand_palm_smooth.items():
            arr = np.array(dq)
            if arr.size == 0: continue
            smoothed_palm[tid] = arr.mean(axis=0)

        # 4) assign person ids per hand
        hand_to_person = {}
        for tid in list(hand_info_by_tid.keys()):
            p = smoothed_palm.get(tid, np.array(rect_center(hand_info_by_tid[tid]['bbox'])))
            pid_smoothed = get_smoothed_person_id_for_hand(tid, p)
            hand_to_person[tid] = pid_smoothed

        # 5) rebalance hands
        hand_to_person = rebalance_hands_by_diagonal(hand_to_person, smoothed_palm)

        # 6) assign hand slots
        hand_slot_labels = assign_hand_slots(hand_to_person, smoothed_palm)

        # 7) HSV object detection + filter objects on hands
        detected_objs_raw = detect_hsv_objects(frame, min_area_local=MIN_OBJ_AREA)
        filtered_objs = []
        for obj in detected_objs_raw:
            obb = obj["bbox"]
            keep = True
            for h_bb in hand_rects:
                overlap = object_overlap_with_hand(obb, h_bb)
                if overlap >= HAND_OBJECT_IOU_THRESH:
                    keep = False
                    break
            if keep: filtered_objs.append(obj)

        rects_full = [o['bbox'] for o in filtered_objs]

        # 8) update object tracker
        tracked_objects = obj_tracker.update(rects_full, frame_idx)
        tracked_info = {}
        for tid, tb in tracked_objects.items():
            best_color="unknown"; best_iou_val = 0.0
            for o in filtered_objs:
                v = iou(tb, o['bbox'])
                if v > best_iou_val:
                    best_iou_val = v; best_color = o['color']
            tracked_info[tid] = {'bbox': tb, 'color': best_color}

        # 9) store hand positions and speeds
        for htid, hinfo in hand_info_by_tid.items():
            pid = hand_to_person.get(htid, -1)
            if pid not in (0,1): continue
            p = smoothed_palm.get(htid, np.array(hinfo['palm']))
            hand_positions_by_person[pid].append((float(p[0]), float(p[1])))
            frames_present_by_person[pid] += 1
            prev = hand_prev_pos_by_tid.get(htid)
            if prev is not None:
                s = dist(prev, p)
                hand_speed_samples[pid].append(s)
            hand_prev_pos_by_tid[htid] = (float(p[0]), float(p[1]))
            if tracked_info:
                obj_centers = [rect_center(v['bbox']) for v in tracked_info.values()]
                dists = [dist(p, oc) for oc in obj_centers]
                min_d = min(dists) if dists else None
                if min_d is not None:
                    hand_distance_sums[pid] += min_d
                    hand_distance_counts[pid] += 1

        # 10) interaction detection & update states
        object_fingertips_by_person = defaultdict(lambda: defaultdict(int))
        observed_pairs_this_frame = set()
        for htid, hinfo in hand_info_by_tid.items():
            palm = smoothed_palm.get(htid, np.array(hinfo['palm']))
            tips = hinfo['tips']
            pid = hand_to_person.get(htid, -1)
            hand_label = hinfo.get('label','Unknown')
            if pid not in (0,1): continue

            for oid, oinfo in tracked_info.items():
                obb = oinfo['bbox']; color = oinfo['color']
                fingertips_inside = 0
                for t in tips:
                    if obb[0] <= t[0] <= obb[2] and obb[1] <= t[1] <= obb[3]:
                        fingertips_inside += 1
                if fingertips_inside > 0:
                    object_fingertips_by_person[oid][pid] += fingertips_inside

                palm_dist = float(np.linalg.norm(palm - np.array(rect_center(obb))))
                typ = "NO_INTERACTION"
                if fingertips_inside >= 3 or palm_dist < min(80, REACH_THRESHOLD * 0.5):
                    typ = "GRASPING_CANDIDATE"
                elif obb[0] <= palm[0] <= obb[2] and obb[1] <= palm[1] <= obb[3]:
                    typ = "TOUCHING"
                elif palm_dist < REACH_THRESHOLD:
                    typ = "REACHING"
                else:
                    typ = "NO_INTERACTION"

                key = (htid, oid)
                observed_pairs_this_frame.add(key)
                interaction_miss_counts[key] = 0

                if typ == "GRASPING_CANDIDATE":
                    grasp_counters[key] += 1
                else:
                    grasp_counters[key] = max(0, grasp_counters.get(key, 0) - 1)
                confirmed_grasp = grasp_counters.get(key, 0) >= GRASP_FRAMES
                final_typ = typ
                if typ == "GRASPING_CANDIDATE" and confirmed_grasp:
                    final_typ = "GRASPING"

                prev = active_interactions.get(key)
                last_start = start_cooldown.get(key, -9999)
                can_start = (frame_idx - last_start) > START_COOLDOWN_FRAMES

                if final_typ != "NO_INTERACTION" and final_typ != "GRASPING_CANDIDATE":
                    if prev is None and can_start:
                        active_interactions[key] = {'type': final_typ, 'start_frame': frame_idx, 'start_time': frame_idx/fps,
                                                   'hand_label': hand_label, 'person_id': pid, 'object_tid': oid, 'last_seen_frame': frame_idx}
                        start_cooldown[key] = frame_idx
                        interaction_log.append({'event':'interaction_start','frame': frame_idx, 'time_s': frame_idx/fps,
                                                'type': final_typ, 'hand_label': hand_label, 'person_id': pid, 'object_tid': oid})
                        log_write(f"interaction_start hand={htid} obj={oid} type={final_typ} pid={pid} frame={frame_idx}", "INFO")
                    else:
                        if prev is not None:
                            prev['last_seen_frame'] = frame_idx
                            if prev['type'] != final_typ and can_start:
                                # change type -> end previous, start new
                                s_prev = prev
                                # end previous
                                end_rec = active_interactions.pop(key)
                                duration_s = max(0.0, (frame_idx - 1 - end_rec['start_frame'])/fps)
                                interaction_sessions.append({'type': end_rec['type'], 'start_frame': end_rec['start_frame'], 'end_frame': frame_idx-1, 'duration_s': duration_s, 'hand_label': end_rec.get('hand_label','Unknown'), 'person_id': end_rec.get('person_id', -1), 'object_tid': end_rec.get('object_tid', -1)})
                                interaction_log.append({'event':'interaction_end','start_frame': end_rec['start_frame'], 'end_frame': frame_idx-1, 'duration_s': duration_s, 'type': end_rec['type'], 'hand_label': end_rec.get('hand_label','Unknown'), 'person_id': end_rec.get('person_id', -1), 'object_tid': end_rec.get('object_tid', -1)})
                                # start new
                                active_interactions[key] = {'type': final_typ, 'start_frame': frame_idx, 'start_time': frame_idx/fps,
                                                           'hand_label': hand_label, 'person_id': pid, 'object_tid': oid, 'last_seen_frame': frame_idx}
                                start_cooldown[key] = frame_idx
                                interaction_log.append({'event':'interaction_start','frame': frame_idx, 'time_s': frame_idx/fps,
                                                        'type': final_typ, 'hand_label': hand_label, 'person_id': pid, 'object_tid': oid})
                                log_write(f"interaction_restart hand={htid} obj={oid} type={final_typ} pid={pid} frame={frame_idx}", "INFO")

                    pk = f"p{pid}_o{oid}"
                    if final_typ == "GRASPING":
                        if pk not in holding_state:
                            holding_state[pk] = {"start_frame": frame_idx, "start_time": frame_idx/fps, "person_id": pid, "object_tid": oid, "object_color": color, "hand_label": hand_label}
                            interaction_log.append({'event':'pick_up','frame': frame_idx, 'time_s': frame_idx/fps, 'person_id': pid, 'object_tid': oid, 'object_color': color, 'hand_label': hand_label})
                            color_counts_by_person[pid][color] += 1
                            log_write(f"pick_up person={pid} obj={oid} frame={frame_idx}", "INFO")
                    else:
                        if pk in holding_state and final_typ != "GRASPING":
                            st = holding_state.pop(pk)
                            end_frame = frame_idx
                            duration_s = max(0.0, (end_frame - st['start_frame']) / fps)
                            interaction_log.append({'event':'put_down','start_frame': st['start_frame'], 'end_frame': end_frame, 'duration_s': duration_s, 'person_id': st['person_id'], 'object_tid': st['object_tid'], 'object_color': st['object_color'], 'end_hand_label': hand_label})
                            log_write(f"put_down person={st['person_id']} obj={st['object_tid']} dur={duration_s:.3f} frame={frame_idx}", "INFO")
                else:
                    if prev is not None:
                        # end interaction if previously active but now no interaction
                        end_rec = active_interactions.pop(key)
                        duration_s = max(0.0, (frame_idx - 1 - end_rec['start_frame']) / fps)
                        interaction_sessions.append({'type': end_rec['type'], 'start_frame': end_rec['start_frame'], 'end_frame': frame_idx-1, 'duration_s': duration_s, 'hand_label': end_rec.get('hand_label','Unknown'), 'person_id': end_rec.get('person_id', -1), 'object_tid': end_rec.get('object_tid', -1)})
                        interaction_log.append({'event':'interaction_end','start_frame': end_rec['start_frame'], 'end_frame': frame_idx-1, 'duration_s': duration_s, 'type': end_rec['type'], 'hand_label': end_rec.get('hand_label','Unknown'), 'person_id': end_rec.get('person_id', -1), 'object_tid': end_rec.get('object_tid', -1)})
                        log_write(f"interaction_end hand={htid} obj={oid} dur={duration_s:.3f} at frame={frame_idx-1}", "INFO")

        # 11) pass/shared detection
        for oid in tracked_info.keys():
            persons_with_fingers = [pid for pid in object_fingertips_by_person.get(oid, {}) if object_fingertips_by_person[oid][pid] > 0]
            shared_now = (0 in persons_with_fingers) and (1 in persons_with_fingers)
            if shared_now and not object_shared_state[oid]:
                object_shared_state[oid] = True
                object_pass_count[oid] += 1
                interaction_log.append({'event':'object_shared_start','frame': frame_idx, 'time_s': frame_idx/fps, 'object_tid': oid})
                log_write(f"object_shared_start obj={oid} frame={frame_idx}", "INFO")
            if not shared_now and object_shared_state[oid]:
                object_shared_state[oid] = False
                interaction_log.append({'event':'object_shared_end','frame': frame_idx, 'time_s': frame_idx/fps, 'object_tid': oid})
                log_write(f"object_shared_end obj={oid} frame={frame_idx}", "INFO")
            if shared_now:
                object_shared_frames[oid] += 1

        # 12) per-frame active hands per person
        active_hands_per_person = defaultdict(set)
        for key, rec in active_interactions.items():
            pid = rec.get('person_id', -1)
            lab = rec.get('hand_label', 'Unknown')
            if pid >= 0:
                active_hands_per_person[pid].add(lab)
        for pid in [0,1]:
            if len(active_hands_per_person.get(pid, set())) > 0:
                frames_with_any_interaction_by_person[pid] += 1
            if len(active_hands_per_person.get(pid, set())) >= 2:
                frames_with_both_hands_active_by_person[pid] += 1

        # 13) timeout handling for missing interactions
        for key in list(active_interactions.keys()):
            if key not in observed_pairs_this_frame:
                interaction_miss_counts[key] += 1
                if interaction_miss_counts[key] > GRASP_TIMEOUT_FRAMES:
                    rec = active_interactions.pop(key)
                    s_frame = rec['start_frame']; e_frame = frame_idx - 1
                    duration_s = max(0.0, (e_frame - s_frame) / fps)
                    interaction_sessions.append({'type': rec['type'], 'start_frame': s_frame, 'end_frame': e_frame, 'duration_s': duration_s, 'hand_label': rec.get('hand_label','Unknown'), 'person_id': rec.get('person_id', -1), 'object_tid': rec.get('object_tid', -1)})
                    interaction_log.append({'event':'interaction_end','start_frame': s_frame, 'end_frame': e_frame, 'duration_s': duration_s, 'type': rec['type'], 'hand_label': rec.get('hand_label','Unknown'), 'person_id': rec.get('person_id', -1), 'object_tid': rec.get('object_tid', -1), 'end_reason':'timeout'})
                    log_write(f"interaction_end (timeout) key={key} dur={duration_s:.3f} frame={frame_idx}", "INFO")
                    if key in interaction_miss_counts: del interaction_miss_counts[key]
            else:
                interaction_miss_counts[key] = 0

        # 14) draw overlays
        pt1 = (0, int(C_val)); pt2 = (W, int(M_val * W + C_val))
        cv2.line(frame, pt1, pt2, DRAW_COLOR['diag'], 1)
        cv2.putText(frame, "Side A: pid=0", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, DRAW_COLOR['text'], 1)
        cv2.putText(frame, "Side B: pid=1", (W-150,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, DRAW_COLOR['text'], 1)

        for oid, oinfo in tracked_info.items():
            bb = oinfo['bbox']; col = DRAW_COLOR.get(oinfo['color'], DRAW_COLOR['unknown'])
            x0,y0,x3,y3 = map(int, bb)
            cv2.rectangle(frame, (x0,y0), (x3,y3), col, 2)
            cv2.putText(frame, f"{oinfo['color']}_O{oid}", (x0, max(y0-8,10)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)

        for htid, hinfo in hand_info_by_tid.items():
            hb = hinfo['bbox']; label = hinfo.get('label','Unknown')
            x0,y0,x3,y3 = map(int, hb)
            pid = hand_to_person.get(htid, -1)
            slot_label = hand_slot_labels.get(htid, "H?")
            cv2.rectangle(frame, (x0,y0), (x3,y3), DRAW_COLOR['hand_bbox'], 1)
            pid_text = f"P{pid}" if pid in (0,1) else "P?"
            display_label = f"{label} {slot_label} {pid_text}"
            cv2.putText(frame, display_label, (x0, max(y0-8,10)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,0), 1)
            if htid in smoothed_palm:
                px,py = map(int, smoothed_palm[htid])
                cv2.circle(frame, (px,py), 4, (200,200,0), -1)

        out.write(frame)
        if SHOW_PREVIEW:
            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

finally:
    # finalize active interactions
    for key in list(active_interactions.keys()):
        rec = active_interactions.pop(key)
        s_frame = rec['start_frame']; e_frame = frame_idx
        duration_s = max(0.0, (e_frame - s_frame) / fps)
        interaction_sessions.append({'type': rec['type'], 'start_frame': s_frame, 'end_frame': e_frame, 'duration_s': duration_s, 'hand_label': rec.get('hand_label','Unknown'), 'person_id': rec.get('person_id', -1), 'object_tid': rec.get('object_tid', -1)})
        interaction_log.append({'event':'interaction_end','start_frame': s_frame, 'end_frame': e_frame, 'duration_s': duration_s, 'type': rec['type'], 'hand_label': rec.get('hand_label','Unknown'), 'person_id': rec.get('person_id', -1), 'object_tid': rec.get('object_tid', -1), 'end_reason':'finalize'})
    for pk, st in list(holding_state.items()):
        end_frame = frame_idx
        duration_s = max(0.0, (end_frame - st['start_frame']) / fps)
        interaction_log.append({'event':'put_down','start_frame': st['start_frame'], 'end_frame': end_frame, 'duration_s': duration_s, 'person_id': st['person_id'], 'object_tid': st['object_tid'], 'object_color': st.get('object_color','Unknown'), 'end_hand_label': st.get('hand_label','Unknown')})
    cap.release(); out.release()
    if SHOW_PREVIEW: cv2.destroyAllWindows()
    hands.close()

log_write("Processing finished. Computing features...", "INFO")

# ---------------- Feature computations ----------------
total_interactions = len(interaction_sessions)
num_reaches = len([s for s in interaction_sessions if s['type'].lower().startswith('reach')])
num_touches = len([s for s in interaction_sessions if s['type'].lower().startswith('touch')])
num_grasps = len([s for s in interaction_sessions if s['type'].lower().startswith('grasp')])
unique_objects_interacted = len(set([s['object_tid'] for s in interaction_sessions if s['object_tid']>=0]))

interaction_durations = [s['duration_s'] for s in interaction_sessions if s['duration_s']>0]
avg_interaction_duration = float(np.mean(interaction_durations)) if interaction_durations else 0.0

# --- Average duration per interaction type (global) ---
reach_durations = [s['duration_s'] for s in interaction_sessions if s['type'].lower().startswith('reach') and s['duration_s']>0]
touch_durations = [s['duration_s'] for s in interaction_sessions if s['type'].lower().startswith('touch') and s['duration_s']>0]
grasp_durations_all = [s['duration_s'] for s in interaction_sessions if s['type'].lower().startswith('grasp') and s['duration_s']>0]

avg_reach_duration_global = float(np.mean(reach_durations)) if reach_durations else 0.0
avg_touch_duration_global = float(np.mean(touch_durations)) if touch_durations else 0.0
avg_grasp_duration_global = float(np.mean(grasp_durations_all)) if grasp_durations_all else 0.0

starts_sorted = sorted(interaction_sessions, key=lambda x: x['start_frame'])
gaps = []
for i in range(len(starts_sorted)-1):
    this_end = starts_sorted[i]['end_frame']; next_start = starts_sorted[i+1]['start_frame']
    gaps.append(max(0.0, (next_start - this_end)/fps))
avg_interaction_gap = float(np.mean(gaps)) if gaps else 0.0

grasp_sessions = [s for s in interaction_sessions if s['type'].lower().startswith('grasp')]
grasp_sessions = sorted(grasp_sessions, key=lambda x: (x.get('person_id',-1), x.get('object_tid',-1), x['start_frame']))
merged_grasps = []
for s in grasp_sessions:
    if not merged_grasps:
        merged_grasps.append(dict(s)); continue
    last = merged_grasps[-1]
    if s.get('person_id') == last.get('person_id') and s.get('object_tid') == last.get('object_tid') and (s['start_frame'] - last['end_frame']) <= MERGE_GAP_FRAMES:
        last['end_frame'] = s['end_frame']
        last['duration_s'] = (last['end_frame'] - last['start_frame']) / fps
    else:
        merged_grasps.append(dict(s))

grasp_durations = [g['duration_s'] for g in merged_grasps]
avg_grasp_duration_all = float(np.mean(grasp_durations)) if grasp_durations else 0.0
max_grasp_duration_all = float(np.max(grasp_durations)) if grasp_durations else 0.0
min_grasp_duration_all = float(np.min(grasp_durations)) if grasp_durations else 0.0

total_object_pass_count = int(sum(object_pass_count.values()))
total_object_shared_frames = int(sum(object_shared_frames.values()))

avg_hand_speed = {}
hand_speed_var = {}
for pid in [0,1]:
    samples = hand_speed_samples.get(pid, [])
    if len(samples) > 0:
        avg_px_per_frame = float(np.mean(samples))
        avg_hand_speed[pid] = float(avg_px_per_frame * fps)
        hand_speed_var[pid] = float(np.std(samples) * fps)
    else:
        avg_hand_speed[pid] = 0.0
        hand_speed_var[pid] = 0.0

hand_label_counts = defaultdict(int)
for s in interaction_sessions:
    lbl = s.get('hand_label','Unknown'); hand_label_counts[lbl] += 1
dominant_hand_overall = max(hand_label_counts.items(), key=lambda x: x[1])[0] if hand_label_counts else "Unknown"

def convex_area_and_poly(pts):
    if len(pts) < 3: return 0.0, None
    arr = np.array(pts, dtype=np.float32)
    hull = cv2.convexHull(arr)
    area = float(cv2.contourArea(hull))
    return area, hull

area0, hull0 = convex_area_and_poly(hand_positions_by_person.get(0, []))
area1, hull1 = convex_area_and_poly(hand_positions_by_person.get(1, []))
workspace_overlap_ratio = 0.0
if hull0 is not None and hull1 is not None and area0>0 and area1>0:
    try:
        from shapely.geometry import Polygon
        poly0 = Polygon(hull0.reshape(-1,2)); poly1 = Polygon(hull1.reshape(-1,2))
        inter = poly0.intersection(poly1).area; union = poly0.union(poly1).area
        if union > 0: workspace_overlap_ratio = float(inter / union)
    except Exception:
        workspace_overlap_ratio = 0.0

total_time_s = frame_idx / fps if frame_idx>0 else 0.0
interaction_frequency_per_min = (total_interactions / total_time_s * 60.0) if total_time_s>0 else 0.0

# per-person features
per_person_features = {}
for pid in [0,1]:
    sess = [s for s in interaction_sessions if s.get('person_id')==pid]
    total_interactions_p = len(sess)
    grasp_count_p = len([s for s in sess if s['type'].lower().startswith('grasp')])
    touch_count_p = len([s for s in sess if s['type'].lower().startswith('touch')])
    reach_count_p = len([s for s in sess if s['type'].lower().startswith('reach')])
    grasp_sess = [s for s in sess if s['type'].lower().startswith('grasp')]
    grasp_durs = [s['duration_s'] for s in grasp_sess]
    avg_grasp_dur = float(np.mean(grasp_durs)) if grasp_durs else 0.0
    total_grasp_time_p = float(np.sum(grasp_durs)) if grasp_durs else 0.0
    interaction_frequency_p = (total_interactions_p / total_time_s) if total_time_s>0 else 0.0
    starts_sorted_p = sorted(sess, key=lambda x: x['start_frame'])
    gaps_p = []
    for i in range(len(starts_sorted_p)-1):
        this_end = starts_sorted_p[i]['end_frame']; next_start = starts_sorted_p[i+1]['start_frame']
        gaps_p.append(max(0.0, (next_start - this_end)/fps))
    interaction_gap_mean_p = float(np.mean(gaps_p)) if gaps_p else 0.0
    objs = [s['object_tid'] for s in sess if s.get('object_tid') is not None and s.get('object_tid')>=0]
    unique_objects_touched_p = len(set(objs))
    color_counts = color_counts_by_person.get(pid, {})
    if color_counts:
        vals = np.array(list(color_counts.values()), dtype=np.float64)
        probs = vals / vals.sum()
        color_preference_entropy = float(-np.sum(probs * np.log2(probs)))
    else:
        color_preference_entropy = 0.0
    switch_count = 0; prev_obj = None
    for s in sorted(sess, key=lambda x: x['start_frame']):
        oid = s.get('object_tid', None)
        if prev_obj is not None and oid is not None and oid != prev_obj:
            switch_count += 1
        prev_obj = oid
    object_switch_rate = (switch_count / total_time_s * 60.0) if total_time_s>0 else 0.0
    left_count = sum(1 for s in sess if s.get('hand_label','').lower().startswith('left'))
    right_count = sum(1 for s in sess if s.get('hand_label','').lower().startswith('right'))
    left_right_balance = (left_count / (right_count+left_count)) if (right_count+left_count)>0 else 0.0
    delays = []
    grouped_by_obj = defaultdict(list)
    for s in sess:
        grouped_by_obj[s.get('object_tid')].append(s)
    for oid, lst in grouped_by_obj.items():
        lefts = [x for x in lst if x.get('hand_label','').lower().startswith('left')]
        rights = [x for x in lst if x.get('hand_label','').lower().startswith('right')]
        for L in lefts:
            for R in rights:
                delays.append(abs((L['start_frame'] - R['start_frame'])/fps))
    dominant_hand_delay = float(np.mean(delays)) if delays else 0.0
    frames_with_any = frames_with_any_interaction_by_person.get(pid, 0)
    frames_with_both = frames_with_both_hands_active_by_person.get(pid, 0)
    both_hands_active_ratio = (frames_with_both / frames_with_any) if frames_with_any>0 else 0.0
    shared_frames_count = 0
    person_object_ids = set([s.get('object_tid') for s in sess if s.get('object_tid') is not None])
    for oid in person_object_ids:
        shared_frames_count += object_shared_frames.get(oid, 0)
    simultaneous_object_touch_ratio = (shared_frames_count / frames_with_any) if frames_with_any>0 else 0.0
    avg_reach_distance = hand_distance_sums.get(pid, 0.0) / max(1, hand_distance_counts.get(pid,0)) if hand_distance_counts.get(pid,0)>0 else 0.0
    workspace_area_pid = 0.0
    pts = hand_positions_by_person.get(pid, [])
    if len(pts) >= 3:
        arr = np.array(pts, dtype=np.float32)
        hull = cv2.convexHull(arr)
        workspace_area_pid = float(cv2.contourArea(hull))
    workspace_coverage_ratio = (workspace_area_pid / (W*H)) if (W*H)>0 else 0.0
    avg_hand_speed_p = avg_hand_speed.get(pid, 0.0)
    hand_speed_variability = hand_speed_var.get(pid, 0.0)
    pickups = [ev for ev in interaction_log if ev.get('event')=='pick_up' and ev.get('person_id')==pid]
    putdowns = [ev for ev in interaction_log if ev.get('event')=='put_down']
    resp_times = []
    for pu in pickups:
        oid = pu.get('object_tid')
        t_pick = pu.get('time_s', pu.get('frame', None)/fps if 'frame' in pu else None)
        if oid is None or t_pick is None: continue
        candidates = [pd for pd in putdowns if pd.get('object_tid')==oid]
        if not candidates: continue
        last_put = max(candidates, key=lambda x: x.get('time_s', x.get('frame',0)/fps))
        t_put = last_put.get('time_s', last_put.get('frame',0)/fps)
        if t_put is not None and t_pick >= t_put:
            resp_times.append(t_pick - t_put)
    interaction_response_time = float(np.mean(resp_times)) if resp_times else 0.0
    pass_count_p = 0
    for oid, cnt in object_pass_count.items():
        if oid in person_object_ids: pass_count_p += cnt
    frames_person = frames_with_any_interaction_by_person.get(pid,0)
    frames_both_touch_same_time = sum(object_shared_frames.values())
    cross_person_touch_overlap_ratio = (frames_both_touch_same_time / frames_person) if frames_person>0 else 0.0

    # --- Average durations per interaction type for this person ---
    reach_durs_p = [s['duration_s'] for s in sess if s['type'].lower().startswith('reach') and s['duration_s']>0]
    touch_durs_p = [s['duration_s'] for s in sess if s['type'].lower().startswith('touch') and s['duration_s']>0]
    grasp_durs_p = [s['duration_s'] for s in sess if s['type'].lower().startswith('grasp') and s['duration_s']>0]
    all_durs_p = [s['duration_s'] for s in sess if s['duration_s']>0]

    avg_reach_duration_p = float(np.mean(reach_durs_p)) if reach_durs_p else 0.0
    avg_touch_duration_p = float(np.mean(touch_durs_p)) if touch_durs_p else 0.0
    avg_grasp_duration_p = float(np.mean(grasp_durs_p)) if grasp_durs_p else 0.0
    avg_interaction_duration_p = float(np.mean(all_durs_p)) if all_durs_p else 0.0

    per_person_features[pid] = {
        "total_interactions": int(total_interactions_p),
        "grasp_count": int(grasp_count_p),
        "touch_count": int(touch_count_p),
        "reach_count": int(reach_count_p),
        "avg_grasp_duration": float(avg_grasp_dur),
        "avg_reach_duration": float(avg_reach_duration_p),
        "avg_touch_duration": float(avg_touch_duration_p),
        "avg_interaction_duration": float(avg_interaction_duration_p),
        "total_grasp_time": float(total_grasp_time_p),
        "interaction_frequency": float(interaction_frequency_p),
        "interaction_gap_mean": float(interaction_gap_mean_p),
        "unique_objects_touched": int(unique_objects_touched_p),
        "color_preference_entropy": float(color_preference_entropy),
        "object_switch_rate": float(object_switch_rate),
        "left_right_balance": float(left_right_balance),
        "both_hands_active_ratio": float(both_hands_active_ratio),
        "dominant_hand_delay": float(dominant_hand_delay),
        "simultaneous_object_touch_ratio": float(simultaneous_object_touch_ratio),
        "avg_reach_distance": float(avg_reach_distance),
        "workspace_coverage_ratio": float(workspace_coverage_ratio),
        "interaction_response_time": float(interaction_response_time),
        "avg_hand_speed": float(avg_hand_speed_p),
        "hand_speed_variability": float(hand_speed_variability),
        "object_pass_count": int(pass_count_p),
        "cross_person_touch_overlap_ratio": float(cross_person_touch_overlap_ratio)
    }

# ---------------- Save outputs ----------------
shared_meta = {
    "session_video": input_path,
    "duration_s": total_time_s,
    "fps": fps,
    "total_interactions": int(total_interactions),
    "num_reaches": int(num_reaches),
    "num_touches": int(num_touches),
    "num_grasps": int(num_grasps),
    "unique_objects_interacted": int(unique_objects_interacted),
    "avg_interaction_duration_s": float(avg_interaction_duration),
    "avg_interaction_gap_s": float(avg_interaction_gap),
    "avg_grasp_duration_s": float(avg_grasp_duration_all),
    "max_grasp_duration_s": float(max_grasp_duration_all),
    "min_grasp_duration_s": float(min_grasp_duration_all),
    "object_pass_count": int(total_object_pass_count),
    "total_object_shared_frames": int(total_object_shared_frames),
    "interaction_frequency_per_min": float(interaction_frequency_per_min),
    "dominant_hand_overall": dominant_hand_overall,
    "workspace_overlap_ratio": float(workspace_overlap_ratio),
    "frame_count": int(frame_idx),
    # new global averages per interaction type
    "avg_reach_duration_global": float(avg_reach_duration_global),
    "avg_touch_duration_global": float(avg_touch_duration_global),
    "avg_grasp_duration_global": float(avg_grasp_duration_global),
    "avg_interaction_duration_global": float(avg_interaction_duration)
}

summary = {
    "meta": {
        "processed_at": time.time(),
        "fps": fps,
        "GRASP_FRAMES": GRASP_FRAMES,
        "REACH_THRESHOLD": REACH_THRESHOLD,
        "diagonal_m": M_val,
        "diagonal_c": C_val
    },
    "interaction_log": interaction_log,
    "interaction_sessions": interaction_sessions,
    "merged_grasps": merged_grasps,
    "per_person_features_intermediate": per_person_features,
    "shared_features": shared_meta,
    "object_pass_count_detail": dict(object_pass_count),
    "object_shared_frames": dict(object_shared_frames),
    "color_counts_by_person": {k: dict(v) for k,v in color_counts_by_person.items()}
}

with open(log_json_path, "w") as f:
    json.dump(summary, f, indent=2)
log_write(f"Saved JSON log: {log_json_path}", "INFO")

anomaly_summary = {
    "frame_count": frame_idx,
    "total_interaction_sessions": len(interaction_sessions),
    "merged_grasp_count": len(merged_grasps),
    "short_grasps_count": len([g for g in merged_grasps if g['duration_s'] < 0.2]),
    "total_time_s": total_time_s
}
with open(anomaly_json_path, "w") as f:
    json.dump(anomaly_summary, f, indent=2)
log_write(f"Saved anomaly summary: {anomaly_json_path}", "INFO")

# write per-person CSV
columns = [
    "session_video","person_id","duration_s","fps",
    "total_interactions","grasp_count","touch_count","reach_count",
    "avg_grasp_duration","avg_reach_duration","avg_touch_duration","avg_interaction_duration","total_grasp_time","interaction_frequency","interaction_gap_mean",
    "unique_objects_touched","color_preference_entropy","object_switch_rate",
    "left_right_balance","both_hands_active_ratio","dominant_hand_delay","simultaneous_object_touch_ratio",
    "avg_reach_distance","workspace_coverage_ratio",
    "interaction_response_time","avg_hand_speed","hand_speed_variability","object_pass_count","cross_person_touch_overlap_ratio",
    "total_interactions_global","num_reaches","num_touches","num_grasps",
    "unique_objects_interacted","avg_interaction_duration_s","avg_interaction_gap_s","object_pass_count_global","workspace_overlap_ratio","frame_count",
    # global averages
    "avg_reach_duration_global","avg_touch_duration_global","avg_grasp_duration_global","avg_interaction_duration_global"
]

write_header = not os.path.exists(feature_csv_path)
with open(feature_csv_path, "a", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=columns)
    if write_header: writer.writeheader()
    for pid in [0,1]:
        pf = per_person_features.get(pid, {})
        row = {
            "session_video": input_path,
            "person_id": pid,
            "duration_s": shared_meta["duration_s"],
            "fps": fps,
            "total_interactions": pf.get("total_interactions", 0),
            "grasp_count": pf.get("grasp_count", 0),
            "touch_count": pf.get("touch_count", 0),
            "reach_count": pf.get("reach_count", 0),
            "avg_grasp_duration": pf.get("avg_grasp_duration", 0.0),
            "avg_reach_duration": pf.get("avg_reach_duration", 0.0),
            "avg_touch_duration": pf.get("avg_touch_duration", 0.0),
            "avg_interaction_duration": pf.get("avg_interaction_duration", 0.0),
            "total_grasp_time": pf.get("total_grasp_time", 0.0),
            "interaction_frequency": pf.get("interaction_frequency", 0.0),
            "interaction_gap_mean": pf.get("interaction_gap_mean", 0.0),
            "unique_objects_touched": pf.get("unique_objects_touched", 0),
            "color_preference_entropy": pf.get("color_preference_entropy", 0.0),
            "object_switch_rate": pf.get("object_switch_rate", 0.0),
            "left_right_balance": pf.get("left_right_balance", 0.0),
            "both_hands_active_ratio": pf.get("both_hands_active_ratio", 0.0),
            "dominant_hand_delay": pf.get("dominant_hand_delay", 0.0),
            "simultaneous_object_touch_ratio": pf.get("simultaneous_object_touch_ratio", 0.0),
            "avg_reach_distance": pf.get("avg_reach_distance", 0.0),
            "workspace_coverage_ratio": pf.get("workspace_coverage_ratio", 0.0),
            "interaction_response_time": pf.get("interaction_response_time", 0.0),
            "avg_hand_speed": pf.get("avg_hand_speed", 0.0),
            "hand_speed_variability": pf.get("hand_speed_variability", 0.0),
            "object_pass_count": pf.get("object_pass_count", 0),
            "cross_person_touch_overlap_ratio": pf.get("cross_person_touch_overlap_ratio", 0.0),
            "total_interactions_global": shared_meta["total_interactions"],
            "num_reaches": shared_meta["num_reaches"],
            "num_touches": shared_meta["num_touches"],
            "num_grasps": shared_meta["num_grasps"],
            "unique_objects_interacted": shared_meta["unique_objects_interacted"],
            "avg_interaction_duration_s": shared_meta["avg_interaction_duration_s"],
            "avg_interaction_gap_s": shared_meta["avg_interaction_gap_s"],
            "object_pass_count_global": shared_meta["object_pass_count"],
            "workspace_overlap_ratio": shared_meta["workspace_overlap_ratio"],
            "frame_count": shared_meta["frame_count"],
            # global averages
            "avg_reach_duration_global": shared_meta.get("avg_reach_duration_global", 0.0),
            "avg_touch_duration_global": shared_meta.get("avg_touch_duration_global", 0.0),
            "avg_grasp_duration_global": shared_meta.get("avg_grasp_duration_global", 0.0),
            "avg_interaction_duration_global": shared_meta.get("avg_interaction_duration_global", 0.0)
        }
        writer.writerow(row)

log_write(f"Saved feature CSV: {feature_csv_path}", "INFO")
log_write("Pipeline finished successfully ✅", "INFO")

# concise final prints
print("\nProcessing complete. Outputs generated:")
print(f" - Annotated video: {output_path}")
print(f" - Full JSON log: {log_json_path}")
print(f" - Feature CSV: {feature_csv_path}")
print(f" - Anomaly JSON: {anomaly_json_path}")
print(f" - Debug log: {debug_log_path}")
