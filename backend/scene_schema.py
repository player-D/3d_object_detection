import math


def estimate_ego_speed_mps(nusc, sample_token):
    sample = nusc.get("sample", sample_token)
    prev_token = sample.get("prev")
    if not prev_token:
        return 0.0

    current_pose = _get_sample_pose(nusc, sample_token)
    prev_pose = _get_sample_pose(nusc, prev_token)
    if current_pose is None or prev_pose is None:
        return 0.0

    dt = max((sample["timestamp"] - nusc.get("sample", prev_token)["timestamp"]) / 1e6, 1e-3)
    dx = current_pose["translation"][0] - prev_pose["translation"][0]
    dy = current_pose["translation"][1] - prev_pose["translation"][1]
    return math.sqrt(dx * dx + dy * dy) / dt


def _get_sample_pose(nusc, sample_token):
    sample = nusc.get("sample", sample_token)
    cam_token = sample["data"].get("CAM_FRONT")
    if not cam_token:
        return None
    sample_data = nusc.get("sample_data", cam_token)
    return nusc.get("ego_pose", sample_data["ego_pose_token"])


def compute_threat_level(longitudinal_m, lateral_m, cls_name, score=1.0):
    if longitudinal_m < -4.0 or longitudinal_m > 65.0:
        return "low"

    abs_lat = abs(lateral_m)
    cls_bonus = 0.12 if cls_name in {"pedestrian", "bicycle", "motorcycle"} else 0.0
    dist_score = max(0.0, 1.0 - longitudinal_m / 45.0)
    lane_score = max(0.0, 1.0 - abs_lat / 3.8)
    confidence_score = max(0.25, min(1.0, float(score)))
    total = 0.55 * dist_score + 0.30 * lane_score + 0.15 * confidence_score + cls_bonus

    if total >= 0.72:
        return "high"
    if total >= 0.42:
        return "medium"
    return "low"


def _build_lane(points, lane_id, kind):
    return {
        "id": lane_id,
        "kind": kind,
        "points": [{"x": x, "y": y, "z": 0.0} for x, y in points],
    }


def build_lane_vectors():
    longitudinal = [float(step) for step in range(-8, 73, 4)]
    center_points = [(x, 0.0) for x in longitudinal]
    left_points = [(x, 3.6) for x in longitudinal]
    right_points = [(x, -3.6) for x in longitudinal]
    far_left_points = [(x, 7.2) for x in longitudinal]
    far_right_points = [(x, -7.2) for x in longitudinal]

    corridor = []
    for x in range(0, 65, 3):
        width = 1.7 + x * 0.055
        corridor.append((float(x), float(width)))

    corridor_points = [{"x": x, "left_y": y, "right_y": -y} for x, y in corridor]
    return {
        "centerlines": [
            _build_lane(center_points, "ego-center", "centerline"),
            _build_lane(left_points, "adjacent-left", "centerline"),
            _build_lane(right_points, "adjacent-right", "centerline"),
        ],
        "boundaries": [
            _build_lane(far_left_points, "left-boundary", "boundary"),
            _build_lane(far_right_points, "right-boundary", "boundary"),
        ],
        "corridor": corridor_points,
    }


def _serialize_box(box):
    center_x, center_y, center_z = [float(value) for value in box.center]
    width, length, height = [float(value) for value in box.wlh]
    yaw = float(box.orientation.yaw_pitch_roll[0])
    return {
        "center": {"x": center_x, "y": center_y, "z": center_z},
        "size": {"width": width, "length": length, "height": height},
        "yaw": yaw,
    }


def _serialize_object(object_id, source, cls_name, box, score, label):
    longitudinal = float(box.center[0])
    lateral = float(box.center[1])
    threat_level = compute_threat_level(longitudinal, lateral, cls_name, 1.0 if score is None else score)
    return {
        "id": object_id,
        "source": source,
        "class": cls_name,
        "label": label,
        "score": None if score is None else float(score),
        "threat_level": threat_level,
        "box3d": _serialize_box(box),
    }


def build_scene_stream(sample_token, sample_timestamp, nusc, gt_items, pred_items):
    ego_speed_mps = estimate_ego_speed_mps(nusc, sample_token)
    ego_speed_kph = ego_speed_mps * 3.6

    objects = []
    for item in gt_items:
        object_id = item.get("id") or f"gt-{item['label']}"
        objects.append(
            _serialize_object(
                object_id=object_id,
                source="gt",
                cls_name=item["class_name"],
                box=item["box_ego"],
                score=None,
                label=item["label"],
            )
        )

    for index, item in enumerate(pred_items):
        object_id = item.get("id") or f"pred-{index}-{item['class_name']}"
        objects.append(
            _serialize_object(
                object_id=object_id,
                source="pred",
                cls_name=item["class_name"],
                box=item["box"],
                score=item.get("score"),
                label=item["label"],
            )
        )

    predicted = [obj for obj in objects if obj["source"] == "pred"]
    threat_counts = {"high": 0, "medium": 0, "low": 0}
    for obj in predicted:
        threat_counts[obj["threat_level"]] += 1

    return {
        "schema_version": "1.0.0",
        "frame_id": sample_token,
        "timestamp_ms": int(sample_timestamp / 1000.0),
        "ego": {
            "id": "ego",
            "speed_mps": round(ego_speed_mps, 3),
            "speed_kph": round(ego_speed_kph, 1),
            "heading_rad": 0.0,
            "box3d": {
                "center": {"x": 0.0, "y": 0.0, "z": 0.75},
                "size": {"length": 4.86, "width": 1.92, "height": 1.55},
                "yaw": 0.0,
            },
            "threat_level": "low",
        },
        "lanes": build_lane_vectors(),
        "objects": objects,
        "summary": {
            "gt_count": sum(1 for obj in objects if obj["source"] == "gt"),
            "pred_count": len(predicted),
            "threat_counts": threat_counts,
            "max_threat_level": "high" if threat_counts["high"] else "medium" if threat_counts["medium"] else "low",
        },
    }


def build_example_scene_stream():
    return {
        "schema_version": "1.0.0",
        "frame_id": "sample_demo_0001",
        "timestamp_ms": 1712736000123,
        "ego": {
            "id": "ego",
            "speed_mps": 11.8,
            "speed_kph": 42.5,
            "heading_rad": 0.0,
            "box3d": {
                "center": {"x": 0.0, "y": 0.0, "z": 0.75},
                "size": {"length": 4.86, "width": 1.92, "height": 1.55},
                "yaw": 0.0,
            },
            "threat_level": "low",
        },
        "lanes": build_lane_vectors(),
        "objects": [
            {
                "id": "gt-car-001",
                "source": "gt",
                "class": "car",
                "label": "GT car",
                "score": None,
                "threat_level": "medium",
                "box3d": {
                    "center": {"x": 16.2, "y": 0.4, "z": 0.9},
                    "size": {"length": 4.7, "width": 1.9, "height": 1.6},
                    "yaw": 0.04,
                },
            },
            {
                "id": "pred-car-001",
                "source": "pred",
                "class": "car",
                "label": "Pred car",
                "score": 0.92,
                "threat_level": "high",
                "box3d": {
                    "center": {"x": 14.7, "y": -0.2, "z": 0.9},
                    "size": {"length": 4.6, "width": 1.88, "height": 1.58},
                    "yaw": 0.02,
                },
            },
            {
                "id": "pred-ped-002",
                "source": "pred",
                "class": "pedestrian",
                "label": "Pred ped",
                "score": 0.81,
                "threat_level": "medium",
                "box3d": {
                    "center": {"x": 10.3, "y": 3.2, "z": 0.86},
                    "size": {"length": 0.65, "width": 0.65, "height": 1.72},
                    "yaw": 1.57,
                },
            },
        ],
        "summary": {
            "gt_count": 1,
            "pred_count": 2,
            "threat_counts": {"high": 1, "medium": 1, "low": 0},
            "max_threat_level": "high",
        },
    }
