import time

import numpy as np
from .imaug import transform, create_operators
from .db_postprocess import DBPostProcess
from .logger import get_logger
from .predict_base import PredictBase

log = get_logger("predict_det")


class TextDetector(PredictBase):
    def __init__(self, args):
        self.args = args
        self.det_algorithm = args.det_algorithm
        pre_process_list = [
            {
                "DetResizeForTest": {
                    "limit_side_len": args.det_limit_side_len,
                    "limit_type": args.det_limit_type,
                    "max_side_len": getattr(args, "det_max_side_len", None),
                    "max_pixels": getattr(args, "det_max_pixels", None),
                }
            },
            {
                "NormalizeImage": {
                    "std": [0.229, 0.224, 0.225],
                    "mean": [0.485, 0.456, 0.406],
                    "scale": "1./255.",
                    "order": "hwc",
                }
            },
            {"ToCHWImage": None},
            {"KeepKeys": {"keep_keys": ["image", "shape"]}},
        ]
        postprocess_params = {}
        postprocess_params["name"] = "DBPostProcess"
        postprocess_params["thresh"] = args.det_db_thresh
        postprocess_params["box_thresh"] = args.det_db_box_thresh
        postprocess_params["max_candidates"] = getattr(args, "det_db_max_candidates", 1000)
        postprocess_params["unclip_ratio"] = args.det_db_unclip_ratio
        postprocess_params["use_dilation"] = args.use_dilation
        postprocess_params["score_mode"] = args.det_db_score_mode
        postprocess_params["box_type"] = args.det_box_type

        # Instantiate preprocess ops
        self.preprocess_op = create_operators(pre_process_list)
        # self.postprocess_op = build_post_process(postprocess_params)
        # Instantiate postprocess ops
        self.postprocess_op = DBPostProcess(**postprocess_params)

        # Initialize model
        self.det_onnx_session = self.get_onnx_session(args.det_model_dir, args.use_gpu, gpu_id = args.gpu_id)
        self.det_input_name = self.get_input_name(self.det_onnx_session)
        self.det_output_name = self.get_output_name(self.det_onnx_session)
        self.det_tile_enable = bool(getattr(args, "det_tile_enable", False))
        self.det_tile_height = max(256, int(getattr(args, "det_tile_height", 1600)))
        self.det_tile_overlap = max(
            32,
            min(int(getattr(args, "det_tile_overlap", 160)), self.det_tile_height // 2),
        )
        self.det_tile_aspect_ratio = float(getattr(args, "det_tile_aspect_ratio", 2.5))
        log.info("Detection model loaded: {}", args.det_model_dir)

        # Warm-up: avoid 2-10x slower first call due to lazy kernel initialization
        warmup_side = max(32, int(args.det_limit_side_len))
        dummy = np.zeros((1, 3, warmup_side, warmup_side), dtype=np.float32)
        self.det_onnx_session.run(self.det_output_name, {self.det_input_name[0]: dummy})

    def order_points_clockwise(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        tmp = np.delete(pts, (np.argmin(s), np.argmax(s)), axis=0)
        diff = np.diff(np.array(tmp), axis=1)
        rect[1] = tmp[np.argmin(diff)]
        rect[3] = tmp[np.argmax(diff)]
        return rect

    def clip_det_res(self, points, img_height, img_width):
        for pno in range(points.shape[0]):
            points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
            points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
        return points

    def filter_tag_det_res(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            if type(box) is list:
                box = np.array(box)
            box = self.order_points_clockwise(box)
            box = self.clip_det_res(box, img_height, img_width)
            rect_width = int(np.linalg.norm(box[0] - box[1]))
            rect_height = int(np.linalg.norm(box[0] - box[3]))
            if rect_width <= 3 or rect_height <= 3:
                continue
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    def filter_tag_det_res_only_clip(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            if type(box) is list:
                box = np.array(box)
            box = self.clip_det_res(box, img_height, img_width)
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    def _should_tile(self, img) -> bool:
        if not self.det_tile_enable or img is None or img.ndim < 2:
            return False
        img_h, img_w = img.shape[:2]
        if img_h <= self.det_tile_height or img_w <= 0:
            return False
        return (img_h / float(img_w)) > self.det_tile_aspect_ratio

    def _compute_tile_ranges(self, img_h: int) -> list[tuple[int, int]]:
        stride = max(1, self.det_tile_height - self.det_tile_overlap)
        ranges: list[tuple[int, int]] = []
        y0 = 0
        while y0 < img_h:
            y1 = min(y0 + self.det_tile_height, img_h)
            ranges.append((y0, y1))
            if y1 >= img_h:
                break
            y0 += stride
        min_tail_h = min(self.det_tile_height, self.det_tile_overlap * 2 + 128)
        if len(ranges) > 1 and (ranges[-1][1] - ranges[-1][0]) < min_tail_h:
            ranges[-1] = (max(0, img_h - min_tail_h), img_h)
        return ranges

    @staticmethod
    def _prefer_tile_candidate(cand: dict, prev: dict) -> bool:
        if cand["seam_margin"] > prev["seam_margin"] + 5.0:
            return True
        if prev["seam_margin"] > cand["seam_margin"] + 5.0:
            return False
        return cand["area"] > prev["area"]

    @classmethod
    def _is_duplicate_tile_box(cls, a: dict, b: dict) -> bool:
        if a["tile_idx"] == b["tile_idx"]:
            return False
        ax1, ay1, ax2, ay2 = a["bbox"]
        bx1, by1, bx2, by2 = b["bbox"]
        inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
        inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
        if inter_w <= 0.0 or inter_h <= 0.0:
            return False
        inter_area = inter_w * inter_h
        min_area = min(a["area"], b["area"])
        if min_area <= 0.0:
            return False
        union = a["area"] + b["area"] - inter_area
        iou = inter_area / union if union > 0.0 else 0.0
        if iou >= 0.45:
            return True
        io_min = inter_area / min_area
        min_h = min(max(ay2 - ay1, 1e-6), max(by2 - by1, 1e-6))
        v_overlap = inter_h / min_h
        return io_min >= 0.70 and v_overlap >= 0.60

    @classmethod
    def _merge_tiled_boxes(cls, candidates: list[dict]) -> np.ndarray:
        kept: list[dict] = []
        for cand in candidates:
            matched_indices = [
                idx for idx, prev in enumerate(kept) if cls._is_duplicate_tile_box(cand, prev)
            ]
            if not matched_indices:
                kept.append(cand)
                continue
            if all(cls._prefer_tile_candidate(cand, kept[idx]) for idx in matched_indices):
                kept[matched_indices[0]] = cand
                for idx in reversed(matched_indices[1:]):
                    kept.pop(idx)
        if not kept:
            return np.array([])
        return np.array([item["box"] for item in kept])

    def _detect_tiled(self, img):
        img_h = img.shape[0]
        ranges = self._compute_tile_ranges(img_h)
        num_tiles = len(ranges)
        edge_margin = 6.0
        max_seam_box_h = max(32.0, float(self.det_tile_overlap) - 2.0 * edge_margin)
        candidates: list[dict] = []

        for tile_idx, (y0, y1) in enumerate(ranges):
            tile_img = img[y0:y1, :, :]
            tile_h = float(y1 - y0)
            tile_boxes = self._detect_single(tile_img)
            if tile_boxes is None or len(tile_boxes) == 0:
                continue
            for box in tile_boxes:
                box_arr = np.array(box, dtype=np.float32)
                local_y_min = float(np.min(box_arr[:, 1]))
                local_y_max = float(np.max(box_arr[:, 1]))
                box_h = local_y_max - local_y_min
                if tile_idx > 0 and local_y_min <= edge_margin and box_h <= max_seam_box_h:
                    continue
                if (
                    tile_idx < num_tiles - 1
                    and local_y_max >= tile_h - edge_margin
                    and box_h <= max_seam_box_h
                ):
                    continue
                top_margin = float("inf") if tile_idx == 0 else local_y_min
                bottom_margin = float("inf") if tile_idx == num_tiles - 1 else (tile_h - local_y_max)
                seam_margin = min(top_margin, bottom_margin)

                global_box = box_arr.copy()
                global_box[:, 1] += float(y0)
                x1 = float(np.min(global_box[:, 0]))
                y_min = float(np.min(global_box[:, 1]))
                x2 = float(np.max(global_box[:, 0]))
                y_max = float(np.max(global_box[:, 1]))
                area = max(0.0, x2 - x1) * max(0.0, y_max - y_min)
                candidates.append(
                    {
                        "box": global_box,
                        "tile_idx": tile_idx,
                        "seam_margin": seam_margin,
                        "bbox": (x1, y_min, x2, y_max),
                        "area": area,
                    }
                )

        return self._merge_tiled_boxes(candidates)

    def _detect_single(self, img):
        ori_shape = img.shape
        data = {"image": img}

        data = transform(data, self.preprocess_op)
        img, shape_list = data
        if img is None:
            return None, 0
        img = np.expand_dims(img, axis=0)
        shape_list = np.expand_dims(shape_list, axis=0)

        input_feed = self.get_input_feed(self.det_input_name, img)
        t0 = time.time()
        outputs = self.det_onnx_session.run(self.det_output_name, input_feed=input_feed)
        log.debug("Detection inference time: {:.3f}s", time.time() - t0)

        preds = {}
        preds["maps"] = outputs[0]

        post_result = self.postprocess_op(preds, shape_list)
        dt_boxes = post_result[0]["points"]

        if self.args.det_box_type == "poly":
            dt_boxes = self.filter_tag_det_res_only_clip(dt_boxes, ori_shape)
        else:
            dt_boxes = self.filter_tag_det_res(dt_boxes, ori_shape)

        return dt_boxes

    def __call__(self, img):
        if self._should_tile(img):
            return self._detect_tiled(img)
        return self._detect_single(img)
