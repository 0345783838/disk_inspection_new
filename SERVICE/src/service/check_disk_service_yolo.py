import math
import time
import cv2
import numpy as np
from src.dtos.meta import DataResponse, ErrorCode, DataDebugResponse, DataResponseUv, DataDebugUVResponse, \
    InspectionState, ClassifyResult, Params
from src.service.base_service import BaseService
import base64
import ast


class DiskCheckingService(BaseService):
    def __init__(self):
        super().__init__()
        pass

    # region Utility
    @staticmethod
    def _convert_2_base64(image):
        success, encoded_image = cv2.imencode('.png', image)
        if not success:
            return None
        image_bytes = encoded_image.tobytes()
        img_base64 = base64.b64encode(image_bytes).decode("utf-8")

        return img_base64

    @staticmethod
    def _get_box_centers(boxes):
        centers = []
        for box in boxes:
            x_center = (box[0] + box[2]) / 2
            y_center = (box[1] + box[3]) / 2
            centers.append([x_center, y_center])
        return np.array(centers)

    @staticmethod
    def euclidean_distance(pointA, pointB):
        return np.linalg.norm(pointA - pointB)

    @staticmethod
    def get_box_size(box):
        x1, y1, x2, y2 = box
        distance = x2 - x1 if x2 - x1 > y2 - y1 else y2 - y1
        return distance

    def draw_detected_boxes(self, image, boxes, confs, cls_idxs):
        for box, conf, cls in zip(boxes, confs, cls_idxs):
            xmin = int(box[0])
            ymin = int(box[1])
            xmax = int(box[2])
            ymax = int(box[3])
            object_conf = float(conf)

            # Draw the bounding box
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

            # Prepare the label with class name and confidence
            label = f"{object_conf:.2f}"

            # Calculate the position for the label
            label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            top_left = (
                xmin, ymin - label_size[1] - 10 if ymin - label_size[1] - 10 > 10 else ymin + label_size[1] + 10)

            # Draw the label background
            cv2.rectangle(image, (top_left[0] - 1, top_left[1] + base_line + 10),
                          (top_left[0] + label_size[0], top_left[1] - label_size[1] + 10), (0, 255, 0), cv2.FILLED)

            # Put the label text
            cv2.putText(image, label, (top_left[0], top_left[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    @staticmethod
    def get_center_row(boxes, image_height):
        # Tính tâm Y của tất cả các boxes
        cy = (boxes[:, 1] + boxes[:, 3]) / 2

        # 1. Tìm box gần tâm ảnh nhất làm hệ quy chiếu (anchor)
        center_y_image = image_height / 2
        anchor_idx = np.argmin(np.abs(cy - center_y_image))
        anchor_cy = cy[anchor_idx]

        # 2. Tính chiều cao trung bình của các box để làm ngưỡng (threshold)
        # (Các box cùng một hàng sẽ có tâm Y lệch nhau không quá nửa chiều cao)
        avg_h = np.mean(boxes[:, 3] - boxes[:, 1])
        threshold = avg_h * 0.5

        # 3. Lọc ra các box thuộc hàng giữa (nằm trong ngưỡng threshold so với anchor_cy)
        middle_row_boxes = boxes[np.abs(cy - anchor_cy) < threshold]

        # (Tuỳ chọn) Sort lại các box trong hàng giữa theo trục X từ trái qua phải
        # cx = (middle_row_boxes[:, 0] + middle_row_boxes[:, 2]) / 2
        # middle_row_boxes = middle_row_boxes[np.argsort(cx)]

        return middle_row_boxes

    @staticmethod
    def visualize_edge_spacing(
            image: np.ndarray,
            caliper_result: dict,
            min_dist: float,
            max_dist: float,
            axis: int = 0,
            line_thickness: int = 2,
            font_scale: float = 0.45,
            midpoint_radius: int = 7
    ):

        # 1. compute midpoints
        mids = []
        distance_list = []
        for pair in caliper_result["pairs"]:
            p1 = np.array(pair["e1"]["point"], dtype=np.float32)
            p2 = np.array(pair["e2"]["point"], dtype=np.float32)
            mids.append((p1 + p2) / 2.0)

        # 2. sort midpoints
        mids = sorted(mids, key=lambda m: m[axis])

        results = []

        # 3. draw midpoints (ORANGE)
        for m in mids:
            cv2.circle(
                image,
                tuple(m.astype(int)),
                midpoint_radius,
                (0, 127, 255),  # orange
                -1
            )

        # 4. draw spacing + text AT TRUE CENTER
        for i in range(len(mids) - 1):
            m1 = mids[i]
            m2 = mids[i + 1]

            dist = float(np.linalg.norm(m2 - m1))
            distance_list.append(dist)
            is_ng = dist < min_dist or dist > max_dist
            color = (0, 0, 255) if is_ng else (0, 255, 0)

            p1 = tuple(m1.astype(int))
            p2 = tuple(m2.astype(int))

            # line between edges
            cv2.line(image, p1, p2, color, line_thickness)

            # TRUE center between two edges
            mid_text = ((m1 + m2) / 2).astype(int)
            label = f"{dist:.1f}"

            # center text exactly
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
            )

            cv2.putText(
                image,
                label,
                (mid_text[0] - tw // 2, mid_text[1] - th // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                1,
                cv2.LINE_AA
            )

            results.append(False if is_ng else True)

        # 5. re - draw midpoints (ORANGE)
        for m in mids:
            cv2.circle(
                image,
                tuple(m.astype(int)),
                1,
                (0, 127, 255),  # orange
                -1
            )

        return results, distance_list, [x.tolist() for x in mids]

    @staticmethod
    def visualize_edge_spacing_uv(
            image: np.ndarray,
            center: tuple,
            caliper_result: dict,
            axis: int = 0,
            line_thickness: int = 2,
            midpoint_radius: int = 7,

    ):

        # 1. compute midpoints
        mids = []
        for pair in caliper_result["pairs"]:
            p1 = np.array(pair["e1"]["point"], dtype=np.float32)
            p2 = np.array(pair["e2"]["point"], dtype=np.float32)
            mids.append((p1 + p2) / 2.0)

        # 2. sort midpoints
        mids = sorted(mids, key=lambda m: m[axis])
        # 3. Draw center line
        cv2.line(image, (0, center[1]), (image.shape[1], center[1]), (0, 255, 0), line_thickness)
        # 4. draw midpoints (ORANGE)
        for m in mids:
            cv2.circle(
                image,
                tuple(m.astype(int)),
                midpoint_radius,
                (247, 192, 27),  # orange
                -1
            )

        return image, mids

    @staticmethod
    def merge_boxes_1d_x(box_score_list, x_gap=0):
        """
        Merge boxes assuming they lie on the same horizontal line (Y nearly equal)

        x_gap: cho phép khoảng hở nhỏ giữa các box vẫn được merge
        """
        if len(box_score_list) == 0:
            return []

        boxes = [b for b, _ in box_score_list]

        # sort theo x1
        boxes = sorted(boxes, key=lambda b: b[0])

        merged = []
        cur = boxes[0]

        for box in boxes[1:]:
            # nếu overlap hoặc chạm theo trục X
            if box[0] <= cur[2] + x_gap:
                cur = [
                    min(cur[0], box[0]),
                    min(cur[1], box[1]),
                    max(cur[2], box[2]),
                    max(cur[3], box[3]),
                ]
            else:
                merged.append(cur)
                cur = box

        merged.append(cur)
        return merged

    @staticmethod
    def draw_stripes_on_contour_inplace(
            image, contour,
            stripe_spacing=10,
            color=(0, 255, 0),
            thickness=1
    ):
        if len(contour) == 0:
            return

        cnt = contour[0]

        x, y, w, h = cv2.boundingRect(cnt)

        if w <= 0 or h <= 0:
            return

        roi = image[y:y + h, x:x + w]

        # shift contour về local ROI
        cnt_local = cnt.copy()
        cnt_local[:, 0, 0] -= x
        cnt_local[:, 0, 1] -= y

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [cnt_local], -1, 255, -1)

        stripe_layer = np.zeros_like(roi)

        for i in range(-h, w, stripe_spacing):
            cv2.line(
                stripe_layer,
                (i, h),
                (i + h, 0),
                color,
                thickness
            )

        m = mask > 0
        roi[m] = (
                roi[m].astype(np.float32) * 0.9 +
                stripe_layer[m].astype(np.float32) * 0.9
        ).clip(0, 255).astype(np.uint8)

    @staticmethod
    def draw_boxes(image, boxes, color, class_name):
        for box in boxes:
            if type(box) == tuple:
                bb = box[0]
                score = box[1]
                cv2.rectangle(image, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), color, 1)
                cv2.putText(image, f"{score:.2f}", (int(bb[0]), int(bb[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            else:
                box_height = box[3] - box[1]
                new_y1 = int(box[1] + box_height * 0.3)
                new_y2 = int(box[3] - box_height * 0.3)
                offset_x = 4
                offset_y = 2
                (tw, th), _ = cv2.getTextSize(f"{class_name.upper()}", cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(image, (int(box[0]), new_y1), (int(box[2]), new_y2), color, 2)

                cv2.rectangle(image, (int(box[0]), new_y2 - th - offset_y), (int(box[0]) + tw + offset_x, new_y2),
                              color, cv2.FILLED)
                cv2.putText(image, f"{class_name.upper()}", (int(box[0]) + offset_x // 2, new_y2 - offset_y // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    @staticmethod
    def order_quad_pts(pts):
        """
        Input: pts shape (4,2) unsorted
        Output: ordered as [tl, tr, br, bl]
        """
        pts = np.array(pts, dtype=np.float32)
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1).reshape(-1)
        tl = pts[np.argmin(s)]
        br = pts[np.argmax(s)]
        tr = pts[np.argmin(diff)]
        bl = pts[np.argmax(diff)]
        return np.array([tl, tr, br, bl], dtype=np.float32)

    # @staticmethod
    # def expand_quad_towards_center(quad, ratio=0.10):
    #     """
    #     Expand quad points away from center by ratio.
    #     quad: (4,2)
    #     """
    #     center = np.mean(quad, axis=0)
    #     return center + (quad - center) * (1.0 + ratio)

    @staticmethod
    def expand_quad_towards_center(quad, ratio_x=0.10, ratio_y=0.10):
        """
        Expand quad points away from center by ratio.
        quad: (4,2)
        """
        quad = np.asarray(quad, dtype=np.float32)
        center = np.mean(quad, axis=0)  # (cx, cy)

        # scale matrix for x,y
        scale = np.array([1.0 + ratio_x, 1.0 + ratio_y], dtype=np.float32)

        # (quad - center) * scale + center
        return center + (quad - center) * scale

    @staticmethod
    def quad_to_rect_size(quad):
        """
        Compute destination rectangle width and height from ordered quad (tl,tr,br,bl)
        Use max of top/bottom edge lengths for width, left/right edges for height.
        """
        tl, tr, br, bl = quad
        width_top = np.linalg.norm(tr - tl)
        width_bot = np.linalg.norm(br - bl)
        max_w = int(round(max(width_top, width_bot)))
        height_left = np.linalg.norm(bl - tl)
        height_right = np.linalg.norm(br - tr)
        max_h = int(round(max(height_left, height_right)))
        # Avoid zero
        max_w = max(1, max_w)
        max_h = max(1, max_h)
        return max_w, max_h

    def warp_quad_to_rect(self, image, quad, expand_ratio_x=0.10, expand_ratio_y=0.10,
                          border_mode=cv2.BORDER_REPLICATE):
        """
        quad: (4,2) unsorted
        returns: warped_img, M (3x3), dst_bbox = (w,h)
        """
        # order
        quad_ord = self.order_quad_pts(quad)
        # expand
        quad_exp = self.expand_quad_towards_center(quad_ord, ratio_x=expand_ratio_x, ratio_y=expand_ratio_y)
        # destination size
        w, h = self.quad_to_rect_size(quad_exp)
        dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
        # compute homography
        M = cv2.getPerspectiveTransform(quad_exp.astype(np.float32), dst)
        warped = cv2.warpPerspective(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=border_mode)
        return warped, M, (w, h), quad_exp

    @staticmethod
    def transform_points(M, pts):
        """
        Apply homography M to pts (N,2) and return transformed (N,2)
        """
        pts_h = np.hstack([pts, np.ones((len(pts), 1), dtype=np.float32)])  # (N,3)
        dst_h = (M @ pts_h.T).T  # (N,3)
        dst = dst_h[:, :2] / dst_h[:, 2:3]
        return dst

    def update_boxes_after_warp(self, boxes, M):
        """
        boxes: (N,4) as [x1,y1,x2,y2] axis-aligned in original image
        M: homography from original -> warped
        returns: new_boxes (N,4) in warped image coords (axis-aligned)
        """
        new_boxes = []
        for b in boxes:
            x1, y1, x2, y2 = b
            # four corners
            corners = np.array([
                [x1, y1],
                [x2, y1],
                [x2, y2],
                [x1, y2]
            ], dtype=np.float32)
            t = self.transform_points(M, corners)  # (4,2)
            # axis aligned bbox in warped image
            x_min = float(np.min(t[:, 0]))
            y_min = float(np.min(t[:, 1]))
            x_max = float(np.max(t[:, 0]))
            y_max = float(np.max(t[:, 1]))
            new_boxes.append([x_min, y_min, x_max, y_max])
        return np.array(new_boxes, dtype=np.float32)

    @staticmethod
    def split_rows(boxes, expected_rows=3):
        """Split detected point boxes into top-to-bottom rows."""
        boxes = np.asarray(boxes, dtype=np.float32)
        if boxes.ndim != 2 or boxes.shape[1] != 4 or len(boxes) < expected_rows:
            raise ValueError("Not enough valid point boxes to split rows")
        if expected_rows != 3:
            raise ValueError("Only the three-point-row layout is supported")

        centers_y = (boxes[:, 1] + boxes[:, 3]) / 2.0
        order = np.argsort(centers_y)
        boxes_sorted = boxes[order]
        centers_y = centers_y[order]

        gaps = np.diff(centers_y)
        if len(gaps) < expected_rows - 1:
            raise ValueError("Cannot find three point rows")
        split_indexes = np.sort(np.argsort(gaps)[-(expected_rows - 1):])
        rows = np.split(boxes_sorted, split_indexes + 1)
        if len(rows) != expected_rows or any(len(row) == 0 for row in rows):
            raise ValueError("Cannot find three point rows")

        # Keep every row deterministic and make representative box selection stable.
        return tuple(row[np.argsort((row[:, 0] + row[:, 2]) / 2.0)] for row in rows)

    def full_rectify_pipeline(self, image, row1, row3, expand_ratio_x=0.10, expand_ratio_y=0.10):
        """Rectify the tray from the outer point rows and return source crop geometry."""
        left_top_box = row1[np.argmin(row1[:, 0])]
        right_top_box = row1[np.argmax(row1[:, 2])]
        left_bottom_box = row3[np.argmin(row3[:, 0])]
        right_bottom_box = row3[np.argmax(row3[:, 2])]

        quad = np.array([
            [left_top_box[0], left_top_box[1]],
            [right_top_box[2], right_top_box[1]],
            [right_bottom_box[2], right_bottom_box[3]],
            [left_bottom_box[0], left_bottom_box[3]],
        ], dtype=np.float32)
        quad = self.order_quad_pts(quad)
        quad = self.expand_quad_towards_center(quad, expand_ratio_x, expand_ratio_y)

        # The returned quad is reused on the UV frame, whose validation requires
        # source-image coordinates to remain inside the image.
        quad[:, 0] = np.clip(quad[:, 0], 0, image.shape[1] - 1)
        quad[:, 1] = np.clip(quad[:, 1], 0, image.shape[0] - 1)

        width, height = self.quad_to_rect_size(quad)
        destination = np.array([
            [0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]
        ], dtype=np.float32)
        matrix = cv2.getPerspectiveTransform(quad, destination)
        warped = cv2.warpPerspective(image, matrix, (width, height), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REPLICATE)
        return warped, matrix, (width, height), quad

    # ---------- Example usage ----------
    # image: original image
    # row1, row3: arrays of boxes for top and bottom rows (format [x1,y1,x2,y2])
    # boxes: original full list of boxes (N,4)

    def crop_by_boxes(self, image, middle_boxes, expand_ratio_x=0.10):
        # select P1-P4 as user described
        left_box = middle_boxes[np.argmin(middle_boxes[:, 0])]
        right_box = middle_boxes[np.argmax(middle_boxes[:, 2])]

        expanded_left_x = left_box[0] - (right_box[2] - left_box[0]) * expand_ratio_x if left_box[0] - (right_box[2] - left_box[0]) * expand_ratio_x > 0 else 0
        expanded_right_x = right_box[2] + (right_box[2] - left_box[0]) * expand_ratio_x if right_box[2] + (right_box[2] - left_box[0]) * expand_ratio_x < image.shape[1] else image.shape[1]

        rect = [expanded_left_x, 0, expanded_right_x, image.shape[0]]  # x1, y1, x2, y2
        crop = image[int(rect[1]):int(rect[3]), int(rect[0]):int(rect[2])]

        return crop, [[rect[0], 0], [rect[2], 0], [rect[2], image.shape[0]], [rect[0], image.shape[0]]]

    @staticmethod
    def get_ratio_shift_box(
            image,
            box,
            direction="bottom",
            ratio_width=1.0,
            ratio_height=0.5,
            ratio_shift_y=0.0,

    ):
        """
        box = [x1, y1, x2, y2]
        direction: 'top' hoặc 'bottom'
        ratio_height: phần trăm chiều cao muốn crop (0→1)
        ratio_shift_y: phần trăm dịch tâm theo chiều cao box (0→1)
        ratio_width: mở rộng crop theo chiều rộng (1.0 = giữ nguyên, 2.0 = gấp đôi)
        """

        x1, y1, x2, y2 = map(int, box)

        h = y2 - y1
        w = x2 - x1

        # --- Height crop ---
        crop_h = int(h * ratio_height)

        # center Y
        cy = (y1 + y2) / 2.0
        shift_y = h * ratio_shift_y

        if direction == "bottom":
            cy_shifted = cy + shift_y
        elif direction == "top":
            cy_shifted = cy - shift_y
        else:
            raise ValueError("direction must be 'top' or 'bottom'")

        y1_new = int(cy_shifted - crop_h / 2)
        y2_new = int(y1_new + crop_h)

        # --- Width crop ---
        new_w = w * ratio_width
        cx = (x1 + x2) / 2.0

        x1_new = int(cx - new_w / 2)
        x2_new = int(cx + new_w / 2)

        # Không clamp: numpy tự cắt nếu vượt ảnh
        return [max(x1_new, 0), max(y1_new, 0), max(x2_new, 0), max(y2_new, 0)]

    def get_line_boxes_ratio_shift(self, image, line_boxes, direction, ratio_width=1.5, ratio_height=1.5,
                                   ratio_shift_y=0.5):
        boxes = []
        for box in line_boxes:
            new_box = self.get_ratio_shift_box(image, box, direction, ratio_width, ratio_height, ratio_shift_y)
            boxes.append(new_box)
        return boxes

    @staticmethod
    def crop_boxes(image, boxes, direction):
        crops = []
        if direction == 'top':
            for box in boxes:
                crop = image[box[1]:box[3], box[0]:box[2]]
                crop = cv2.rotate(crop, cv2.ROTATE_180)
                crops.append(crop)
        else:
            for box in boxes:
                crop = image[box[1]:box[3], box[0]:box[2]]
                crops.append(crop)
        return crops

    @staticmethod
    def crop_box_for_segmentation(image, box_middle, start_ratio=0.0, height_ratio=2.8, direction='top'):
        y_1 = int(box_middle[1])
        y_2 = int(box_middle[3])
        start_height = int((y_2 - y_1) * start_ratio)
        ratio_height = int((y_2 - y_1) * height_ratio)

        if direction == 'top':
            start_y = y_1 - start_height
            end_y = start_y - ratio_height
            if end_y < 0:
                end_y = 0
            box = [0, end_y, image.shape[1], start_y]
        else:
            start_y = y_2 + start_height
            end_y = start_y + ratio_height
            if end_y > image.shape[0]:
                end_y = image.shape[0]
            box = [0, start_y, image.shape[1], end_y]

        crop = image[box[1]:box[3], box[0]:box[2]]
        return crop, box

    @staticmethod
    def crop_between_rows(image, upper_row, lower_row, ratio=0.35, direction='top'):
        """Crop a horizontal band between two rectified point rows.

        ``bottom`` anchors the band at the top of the lower row; ``top`` anchors
        it at the bottom of the upper row. This preserves the old three-row
        measurement geometry while using the whole row instead of an arbitrary box.
        """
        upper_bottom = int(round(float(np.median(upper_row[:, 3]))))
        lower_top = int(round(float(np.median(lower_row[:, 1]))))
        gap = lower_top - upper_bottom
        if gap <= 0:
            raise ValueError("Point rows overlap after rectification")

        crop_height = max(1, int(round(gap * ratio)))
        if direction == 'bottom':
            y2 = lower_top
            y1 = y2 - crop_height
        elif direction == 'top':
            y1 = upper_bottom
            y2 = y1 + crop_height
        else:
            raise ValueError("direction must be 'top' or 'bottom'")

        y1 = max(0, y1)
        y2 = min(image.shape[0], y2)
        if y2 <= y1:
            raise ValueError("Invalid segmentation region")
        box = [0, y1, image.shape[1], y2]
        return image[y1:y2, :], box

    def get_caliper_result(self, img, center):
        res = self.caliper.measure(img, center=center)
        # vis = self.caliper.visualize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), center=center)
        return res

    def clean_mask(self, img, min_disk_area):
        img = self.remove_mask_noise(img, min_disk_area)
        # img_close = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((7, 3)))

        return img

    def get_caliper_result_debug(self, img, center, length_rate, min_edge_distance, max_edge_distance, thickness_list):
        # img_open = cv2.morphologyEx(img_close, cv2.MORPH_OPEN, np.ones((1, 3)))

        res = self.caliper.measure_debug(img,
                                         center=center,
                                         min_edge_distance=min_edge_distance,
                                         max_edge_distance=max_edge_distance,
                                         length_rate=length_rate,
                                         thickness_list=thickness_list)

        # vis = self.caliper.visualize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), center=center)
        return res

    def draw_mask_contour(self, img, mask_seg, center, draw_ratio=0.4, mask_color=(255, 153, 51),
                          cnt_color=(255, 102, 0)):
        x, y = center
        img_h, img_w, _ = img.shape

        y1 = max(0, int(y - img_h * draw_ratio / 2.0))
        y2 = min(img_h, int(y + img_h * draw_ratio / 2.0))

        mask_seg_center = mask_seg.copy()
        mask_seg_center[0:y1, :] = 0
        mask_seg_center[y2:, :] = 0

        contours, _ = cv2.findContours(mask_seg_center, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            cv2.drawContours(img, [contour], -1, cnt_color, 1)
            self.draw_stripes_on_contour_inplace(img, [contour], color=mask_color)

    def remove_mask_noise(self, img, min_disk_area):
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_disk_area:
                cv2.drawContours(img, [contour], -1, 0, -1)

        return img

    def remove_mask_noise_uv(self, img, min_disk_area):
        kernel = np.ones((3, 3), np.uint8)
        closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_disk_area:
                cv2.drawContours(closed, [contour], -1, 0, -1)

        return closed

    @staticmethod
    def crop_by_4pts(image, pts):
        """
        image: np.ndarray (H, W, C)
        pts: array-like shape (4, 2), 4 điểm bất kỳ trên ảnh gốc

        return:
            cropped_img: ảnh đã crop + align
            M: perspective transform matrix
        """
        pts = np.array(pts, dtype=np.float32)

        # --- 1. Sắp xếp 4 điểm theo thứ tự: tl, tr, br, bl ---
        def order_points(pts):
            rect = np.zeros((4, 2), dtype=np.float32)

            s = pts.sum(axis=1)
            diff = np.diff(pts, axis=1)

            rect[0] = pts[np.argmin(s)]  # top-left
            rect[2] = pts[np.argmax(s)]  # bottom-right
            rect[1] = pts[np.argmin(diff)]  # top-right
            rect[3] = pts[np.argmax(diff)]  # bottom-left

            return rect

        rect = order_points(pts)

        # --- 2. Tính width / height output ---
        w1 = np.linalg.norm(rect[1] - rect[0])
        w2 = np.linalg.norm(rect[2] - rect[3])
        width = int(max(w1, w2))

        h1 = np.linalg.norm(rect[3] - rect[0])
        h2 = np.linalg.norm(rect[2] - rect[1])
        height = int(max(h1, h2))

        # --- 3. Điểm đích ---
        dst = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype=np.float32)

        # --- 4. Perspective transform ---
        M = cv2.getPerspectiveTransform(rect, dst)
        cropped = cv2.warpPerspective(
            image, M, (width, height),
            flags=cv2.INTER_LINEAR
        )

        return cropped

    # endregion

    def _default_white_params(self):
        return Params(segment_threshold=self.disk_segmentor_yolo.conf_threshold,
                      segment_iou=self.disk_segmentor_yolo.iou_threshold,
                      detect_threshold=self.disk_point_detect_model.conf_threshold,
                      detect_iou=self.disk_point_detect_model.iou_threshold,
                      caliper_min_edge_distance=self.caliper.min_edge_distance,
                      caliper_max_edge_distance=self.caliper.pair_max_gap,
                      caliper_length_rate=self.caliper.length_rate,
                      caliper_thickness_list=self.caliper.thickness_list,
                      disk_num=self.num_disk, disk_max_distance=self.max_disk_distance,
                      disk_min_distance=self.min_disk_distance, disk_min_area=self.min_disk_area)

    def check_disk_white(self, image):
        return self._check_white(image, self._default_white_params())

    def check_disk_debug(self, image, params):
        return self._check_white(image, params, debug=True)

    def _white_failure(self, image, debug):
        encoded = self._convert_2_base64(image)
        error = dict(Result=InspectionState.NG, ErrorCode=ErrorCode.ERR_NUM_DISK[0],
                     ErrorDesc=ErrorCode.ERR_NUM_DISK[1])
        if debug:
            return DataDebugResponse(**error, DetectImg=encoded, FinalImg=encoded,
                                     SegmentImg=self._convert_2_base64(np.zeros(image.shape[:2], dtype=np.uint8)))
        return DataResponse(**error, ResImg=encoded)

    def _check_white(self, image, params, debug=False):
        boxes, confs, cls_idxs = self.disk_point_detect_model.detect_objects_debug(
            image, params.detect_threshold, params.detect_iou)
        expected_rows = 3
        if len(boxes) < params.disk_num * expected_rows:
            return self._white_failure(image, debug)
        detect_image = None
        if debug:
            detect_image = image.copy()
            self.draw_detected_boxes(detect_image, boxes, confs, cls_idxs)

        # Group point detections into the three physical rows.
        try:
            boxes_l1, boxes_l2, boxes_l3 = self.split_rows(boxes, expected_rows=expected_rows)
        except ValueError:
            return self._white_failure(image, debug)
        if any(len(row) < params.disk_num for row in (boxes_l1, boxes_l2, boxes_l3)):
            return self._white_failure(image, debug)

        # Rectify the complete tray from the two outer point rows.
        crop_img, matrix, (crop_width, _), crop_quad = self.full_rectify_pipeline(
            image, boxes_l1, boxes_l3, expand_ratio_x=0.2, expand_ratio_y=0.1)

        boxes_l1 = self.update_boxes_after_warp(boxes_l1, matrix)
        boxes_l2 = self.update_boxes_after_warp(boxes_l2, matrix)
        boxes_l3 = self.update_boxes_after_warp(boxes_l3, matrix)

        # The UV contract remains two regions: below row 1 and above row 3.
        row1_reference = np.median(boxes_l1, axis=0)
        row3_reference = np.median(boxes_l3, axis=0)
        uv_box_l1 = self.get_uv_box(row1_reference, crop_width, start_ratio=0.0,
                                    ratio_height=2.0, direction="bottom")
        uv_box_l3 = self.get_uv_box(row3_reference, crop_width, start_ratio=0.0,
                                    ratio_height=2.0, direction="top")

        # Classify all four faces around the three point rows. Merge each face
        # independently so boxes at the same X on different rows are not combined.
        classify_regions = (
            (boxes_l1, "bottom"),
            (boxes_l2, "top"),
            (boxes_l2, "bottom"),
            (boxes_l3, "top"),
        )
        ng_groups = []
        no_disk_groups = []
        for row_boxes, direction in classify_regions:
            rects = self.get_line_boxes_ratio_shift(crop_img, row_boxes, direction)
            crops = self.crop_boxes(crop_img, rects, direction)
            if not crops or any(crop.size == 0 for crop in crops):
                return self._white_failure(image, debug)

            labels, class_confidences = self.point_classification_model.predict_batch(crops)
            if labels is None or class_confidences is None:
                raise RuntimeError("Classification inference failed")

            ng_groups.append(self.merge_boxes_1d_x([
                (box, confidence)
                for label, box, confidence in zip(labels, rects, class_confidences)
                if label == ClassifyResult.NG
            ]))
            no_disk_groups.append(self.merge_boxes_1d_x([
                (box, confidence)
                for label, box, confidence in zip(labels, rects, class_confidences)
                if label == ClassifyResult.NO_DISK
            ]))

        ng_boxes = [box for group in ng_groups for box in group]
        no_disk_boxes = [box for group in no_disk_groups for box in group]

        # Preserve the old three-row segmentation geometry: one band immediately
        # above row 2 and one immediately below it.
        try:
            crop_seg_1, box_seg_1 = self.crop_between_rows(
                crop_img, boxes_l1, boxes_l2, ratio=0.35, direction='bottom')
            crop_seg_2, box_seg_2 = self.crop_between_rows(
                crop_img, boxes_l2, boxes_l3, ratio=0.35, direction='top')
        except ValueError:
            return self._white_failure(image, debug)

        mask_seg_1, _ = self.disk_segmentor_yolo.segment_large_image_debug(crop_seg_1, params.segment_threshold, params.segment_iou)
        mask_seg_2, _ = self.disk_segmentor_yolo.segment_large_image_debug(crop_seg_2, params.segment_threshold, params.segment_iou)

        mask_seg_1 = self.clean_mask(mask_seg_1, params.disk_min_area)
        mask_seg_2 = self.clean_mask(mask_seg_2, params.disk_min_area)

        segment_image = None
        if debug:
            segment_image = np.zeros(crop_img.shape[:2], dtype=np.uint8)
            segment_image[box_seg_1[1]:box_seg_1[3], box_seg_1[0]:box_seg_1[2]] = mask_seg_1
            segment_image[box_seg_2[1]:box_seg_2[3], box_seg_2[0]:box_seg_2[2]] = mask_seg_2

        # Apply caliper at the production measurement positions.
        center_1 = mask_seg_1.shape[1] // 2, int(mask_seg_1.shape[0] * 0.75)
        center_2 = mask_seg_1.shape[1] // 2, int(mask_seg_1.shape[0] * 0.25)
        center_3 = mask_seg_2.shape[1] // 2, int(mask_seg_2.shape[0] * 0.25)
        center_4 = mask_seg_2.shape[1] // 2, int(mask_seg_2.shape[0] * 0.75)
        caliper_res_1 = self.caliper.measure_with_params(mask_seg_1, center_1, params.caliper_min_edge_distance, params.caliper_max_edge_distance, params.caliper_length_rate, params.caliper_thickness_list)
        caliper_res_2 = self.caliper.measure_with_params(mask_seg_1, center_2, params.caliper_min_edge_distance, params.caliper_max_edge_distance, params.caliper_length_rate, params.caliper_thickness_list)
        caliper_res_3 = self.caliper.measure_with_params(mask_seg_2, center_3, params.caliper_min_edge_distance, params.caliper_max_edge_distance, params.caliper_length_rate, params.caliper_thickness_list)
        caliper_res_4 = self.caliper.measure_with_params(mask_seg_2, center_4, params.caliper_min_edge_distance, params.caliper_max_edge_distance, params.caliper_length_rate, params.caliper_thickness_list)

        # Visualize result:
        self.draw_boxes(crop_img, ng_boxes, (0, 0, 255), ClassifyResult.NG)
        self.draw_boxes(crop_img, no_disk_boxes, (0, 102, 255), "Empty")
        self.draw_mask_contour(crop_seg_1, mask_seg_1, center_1)
        self.draw_mask_contour(crop_seg_1, mask_seg_1, center_2)
        self.draw_mask_contour(crop_seg_2, mask_seg_2, center_3)
        self.draw_mask_contour(crop_seg_2, mask_seg_2, center_4)

        res_spacing_1, dis_list_1, mids_1 = self.visualize_edge_spacing(crop_seg_1, caliper_res_1,
                                                                        params.disk_min_distance,
                                                                        params.disk_max_distance)
        res_spacing_2, dis_list_2, mids_2 = self.visualize_edge_spacing(crop_seg_1, caliper_res_2,
                                                                        params.disk_min_distance,
                                                                        params.disk_max_distance)
        res_spacing_3, dis_list_3, mids_3 = self.visualize_edge_spacing(crop_seg_2, caliper_res_3,
                                                                        params.disk_min_distance,
                                                                        params.disk_max_distance)
        res_spacing_4, dis_list_4, mids_4 = self.visualize_edge_spacing(crop_seg_2, caliper_res_4,
                                                                        params.disk_min_distance,
                                                                        params.disk_max_distance)

        # Summary result
        res_classification = InspectionState.NG
        if len(ng_boxes) == 0 and len(no_disk_boxes) == 0:
            res_classification = InspectionState.OK
        elif len(ng_boxes) > 0:
            res_classification = InspectionState.NG
        elif len(no_disk_boxes) > 0:
            res_classification = InspectionState.WARNING

        res_spacing = False not in res_spacing_1 + res_spacing_2 + res_spacing_3 + res_spacing_4
        res_count = (len(caliper_res_1["pairs"]) <= params.disk_num and len(caliper_res_2["pairs"]) <= params.disk_num
                     and len(caliper_res_3["pairs"]) <= params.disk_num and len(caliper_res_4["pairs"]) <= params.disk_num)

        ##########################
        # Summary result
        ##########################
        # -- first value
        sum_res = InspectionState.NG
        error_code = ErrorCode.ABNORMAL[0]
        error_desc = ErrorCode.ABNORMAL[1]
        if res_classification == InspectionState.OK and res_spacing and res_count:
            sum_res = InspectionState.OK
            error_code = ErrorCode.PASS[0]
            error_desc = ErrorCode.PASS[1]
        else:
            if res_classification == InspectionState.WARNING and res_spacing and res_count:
                sum_res = InspectionState.WARNING
                error_code = ErrorCode.WARNING_NUM_DISK[0]
                error_desc = ErrorCode.WARNING_NUM_DISK[1]
            else:
                sum_res = InspectionState.NG
                error_code = ErrorCode.ABNORMAL[0]
                error_desc = ErrorCode.ABNORMAL[1]

        try:
            min_disk_distance = min(dis_list_1 + dis_list_2 + dis_list_3 + dis_list_4)
            max_disk_distance = max(dis_list_1 + dis_list_2 + dis_list_3 + dis_list_4)
        except:
            min_disk_distance = 0
            max_disk_distance = 0

        if debug:
            return DataDebugResponse(Result=sum_res, ErrorCode=error_code, ErrorDesc=error_desc,
                                     DetectImg=self._convert_2_base64(detect_image),
                                     SegmentImg=self._convert_2_base64(segment_image),
                                     FinalImg=self._convert_2_base64(crop_img),
                                     CropBox=str(crop_quad.tolist()), UvBox1=str(uv_box_l1.tolist()),
                                     UvBox2=str(uv_box_l3.tolist()), Mid1=str(mids_1), Mid2=str(mids_3))

        return DataResponse(Result=sum_res,
                            ErrorCode=error_code,
                            ErrorDesc=error_desc,
                            ResImg=self._convert_2_base64(crop_img),
                            MaxDiskDistance=max_disk_distance,
                            MinDiskDistance=min_disk_distance,
                            CropBox=str(crop_quad.tolist()),
                            UvBox1=str(uv_box_l1.tolist()),
                            UvBox2=str(uv_box_l3.tolist()),
                            Mid1=str(mids_1),
                            Mid2=str(mids_3),
                            )

    def update_boxes_after_crop(self, middle_boxes, crop_rect):
        for i in range(len(middle_boxes)):
            middle_boxes[i][0] = middle_boxes[i][0] - crop_rect[0][0]
            middle_boxes[i][1] = middle_boxes[i][1] - crop_rect[0][1]
            middle_boxes[i][2] = middle_boxes[i][2] - crop_rect[0][0]
            middle_boxes[i][3] = middle_boxes[i][3] - crop_rect[0][1]

        return middle_boxes

    # endregion

    # region Check Disk White Swagger
    def check_disk_swagger(self, image):
        result = self.check_disk_white(image)
        return cv2.imdecode(np.frombuffer(base64.b64decode(result.ResImg), dtype=np.uint8), cv2.IMREAD_COLOR)

    def check_disk_uv(self, img, crop_box, uv_box_1, uv_box_2, mid_1, mid_2):
        return self._check_uv(img, crop_box, uv_box_1, uv_box_2, mid_1, mid_2,
                              self.uv_disk_lower_threshold, self.uv_disk_upper_threshold,
                              self.uv_min_disk_area)

    def _check_uv(self, img, crop_box, uv_box_1, uv_box_2, mid_1, mid_2,
                  lower, upper, min_area, debug=False):
        crop_box = np.array(ast.literal_eval(crop_box), dtype=np.float32)
        uv_box_1 = np.array(ast.literal_eval(uv_box_1), dtype=np.int32)
        uv_box_2 = np.array(ast.literal_eval(uv_box_2), dtype=np.int32)
        mid_1 = np.array(ast.literal_eval(mid_1), dtype=np.float32)
        mid_2 = np.array(ast.literal_eval(mid_2), dtype=np.float32)

        self._validate_uv_geometry(img, crop_box, uv_box_1, uv_box_2, mid_1, mid_2)
        crop_img = self.crop_by_4pts(img, crop_box)

        uv_box_1 = np.array(uv_box_1, dtype=np.int32)
        uv_box_2 = np.array(uv_box_2, dtype=np.int32)

        uv_crop_1 = crop_img[uv_box_1[0][1]:uv_box_1[2][1], uv_box_1[0][0]:uv_box_1[2][0]]
        uv_crop_2 = crop_img[uv_box_2[0][1]:uv_box_2[2][1], uv_box_2[0][0]:uv_box_2[2][0]]

        if uv_crop_1.size == 0 or uv_crop_2.size == 0:
            raise ValueError("Empty UV region")

        uv_thresh_1 = self.preprocess_uv_image_hsv(uv_crop_1, lower=tuple(lower), upper=tuple(upper))
        uv_thresh_2 = self.preprocess_uv_image_hsv(uv_crop_2, lower=tuple(lower), upper=tuple(upper))

        uv_thresh_1 = self.remove_mask_noise_uv(uv_thresh_1, min_disk_area=min_area)
        uv_thresh_2 = self.remove_mask_noise_uv(uv_thresh_2, min_disk_area=min_area)

        caliper_res_1 = self.get_caliper_result_debug(uv_thresh_1,
                                                      center=(uv_crop_1.shape[1] // 2, uv_crop_1.shape[0] // 2),
                                                      length_rate=0.95,
                                                      max_edge_distance=50,
                                                      min_edge_distance=5,
                                                      thickness_list=[1, 3])

        caliper_res_2 = self.get_caliper_result_debug(uv_thresh_2,
                                                      center=(uv_crop_2.shape[1] // 2, uv_crop_2.shape[0] // 2),
                                                      length_rate=self.caliper.length_rate,
                                                      max_edge_distance=self.caliper.pair_max_gap,
                                                      min_edge_distance=self.caliper.min_edge_distance,
                                                      thickness_list=self.caliper.thickness_list
                                                      )

        uv_crop_1 = self.draw_uv_mask(uv_crop_1, uv_thresh_1)
        uv_crop_2 = self.draw_uv_mask(uv_crop_2, uv_thresh_2)

        result_1, mid_uv_1 = self.visualize_edge_spacing_uv(uv_crop_1,
                                                            (uv_crop_1.shape[1] // 2, uv_crop_1.shape[0] // 2),
                                                            caliper_res_1)
        result_2, mid_uv_2 = self.visualize_edge_spacing_uv(uv_crop_2,
                                                            (uv_crop_2.shape[1] // 2, uv_crop_2.shape[0] // 2),
                                                            caliper_res_2)

        self.draw_uv_index(uv_crop_1, mid_uv_1, mid_1)
        self.draw_uv_index(uv_crop_2, mid_uv_2, mid_2)

        # Draw segment on crop image
        mask_crop = crop_img.copy()
        # mask_crop = np.zeros((crop_img.shape[0], crop_img.shape[1], 3), dtype=np.uint8)
        mask_crop[uv_box_1[0][1]:uv_box_1[2][1], uv_box_1[0][0]:uv_box_1[2][0]] = result_1
        mask_crop[uv_box_2[0][1]:uv_box_2[2][1], uv_box_2[0][0]:uv_box_2[2][0]] = result_2

        count_uv_disk_1 = len(caliper_res_1["pairs"])
        count_uv_disk_2 = len(caliper_res_2["pairs"])

        count_white_disk_1 = len(mid_1)
        count_white_disk_2 = len(mid_2)

        count_uv_disk = count_uv_disk_1 + count_uv_disk_2

        line_1_ok = count_uv_disk_1 == count_white_disk_1
        line_2_ok = count_uv_disk_2 == count_white_disk_2

        passed = count_uv_disk == 0 or (line_1_ok and line_2_ok)
        # Existing production UI reports zero for an accepted homogeneous tray.
        reported_count = 0 if passed else count_uv_disk
        if debug:
            threshold_image = np.zeros(crop_img.shape[:2], dtype=np.uint8)
            threshold_image[uv_box_1[0][1]:uv_box_1[2][1], uv_box_1[0][0]:uv_box_1[2][0]] = uv_thresh_1
            threshold_image[uv_box_2[0][1]:uv_box_2[2][1], uv_box_2[0][0]:uv_box_2[2][0]] = uv_thresh_2
            return DataDebugUVResponse(Result=passed, CountUvDisk=reported_count,
                                       ThresholdImg=self._convert_2_base64(threshold_image),
                                       FinalImg=self._convert_2_base64(mask_crop))
        error = ErrorCode.PASS if passed else ErrorCode.ERR_MIXING_DISK
        return DataResponseUv(Result=passed, CountUvDisk=reported_count,
                              ErrorCode=error[0], ErrorDesc=error[1],
                              ResImg=self._convert_2_base64(mask_crop))

    @staticmethod
    def _validate_uv_geometry(image, crop, box1, box2, mid1, mid2):
        for box in (crop, box1, box2):
            if box.shape != (4, 2) or not np.isfinite(box).all():
                raise ValueError("UV boxes must contain four finite points")
        for mids in (mid1, mid2):
            if mids.size and (mids.ndim != 2 or mids.shape[1] != 2 or not np.isfinite(mids).all()):
                raise ValueError("Invalid white-light midpoints")
        if (crop[:, 0].min() < 0 or crop[:, 1].min() < 0 or
                crop[:, 0].max() > image.shape[1] or crop[:, 1].max() > image.shape[0]):
            raise ValueError("Crop box is outside the image")
        width = int(max(np.linalg.norm(crop[1] - crop[0]), np.linalg.norm(crop[2] - crop[3])))
        height = int(max(np.linalg.norm(crop[3] - crop[0]), np.linalg.norm(crop[2] - crop[1])))
        for box in (box1, box2):
            # White slicing truncates each x endpoint separately, whereas the existing
            # UV warp truncates their difference. Preserve its one-pixel rounding margin.
            if not (0 <= box[0, 0] < box[2, 0] <= width + 1 and 0 <= box[0, 1] < box[2, 1] <= height):
                raise ValueError("UV region is outside the cropped image")

    @staticmethod
    def preprocess_uv_image(image, threshold):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.medianBlur(gray, 5)
        _, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)

        return thresh

    @staticmethod
    def preprocess_uv_image_hsv(image, lower, upper):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        thresh = cv2.inRange(hsv, lower, upper)

        return thresh

    def get_uv_box(self, box, w, direction="bottom", start_ratio=0.0, ratio_height=2.0):
        x1, y1, x2, y2 = box
        height = y2 - y1
        if direction == "bottom":
            y1_uv = y2 + start_ratio * height
            y2_uv = y1_uv + height * ratio_height + start_ratio * height

            pts_warped = np.array([
                [0, y1_uv],
                [w, y1_uv],
                [w, y2_uv],
                [0, y2_uv]
            ], dtype=np.float32)

        else:
            y1_uv = y1 - height * ratio_height - start_ratio * height
            y2_uv = y1 - start_ratio * height

            pts_warped = np.array([
                [0, y1_uv],
                [w, y1_uv],
                [w, y2_uv],
                [0, y2_uv]
            ], dtype=np.float32)

            # pts_warped = pts_warped.reshape(-1, 1, 2)
            # M_inv = np.linalg.inv(M)
            # pts_image = cv2.perspectiveTransform(pts_warped, M_inv)
            # pts_image = pts_image.reshape(-1, 2)

        return pts_warped

    def draw_uv_mask(self, uv_crop_1, uv_thresh_1, cnt_color=(0, 0, 255), mask_color=(0, 255, 0)):
        contours, _ = cv2.findContours(uv_thresh_1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            cv2.drawContours(uv_crop_1, [contour], -1, cnt_color, 1)
            self.draw_stripes_on_contour_inplace(uv_crop_1, [contour], color=mask_color)

        return uv_crop_1

    def draw_uv_index(self, uv_crop_1, mid_uv_1, mid_1):
        if len(mid_uv_1) < 1 or len(mid_1) < 1:
            return

        query = np.array(mid_uv_1)
        x_pts = mid_1[:, 0]  # (N,)
        x_q = query[:, 0]  # (M,)

        # Tính |x_q - x_pts| cho toàn bộ cặp (broadcast)
        dx = np.abs(x_q[:, None] - x_pts[None, :])  # (M,N)

        # Lấy index pts gần nhất cho từng query
        idxs = np.argmin(dx, axis=1)  # (M,)

        # vẽ số index lên điểm các điểm trong mid_1
        for i, pt in enumerate(mid_uv_1):
            cv2.putText(uv_crop_1, f"{idxs[i] + 1}", (int(pt[0]) - 30, int(pt[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 255), 2)

    # endregion

    # region Check Disk UV Debug
    def check_disk_uv_debug(self, img, params):
        return self._check_uv(img, params.crop_box, params.uv_box_1, params.uv_box_2,
                              params.mid_1, params.mid_2, params.uv_disk_lower_threshold,
                              params.uv_disk_upper_threshold, params.uv_disk_min_area, debug=True)

if __name__ == '__main__':
    import glob
    from tqdm import tqdm
    import os

    IMAGE_PATH = r"D:\huynhvc\OTHERS\disk_checking\disk_checking\raw_data\real_images\empty_disk"
    OUTPUT_PATH = r"D:\huynhvc\OTHERS\disk_checking\disk_checking\datasets\dataset_cls\new_data_22_05_classify"
    save_path_bottom_rect = f"{OUTPUT_PATH}/bottom"
    save_path_top_rect = f"{OUTPUT_PATH}/top"
    os.makedirs(save_path_bottom_rect, exist_ok=True)
    os.makedirs(save_path_top_rect, exist_ok=True)

    disk_checking_service = DiskCheckingService()

    paths = glob.glob(f"{IMAGE_PATH}/*")

    for path in tqdm(paths):
        image = cv2.imread(path)
        boxes, confs, cls_idxs = disk_checking_service.disk_point_detect_model(image)


        # Group and rectify the three point rows exactly as the production path.
        boxes_l1, boxes_l2, boxes_l3 = disk_checking_service.split_rows(boxes)
        crop_img, matrix, _, _ = disk_checking_service.full_rectify_pipeline(
            image, boxes_l1, boxes_l3, expand_ratio_x=0.2, expand_ratio_y=0.1)
        boxes_l1 = disk_checking_service.update_boxes_after_warp(boxes_l1, matrix)
        boxes_l2 = disk_checking_service.update_boxes_after_warp(boxes_l2, matrix)
        boxes_l3 = disk_checking_service.update_boxes_after_warp(boxes_l3, matrix)

        # region GET THE CLASSIFICATION BOXES

        classify_regions = (
            (boxes_l1, "bottom"),
            (boxes_l2, "top"),
            (boxes_l2, "bottom"),
            (boxes_l3, "top"),
        )
        region_crops = []
        for row_boxes, direction in classify_regions:
            rects = disk_checking_service.get_line_boxes_ratio_shift(crop_img, row_boxes, direction)
            region_crops.append(disk_checking_service.crop_boxes(crop_img, rects, direction))

        bottom_crops = region_crops[0] + region_crops[2]
        top_crops = region_crops[1] + region_crops[3]
        for i, crop_rect in enumerate(bottom_crops):
            img_name = os.path.basename(path).replace('.bmp', f'_{i}.bmp')
            cv2.imwrite(fr"{save_path_bottom_rect}/{img_name}", crop_rect)

        for j, crop_rect in enumerate(top_crops):
            img_name = os.path.basename(path).replace('.bmp', f'_{len(bottom_crops) + j}.bmp')
            cv2.imwrite(fr"{save_path_top_rect}/{img_name}", crop_rect)

        # endregion

        # region GET THE CROPS FOR SEGMENTATION

        crop_seg_1, _ = disk_checking_service.crop_between_rows(
            crop_img, boxes_l1, boxes_l2, ratio=0.35, direction='bottom')
        crop_seg_2, _ = disk_checking_service.crop_between_rows(
            crop_img, boxes_l2, boxes_l3, ratio=0.35, direction='top')
        # crop boxes
        def crop_images(image):
            H, W = image.shape[:2]
            step = W // 6
            overlap = int(step * 0.05)

            crops = []
            positions = []

            x = 0
            while x < W and len(crops) < 6:
                x_start = max(0, x - overlap)
                x_end = min(W, x + step + overlap)

                crop = image[:, x_start:x_end]
                crops.append(crop)
                positions.append((x_start, x_end))

                x += step

            return crops


        crops_1 = crop_images(crop_seg_1)
        crops_2 = crop_images(crop_seg_2)

        for i, crop in enumerate(crops_1 + crops_2):
            img_name = os.path.basename(path).replace('.bmp', f'_crop_{i}.bmp')
            cv2.imwrite(fr"D:\huynhvc\OTHERS\disk_checking\disk_checking\datasets\dataset_segment\new_data_22_05\images/{img_name}",
                        crop)

        # img_name_1 = os.path.basename(path).replace('.bmp', f'_seg_1.bmp')
        # img_name_2 = os.path.basename(path).replace('.bmp', f'_seg_2.bmp')
        # cv2.imwrite(fr"D:\huynhvc\OTHERS\disk_checking\disk_checking\testing\out_rect_segment/{img_name_1}", crop_seg_1)
        # cv2.imwrite(fr"D:\huynhvc\OTHERS\disk_checking\disk_checking\testing\out_rect_segment/{img_name_2}", crop_seg_2)

        # endregion
