import time
import numpy as np
import cv2
import math
from typing import List, Dict, Tuple, Any, Optional


class AdvancedMultiEdgeCaliper:
    """
    Industrial-style multi-edge caliper
    - No image rotation
    - Subpixel sampling
    - Mask-friendly (0/1 or 0/255)
    - Clean edge model
    """

    def __init__(
            self,
            min_edge_distance: float = 5.0,
            subpixel: bool = True,
            max_pairs: int = 10,
            pair_max_gap: Optional[float] = None,
            thickness_list: Optional[List[float]] = None,
            length_rate: float = 0.9,
            polarity: str = "both",
            angle_deg: Optional[float] = None,
            return_profiles: bool = False,
    ):
        self.min_edge_distance = float(min_edge_distance)
        self.subpixel = bool(subpixel)
        self.max_pairs = int(max_pairs)
        self.pair_max_gap = pair_max_gap
        self.thickness_list = thickness_list
        self.length_rate = float(length_rate)
        self.polarity = str(polarity)
        self.angle_deg = angle_deg
        self.return_profiles = bool(return_profiles)

    # ------------------------------------------------------------
    @staticmethod
    def _bilinear(img: np.ndarray, x: float, y: float) -> float:
        h, w = img.shape[:2]
        if x < 0 or x >= w - 1 or y < 0 or y >= h - 1:
            xi = int(np.clip(round(x), 0, w - 1))
            yi = int(np.clip(round(y), 0, h - 1))
            return float(img[yi, xi])

        x0, y0 = int(x), int(y)
        x1, y1 = x0 + 1, y0 + 1
        dx, dy = x - x0, y - y0

        Ia = img[y0, x0]
        Ib = img[y0, x1]
        Ic = img[y1, x0]
        Id = img[y1, x1]

        return (
            Ia * (1 - dx) * (1 - dy) +
            Ib * dx * (1 - dy) +
            Ic * (1 - dx) * dy +
            Id * dx * dy
        )

    # ------------------------------------------------------------
    @staticmethod
    def _sample_profile_vectorized(
            img: np.ndarray,
            center: Tuple[float, float],
            angle_deg: float,
            length: int,
            thickness: float
    ) -> np.ndarray:

        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        h, w = img.shape
        cx, cy = center

        theta = np.deg2rad(angle_deg)
        dx, dy = np.cos(theta), np.sin(theta)
        nx, ny = -dy, dx

        half_len = (length - 1) / 2.0
        half_th = thickness / 2.0

        jmin = int(np.ceil(-half_th))
        jmax = int(np.floor(half_th))

        if jmin > jmax:
            jmin = jmax = 0

        js = np.arange(jmin, jmax + 1, dtype=np.float32)  # thickness axis
        ts = np.arange(length, dtype=np.float32) - half_len  # length axis

        # meshgrid: shape = (H, W)
        t_grid, j_grid = np.meshgrid(ts, js)

        xs = cx + dx * t_grid + nx * j_grid
        ys = cy + dy * t_grid + ny * j_grid

        # clamp biên
        xs = np.clip(xs, 0, w - 1.001)
        ys = np.clip(ys, 0, h - 1.001)

        x0 = xs.astype(np.int32)
        y0 = ys.astype(np.int32)
        x1 = np.clip(x0 + 1, 0, w - 1)
        y1 = np.clip(y0 + 1, 0, h - 1)

        dxs = xs - x0
        dys = ys - y0

        Ia = img[y0, x0]
        Ib = img[y0, x1]
        Ic = img[y1, x0]
        Id = img[y1, x1]

        patch = (
                Ia * (1 - dxs) * (1 - dys) +
                Ib * dxs * (1 - dys) +
                Ic * (1 - dxs) * dys +
                Id * dxs * dys
        )

        profile = patch.mean(axis=0)

        return profile.astype(np.float32)

    # ------------------------------------------------------------
    def _pair_edges(self, edges: List[Dict[str, Any]], pair_max_gap) -> List[Dict[str, Any]]:
        pairs = []
        used = [False] * len(edges)

        for i, e1 in enumerate(edges):
            if used[i]:
                continue
            for j in range(i + 1, len(edges)):
                if used[j]:
                    continue
                e2 = edges[j]
                if e1["sign"] + e2["sign"] != 0:
                    continue

                dx = e2["point"][0] - e1["point"][0]
                dy = e2["point"][1] - e1["point"][1]
                dist = np.hypot(dx, dy)

                if pair_max_gap is not None and dist > pair_max_gap:
                    continue

                pairs.append({
                    "e1": e1,
                    "e2": e2,
                    "distance": dist
                })
                used[i] = used[j] = True
                break

            if len(pairs) >= self.max_pairs:
                break

        return pairs

    # ------------------------------------------------------------
    def measure(self, img_gray: np.ndarray, center: Tuple[float, float]) -> Dict[str, Any]:
        return self.measure_with_params(img_gray, center, self.min_edge_distance,
                                        self.pair_max_gap, self.length_rate, self.thickness_list)

    def measure_with_params(self, img_gray, center, min_edge_distance,
                            max_edge_distance, length_rate, thickness_list,
                            debug_coordinates=False):
        """Return a request-owned measurement; never store image results on the shared instance."""
        if img_gray.ndim != 2 or img_gray.size == 0:
            raise ValueError("Caliper requires a non-empty grayscale image")
        cx, cy = center
        length = int(length_rate * img_gray.shape[1])

        best_edges = []
        profiles = {}

        theta = np.deg2rad(self.angle_deg)
        dx, dy = np.cos(theta), np.sin(theta)
        # Preserve the existing UV coordinate convention used by measure_debug.
        half_len = length / 2.0 if debug_coordinates else (length - 1) / 2.0

        for t in thickness_list:
            if t <= 0:
                continue

            profile = self._sample_profile_vectorized(
                img_gray, center, self.angle_deg, length, t
            )
            if self.return_profiles:
                profiles[t] = profile

            detected = self._detect_edges_mask(
                profile, self.polarity, min_edge_distance
            )
            if not detected:
                continue

            edges_this = []
            for e in detected:
                tpos = e["index"] - half_len
                x = cx + tpos * dx
                y = cy + tpos * dy

                edges_this.append({
                    "point": (float(x), float(y)),
                    "index": e["index"],
                    "sign": e["sign"],
                    "polarity": e["polarity"],
                    "strength": e["strength"],
                    "thickness": t
                })

            if len(edges_this) > len(best_edges):
                best_edges = edges_this

        edges = sorted(best_edges, key=lambda e: e["index"])
        result = {"edges": edges, "pairs": self._pair_edges(edges, max_edge_distance)}
        if self.return_profiles:
            result["profiles"] = profiles
        return result

    def _detect_edges_mask(
            self,
            profile: np.ndarray,
            polarity: str,
            min_edge_distance: float,
            grad_thresh: float = 0.15
    ):
        p = profile.astype(np.float32)
        if p.size < 3:
            return []

        # --- smoothing ---
        kernel = np.array([1, 2, 1], dtype=np.float32)
        kernel /= kernel.sum()
        p_s = np.convolve(p, kernel, mode="same")

        # --- gradient ---
        dp = np.diff(p_s)
        g = np.abs(dp)

        if g.max() < 1e-6:
            return []

        # --- Non-maximum suppression ---
        g_prev = np.r_[g[0], g[:-1]]
        g_next = np.r_[g[1:], g[-1]]
        peaks = (g >= g_prev) & (g >= g_next)

        # --- threshold ---
        g_norm = g / (g.max() + 1e-6)
        idx = np.where((g_norm > grad_thresh) & peaks)[0]

        if idx.size == 0:
            return []

        signs = np.sign(dp[idx])
        strengths = g[idx]

        # --- polarity filter ---
        if polarity == "positive":
            mask = signs > 0
        elif polarity == "negative":
            mask = signs < 0
        else:
            mask = np.ones_like(signs, dtype=bool)

        idx = idx[mask]
        signs = signs[mask]
        strengths = strengths[mask]

        if idx.size == 0:
            return []

        # --- subpixel ---
        if self.subpixel:
            indices = idx.astype(np.float32) + 0.5
        else:
            indices = idx.astype(np.float32)

        # --- min distance suppression ---
        edges = []
        last_index = -1e9

        for ind, s, st in zip(indices, signs, strengths):
            if ind - last_index >= min_edge_distance:
                edges.append({
                    "index": float(ind),
                    "sign": int(s),
                    "polarity": "positive" if s > 0 else "negative",
                    "strength": float(st)
                })
                last_index = ind

        return edges

    def measure_debug(
        self,
        img_gray: np.ndarray,
        center: Tuple[float, float],
        min_edge_distance: float,
        max_edge_distance: float,
        length_rate: float,
        thickness_list: Optional[List[int]],
    ) -> Dict[str, Any]:

        return self.measure_with_params(img_gray, center, min_edge_distance,
                                        max_edge_distance, length_rate, thickness_list,
                                        debug_coordinates=True)

    def visualize(
        self,
        img_color: np.ndarray,
        center: Tuple[float, float],
        show_index: bool = True,
        show_distance: bool = True,
        result: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        Visualize result of measure()
        - img_color: BGR image
        - center, angle_deg, length: same as measure
        """
        out = img_color.copy()
        if result is None:
            result = self.measure(cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY), center)
        cx, cy = center
        theta = np.deg2rad(self.angle_deg)
        length = int(self.length_rate * img_color.shape[1])

        dx, dy = np.cos(theta), np.sin(theta)
        nx, ny = -dy, dx

        half_len = length / 2.0

        # ---------------- caliper line ----------------
        p1 = (
            int(round(cx - dx * half_len)),
            int(round(cy - dy * half_len))
        )
        p2 = (
            int(round(cx + dx * half_len)),
            int(round(cy + dy * half_len))
        )
        cv2.line(out, p1, p2, (0, 255, 0), 2)

        # center mark
        cv2.circle(out, (int(cx), int(cy)), 2, (0, 255, 255), -1)

        # ---------------- edges ----------------
        for e in result["edges"]:
            x, y = map(lambda v: int(round(v)), e["point"])

            if e["polarity"] == "positive":
                color = (255, 0, 0)   # blue
            else:
                color = (0, 0, 255)   # red

            cv2.circle(out, (x, y), 4, color, -1)

            if show_index:
                cv2.putText(
                    out,
                    f"{e['index']:.1f}",
                    (x + 6, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    color,
                    1,
                    cv2.LINE_AA
                )

        # ---------------- pairs ----------------
        for p in result["pairs"]:
            x1, y1 = map(lambda v: int(round(v)), p["e1"]["point"])
            x2, y2 = map(lambda v: int(round(v)), p["e2"]["point"])

            cv2.line(out, (x1, y1), (x2, y2), (0, 255, 255), 1)

            if show_distance:
                mx = int(round((x1 + x2) / 2))
                my = int(round((y1 + y2) / 2))
                cv2.putText(
                    out,
                    f"{p['distance']:.2f}",
                    (mx + 4, my - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA
                )

        # ---------------- normal direction (optional debug) ----------------
        # draw normal arrow (small)
        nlen = max(15, int(length * 0.05))
        pn = (
            int(round(cx + nx * nlen)),
            int(round(cy + ny * nlen))
        )
        cv2.arrowedLine(out, (int(cx), int(cy)), pn, (255, 255, 0), 1, tipLength=0.3)

        return out


if __name__ == "__main__":
    import cv2
    img = cv2.imread(r"D:\huynhvc\OTHERS\disk_checking\CORE\caliper\mask_seg_2_channel_0.png")

    cal = AdvancedMultiEdgeCaliper(
        min_edge_distance=6,
        subpixel=True,
        max_pairs=50,
        pair_max_gap=15  # px, optional
    )
    time_st = time.time()
    img_h, img_w, _ = img.shape

    center = (img_w//2, img_h//2)  # điểm đặt thước
    angle_deg = 0  # thước song song trục X (0°)
    length = int(img_w*0.95)
    thickness_list = [5, 10, 15]

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    res = cal.measure(img_gray, center=center, angle_deg=angle_deg,
                      length=length, thickness_list=thickness_list,
                      return_profiles=True, polarity="both")

    print("Time:", time.time() - time_st)
    vis = cal.visualize(img, center=center, angle_deg=angle_deg, length=length)
    pass
