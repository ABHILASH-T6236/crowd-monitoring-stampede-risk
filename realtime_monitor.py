import cv2
import numpy as np

class RealtimeMonitor:
    def __init__(self, grid_rows=4, grid_cols=4):
        self.prev_gray = None
        self.frame_id = 0
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.last_flow = None


    # --------------------
    # MOTION COMPUTATION
    # --------------------
    def compute_motion(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        motion_score = 0.0

        if self.prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_gray, gray,
                None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            motion_score = np.mean(mag)
            self.last_flow = flow   # ⬅ STORE FLOW

        self.prev_gray = gray
        return motion_score

    # --------------------
    # DENSITY LEVEL
    # --------------------
    def density_level(self, avg_density):
        if avg_density < 0.01:
            return "LOW", (0, 255, 0)
        elif avg_density < 0.05:
            return "MEDIUM", (0, 255, 255)
        else:
            return "HIGH", (0, 0, 255)

    # --------------------
    # RISK LEVEL
    # --------------------
    def risk_level(self, sri):
        if sri < 0.002:
            return "LOW", (0, 255, 0)
        elif sri < 0.006:
            return "MEDIUM", (0, 255, 255)
        else:
            return "HIGH", (0, 0, 255)

    # --------------------
    # GRID OVERCROWD
    # --------------------
    def grid_danger_zones(self, density):
        h, w = density.shape
        gh, gw = h // self.grid_rows, w // self.grid_cols
        danger = []

        for i in range(self.grid_rows):
            for j in range(self.grid_cols):
                count = density[i*gh:(i+1)*gh, j*gw:(j+1)*gw].sum()
                if count > 10:
                    danger.append((i, j))

        return danger

    # --------------------
    # OVERLAY VALUES
    # --------------------
    def draw_overlay(self, frame, values):
        self.frame_id += 1
        output = frame.copy()

        texts = [
            f"Frame: {self.frame_id}",
            f"Count: {int(values['count'])}",
            f"Avg Density: {values['avg_density']:.4f}",
            f"Motion Score: {values['motion_score']:.4f}",
            f"SRI: {values['sri']:.5f}",
            f"Risk: {values['risk_level']}"
        ]

        y = 30
        for t in texts:
            cv2.putText(output, t, (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        values["risk_color"], 2)
            y += 30

        return output

    # --------------------
    # TERMINAL LOG
    # --------------------
    def log(self, values):
        print(
            f"[Frame {self.frame_id}] "
            f"Count={values['count']:.1f} | "
            f"AvgDensity={values['avg_density']:.4f} | "
            f"Motion={values['motion_score']:.4f} | "
            f"SRI={values['sri']:.5f} | "
            f"Risk={values['risk_level']}"
        )
