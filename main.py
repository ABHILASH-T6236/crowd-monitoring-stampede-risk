import cv2
import torch
import numpy as np
from csrnet import CSRNet
from realtime_monitor import RealtimeMonitor
from crowd_flow_predictor import CrowdFlowPredictor

predictor = CrowdFlowPredictor(alpha=0.1)


# --------------------
# DEVICE
# --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------
# LOAD MODEL
# --------------------
model = CSRNet()
checkpoint = torch.load("pretrained/csrnet.pth", map_location=device)

state_dict = checkpoint["state_dict"]
clean_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

model.load_state_dict(clean_state_dict)
model.to(device)
model.eval()

print("CSRNet loaded successfully")

# --------------------
# VIDEO INPUT
# --------------------
cap = cv2.VideoCapture("STAMPEDE.mp4")

# --------------------
# REALTIME MONITOR
# --------------------
monitor = RealtimeMonitor(grid_rows=4, grid_cols=4)

# --------------------
# EVALUATION STORAGE
# --------------------
pred_counts = []
avg_densities = []
motion_scores = []
sri_values = []

# --------------------
# PREPROCESS
# --------------------
def preprocess(frame):
    frame = cv2.resize(frame, (640, 480))
    img = frame.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)
    img = torch.tensor(img).unsqueeze(0)
    return img.to(device)

# --------------------
# MAIN LOOP
# --------------------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # --------------------
    # MODEL INFERENCE
    # --------------------
    input_tensor = preprocess(frame)
    with torch.no_grad():
        density_map = model(input_tensor)
        count = density_map.sum().item()

    density = density_map.squeeze().cpu().numpy()
    avg_density = np.mean(density)
    flow = monitor.last_flow
    if flow is not None:
        future_density = predictor.predict(density, flow)
        future_avg_density = np.mean(future_density)
    else:
        future_avg_density = avg_density
    
    print(f"Now Density: {avg_density:.4f} | "
    f"Predicted Density: {future_avg_density:.4f}")


    # --------------------
    # MOTION + RISK
    # --------------------
    motion_score = monitor.compute_motion(frame)
    sri = avg_density * motion_score
    flow = monitor.last_flow
    if flow is not None:
        print("Flow shape:", flow.shape)


    

    # --------------------
    # STORE VALUES (CORRECT PLACE)
    # --------------------
    pred_counts.append(count)
    avg_densities.append(avg_density)
    motion_scores.append(motion_score)
    sri_values.append(sri)

    # --------------------
    # REALTIME ANALYTICS
    # --------------------
    density_level, density_color = monitor.density_level(avg_density)
    risk_level, risk_color = monitor.risk_level(sri)
    danger_zones = monitor.grid_danger_zones(density)

    # --------------------
    # HEATMAP
    # --------------------
    density_norm = cv2.normalize(density, None, 0, 255, cv2.NORM_MINMAX)
    density_norm = density_norm.astype(np.uint8)

    heatmap = cv2.applyColorMap(density_norm, cv2.COLORMAP_JET)
    heatmap = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))

    output = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)

    # --------------------
    # DRAW GRID
    # --------------------
    frame_h, frame_w, _ = frame.shape
    cell_h = frame_h // monitor.grid_rows
    cell_w = frame_w // monitor.grid_cols

    for i in range(monitor.grid_rows):
        for j in range(monitor.grid_cols):
            x1, y1 = j * cell_w, i * cell_h
            x2, y2 = x1 + cell_w, y1 + cell_h
            color, thickness = ((0, 0, 255), 3) if (i, j) in danger_zones else ((200, 200, 200), 1)
            cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)

    # --------------------
    # OVERLAY + LOGGING
    # --------------------
    values = {
        "count": count,
        "avg_density": avg_density,
        "motion_score": motion_score,
        "sri": sri,
        "risk_level": risk_level,
        "risk_color": risk_color
    }

    output = monitor.draw_overlay(output, values)
    monitor.log(values)

    # --------------------
    # DISPLAY
    # --------------------
    cv2.imshow("Crowd Monitoring - Realtime Values", output)
    if cv2.waitKey(1) & 0xFF == 27:
        break

# --------------------
# UNSUPERVISED EVALUATION
# --------------------
from evaluation_unsupervised import (
    temporal_consistency_error,
    motion_density_correlation,
    risk_event_responsiveness
)

tce = temporal_consistency_error(pred_counts)
mdc = motion_density_correlation(avg_densities, motion_scores)
rer = risk_event_responsiveness(sri_values)

print("\n--- UNSUPERVISED EVALUATION RESULTS ---")
print(f"TCE (↓ better): {tce:.4f}")
print(f"MDC (↑ better): {mdc:.4f}")
print(f"RER (↑ better): {rer:.4f}")

# --------------------
# CLEANUP
# --------------------
cap.release()
cv2.destroyAllWindows()
