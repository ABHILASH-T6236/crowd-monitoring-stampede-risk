# import cv2
# import torch
# import numpy as np
# from csrnet import CSRNet

# # Device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Load model
# model = CSRNet()
# checkpoint = torch.load("pretrained/csrnet.pth", map_location=device)

# state_dict = checkpoint["state_dict"]
# new_state_dict = {}
# for k, v in state_dict.items():
#     if k.startswith("module."):
#         new_state_dict[k[7:]] = v
#     else:
#         new_state_dict[k] = v

# model.load_state_dict(new_state_dict)
# model.to(device)
# model.eval()

# print("CSRNet ready")

# # Camera
# cap = cv2.VideoCapture("STAMPEDE.mp4")


# def preprocess(frame):
#     frame = cv2.resize(frame, (640, 480))
#     img = frame.astype(np.float32) / 255.0
#     img = img.transpose(2, 0, 1)
#     img = torch.tensor(img).unsqueeze(0)
#     return img.to(device)

#     with torch.no_grad():
#         density_map = model(input_tensor)
#         count = density_map.sum().item()


#     density = density_map.squeeze().cpu().numpy()# Convert density map to numpy

    
#     density_norm = cv2.normalize(density, None, 0, 255, cv2.NORM_MINMAX)# Normalize for visualization
#     density_norm = density_norm.astype(np.uint8)

    
#     heatmap = cv2.applyColorMap(density_norm, cv2.COLORMAP_JET)# Heatmap
