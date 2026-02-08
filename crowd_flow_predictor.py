import cv2
import numpy as np

class CrowdFlowPredictor:
    def __init__(self, alpha=0.1):
        self.alpha = alpha

    def predict(self, density, flow):
        """
        density: (H_d, W_d)
        flow: (H_f, W_f, 2)
        """

        H, W = density.shape

        # Resize flow to density resolution
        fx = cv2.resize(flow[..., 0], (W, H))
        fy = cv2.resize(flow[..., 1], (W, H))

        # Divergence
        div_x = np.gradient(fx, axis=1)
        div_y = np.gradient(fy, axis=0)
        divergence = div_x + div_y

        future_density = density - self.alpha * density * divergence
        return np.clip(future_density, 0, None)
