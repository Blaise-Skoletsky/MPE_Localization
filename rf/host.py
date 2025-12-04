import asyncio
import json
import logging
import argparse
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any

# --- Import Paths Setup ---
# Add subdirectories to path to find models
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'knn'))
sys.path.append(os.path.join(current_dir, 'rf'))

# Conditional Imports
try:
    from realtime_localization import RealtimeLocalizer
    KNN_AVAILABLE = True
except ImportError:
    KNN_AVAILABLE = False

try:
    from rf_inference import RFLocalizer
    RF_AVAILABLE = True
except ImportError:
    RF_AVAILABLE = False

# --- Configuration ---
HOST = '0.0.0.0'
PORT = 65432

# Configure logging to show timestamps
logging.basicConfig(level=logging.INFO, format='%(asctime)s [Server] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

class UnifiedServer:
    def __init__(self, model_type: str):
        self.model_type = model_type
        self.localizer = None
        
        # Initialize the selected model
        if model_type == 'knn':
            if not KNN_AVAILABLE:
                raise ImportError("KNN code not found in 'knn/' folder.")
            logger.info("Initializing KNN Model...")
            self.localizer = RealtimeLocalizer(enable_plot=False, verbose=False)
            
        elif model_type == 'rf':
            if not RF_AVAILABLE:
                raise ImportError("RF code not found in 'rf/' folder.")
            logger.info("Initializing Random Forest Model...")
            self.localizer = RFLocalizer(wifi_weight=0.5, light_weight=0.5, verbose=False)

        # Setup Plotting
        self._init_plot()

    def _init_plot(self):
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.ax.set_title(f'Realtime Localization ({self.model_type.upper()})')
        self.ax.set_xlabel('X (cm)')
        self.ax.set_ylabel('Y (cm)')
        self.ax.set_xlim(250, 750)
        self.ax.set_ylim(150, 600)
        self.ax.grid(True, linestyle='--', alpha=0.6)
        
        self.scatter, = self.ax.plot([], [], 'bo', markersize=10, label='Current')
        self.trail_x, self.trail_y = [], []
        self.trail_line, = self.ax.plot([], [], 'b-', alpha=0.3, label='Trail')
        self.ax.legend()

    def update_plot(self, x, y):
        self.scatter.set_data([x], [y])
        self.trail_x.append(x)
        self.trail_y.append(y)
        if len(self.trail_x) > 50:
            self.trail_x.pop(0)
            self.trail_y.pop(0)
        self.trail_line.set_data(self.trail_x, self.trail_y)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def predict(self, raw_packet: Dict[str, Any]):
        wifi_data = raw_packet.get('wifi', {})
        light_data = raw_packet.get('light', {})

        if self.model_type == 'knn':
            # Adapter for KNN: Flatten and add prefixes
            combined_features = {}
            for k, v in wifi_data.items():
                combined_features[f"wifi_{k}"] = float(v)
            for k, v in light_data.items():
                combined_features[f"light_{k}"] = float(v)
            return self.localizer.predict_sample(combined_features, update_plot=False)

        elif self.model_type == 'rf':
            # Adapter for RF: Pass raw dicts directly
            if not wifi_data and not light_data:
                return None
            result = self.localizer.predict_weighted(wifi_data, light_data)
            return result[0]

    async def handle_client(self, reader, writer):
        addr = writer.get_extra_info('peername')
        logger.info(f"Connected to {addr}")
        buffer = ""

        try:
            while True:
                data = await reader.read(4096)
                if not data: break
                
                buffer += data.decode()
                while "\n" in buffer:
                    message, buffer = buffer.split("\n", 1)
                    if not message.strip(): continue
                    
                    try:
                        packet = json.loads(message)
                        
                        # --- [ADDED DEBUG LINE] ---
                        # This confirms data arrived and was parsed successfully
                        wifi_count = len(packet.get('wifi', {}))
                        light_count = len(packet.get('light', {}))
                        logger.info(f"DEBUG: Rx Packet | WiFi APs: {wifi_count} | Light Ch: {light_count}")
                        # --------------------------

                        prediction = self.predict(packet)
                        
                        if prediction is not None:
                            x, y = prediction[0], prediction[1]
                            logger.info(f"[{self.model_type.upper()}] Pos: ({x:.1f}, {y:.1f})")
                            self.update_plot(x, y)
                        else:
                            logger.warning("Prediction returned None (insufficient data?)")

                    except json.JSONDecodeError:
                        logger.error("JSON Decode Error: Received malformed packet")
                    except Exception as e:
                        logger.error(f"Processing Error: {e}")

        except asyncio.CancelledError:
            pass
        finally:
            logger.info("Connection closed")
            writer.close()
            await writer.wait_closed()

    async def start(self):
        server = await asyncio.start_server(self.handle_client, HOST, PORT)
        logger.info(f"Server listening on {HOST}:{PORT} using [{self.model_type.upper()}] model...")
        async with server:
            await server.serve_forever()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Realtime Localization Server')
    parser.add_argument('--model', type=str, choices=['knn', 'rf'], default='rf',
                        help='Choose which model to use: "knn" or "rf" (default: rf)')
    args = parser.parse_args()

    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    server = UnifiedServer(args.model)
    try:
        asyncio.run(server.start())
    except KeyboardInterrupt:
        print("\nStopping server...")
