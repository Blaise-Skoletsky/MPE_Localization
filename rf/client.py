import asyncio
import json
import logging
import board
import socket
from adafruit_as7341 import AS7341
from WiFiScanner import WiFiScanner

# --- Configuration ---
SERVER_IP = '172.20.10.3'  # <--- CHANGE TO YOUR COMPUTER IP
SERVER_PORT = 65432
SEND_INTERVAL = 0.5          # Send data every 0.5 seconds
WIFI_SCAN_INTERVAL = 4.0     # Scan WiFi every 4 seconds (runs in background)

logging.basicConfig(level=logging.INFO, format='[Pi] %(message)s')
logger = logging.getLogger(__name__)

class DataStreamer:
    def __init__(self):
        self.writer = None
        self.reader = None
        
        # Init Light Sensor
        try:
            i2c = board.I2C()
            self.light_sensor = AS7341(i2c)
            logger.info("Light sensor initialized.")
        except Exception as e:
            self.light_sensor = None
            logger.error(f"Light sensor init failed: {e}")

        # Init WiFi Scanner
        self.wifi_scanner = WiFiScanner(interface='wlan0', scan_interval=WIFI_SCAN_INTERVAL)

    async def connect(self):
        """Connect to the PC server, retrying if necessary."""
        while True:
            try:
                logger.info(f"Connecting to {SERVER_IP}:{SERVER_PORT}...")
                self.reader, self.writer = await asyncio.open_connection(SERVER_IP, SERVER_PORT)
                logger.info("Connected!")
                return
            except Exception as e:
                logger.warning(f"Connection failed: {e}. Retrying in 2s...")
                await asyncio.sleep(2)

    def read_light(self):
        """Reads AS7341 channels."""
        if not self.light_sensor:
            return {}
        try:
            # RFLocalizer expects keys: 'f1'...'f8', 'clear', 'nir'
            data = {}
            for i, val in enumerate(self.light_sensor.all_channels, 1):
                data[f"f{i}"] = val
            data["clear"] = self.light_sensor.channel_clear
            data["nir"] = self.light_sensor.channel_nir
            return data
        except Exception as e:
            logger.error(f"Light read error: {e}")
            return {}

    def read_wifi(self):
        """Gets latest WiFi scan results."""
        try:
            # RFLocalizer expects keys: 'AA:BB:CC:DD:EE:FF' (BSSID)
            # We get all networks; the server filters relevant ones.
            networks = self.wifi_scanner.get_networks(min_signal=-100)
            return {bssid: net.signal_strength for bssid, net in networks.items()}
        except Exception as e:
            logger.error(f"WiFi read error: {e}")
            return {}

    async def run(self):
        # Start background wifi scanning
        await self.wifi_scanner.start_continuous_scan()
        
        # Connect to server
        await self.connect()

        try:
            while True:
                # 1. Collect Data
                light_data = self.read_light()
                wifi_data = self.read_wifi()

                # 2. Package into JSON
                packet = {
                    "light": light_data,
                    "wifi": wifi_data
                }
                
                # 3. Send (Newline delimited)
                message = json.dumps(packet) + "\n"
                self.writer.write(message.encode())
                await self.writer.drain()
                
                logger.info(f"Sent: Light={len(light_data)}ch, WiFi={len(wifi_data)}APs")
                
                # 4. Wait
                await asyncio.sleep(SEND_INTERVAL)

        except (ConnectionResetError, BrokenPipeError):
            logger.error("Connection lost.")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
        finally:
            logger.info("Shutting down...")
            await self.wifi_scanner.stop_continuous_scan()
            if self.writer:
                self.writer.close()
                await self.writer.wait_closed()

if __name__ == "__main__":
    streamer = DataStreamer()
    try:
        asyncio.run(streamer.run())
    except KeyboardInterrupt:
        pass
