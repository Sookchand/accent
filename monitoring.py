import time
import os
import json
import logging
from datetime import datetime
import threading
import psutil
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("monitoring.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("accent-detector-monitor")

class AccentDetectorMonitor:
    """
    A simple monitoring system for the Accent Detector application.
    
    This class provides basic monitoring capabilities:
    - CPU and memory usage
    - Application health checks
    - Request metrics
    - Error logging
    """
    
    def __init__(self, app_url="http://localhost:8501", check_interval=60):
        """
        Initialize the monitor.
        
        Args:
            app_url (str): The URL of the application to monitor
            check_interval (int): The interval in seconds between health checks
        """
        self.app_url = app_url
        self.check_interval = check_interval
        self.metrics = {
            "requests": 0,
            "errors": 0,
            "processing_times": [],
            "last_check": None,
            "health_status": "unknown"
        }
        self.running = False
        self.metrics_file = "metrics.json"
        
        # Load existing metrics if available
        self._load_metrics()
    
    def _load_metrics(self):
        """Load metrics from file if it exists."""
        try:
            if os.path.exists(self.metrics_file):
                with open(self.metrics_file, 'r') as f:
                    self.metrics = json.load(f)
                logger.info(f"Loaded metrics from {self.metrics_file}")
        except Exception as e:
            logger.error(f"Error loading metrics: {str(e)}")
    
    def _save_metrics(self):
        """Save metrics to file."""
        try:
            with open(self.metrics_file, 'w') as f:
                json.dump(self.metrics, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving metrics: {str(e)}")
    
    def record_request(self, processing_time=None, error=False):
        """
        Record a request to the application.
        
        Args:
            processing_time (float): The time taken to process the request in seconds
            error (bool): Whether the request resulted in an error
        """
        self.metrics["requests"] += 1
        
        if error:
            self.metrics["errors"] += 1
        
        if processing_time is not None:
            self.metrics["processing_times"].append(processing_time)
            
            # Keep only the last 100 processing times
            if len(self.metrics["processing_times"]) > 100:
                self.metrics["processing_times"] = self.metrics["processing_times"][-100:]
        
        self._save_metrics()
    
    def check_health(self):
        """
        Check the health of the application.
        
        Returns:
            bool: True if the application is healthy, False otherwise
        """
        try:
            response = requests.get(f"{self.app_url}/healthz", timeout=5)
            healthy = response.status_code == 200
            self.metrics["health_status"] = "healthy" if healthy else "unhealthy"
            self.metrics["last_check"] = datetime.now().isoformat()
            self._save_metrics()
            return healthy
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            self.metrics["health_status"] = "unhealthy"
            self.metrics["last_check"] = datetime.now().isoformat()
            self._save_metrics()
            return False
    
    def get_system_metrics(self):
        """
        Get system metrics (CPU, memory).
        
        Returns:
            dict: A dictionary containing system metrics
        """
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            metrics = {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_used_mb": memory.used / (1024 * 1024),
                "memory_total_mb": memory.total / (1024 * 1024)
            }
            
            return metrics
        except Exception as e:
            logger.error(f"Error getting system metrics: {str(e)}")
            return {}
    
    def _monitoring_loop(self):
        """The main monitoring loop."""
        while self.running:
            try:
                # Check application health
                healthy = self.check_health()
                logger.info(f"Health check: {'Healthy' if healthy else 'Unhealthy'}")
                
                # Get system metrics
                system_metrics = self.get_system_metrics()
                logger.info(f"System metrics: CPU: {system_metrics.get('cpu_percent', 'N/A')}%, "
                           f"Memory: {system_metrics.get('memory_percent', 'N/A')}%")
                
                # Log request metrics
                avg_processing_time = 0
                if self.metrics["processing_times"]:
                    avg_processing_time = sum(self.metrics["processing_times"]) / len(self.metrics["processing_times"])
                
                logger.info(f"Request metrics: Total: {self.metrics['requests']}, "
                           f"Errors: {self.metrics['errors']}, "
                           f"Avg processing time: {avg_processing_time:.2f}s")
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
            
            # Sleep until next check
            time.sleep(self.check_interval)
    
    def start(self):
        """Start the monitoring system."""
        if not self.running:
            self.running = True
            self.monitor_thread = threading.Thread(target=self._monitoring_loop)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
            logger.info("Monitoring started")
    
    def stop(self):
        """Stop the monitoring system."""
        self.running = False
        logger.info("Monitoring stopped")

# Example usage
if __name__ == "__main__":
    monitor = AccentDetectorMonitor()
    monitor.start()
    
    try:
        # Keep the script running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        monitor.stop()
        print("Monitoring stopped")
