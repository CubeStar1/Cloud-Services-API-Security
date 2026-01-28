import asyncio
import os
import json
import logging
import subprocess
import threading
import sys
from pathlib import Path
from typing import Optional, AsyncGenerator

# Configure logging
logger = logging.getLogger(__name__)

class AnyProxyManager:
    _instance = None
    _process: Optional[subprocess.Popen] = None
    _current_log_file: Optional[str] = None
    _listeners: set = set()
    _monitor_thread: Optional[threading.Thread] = None
    _stop_event: threading.Event = threading.Event()
    _loop: Optional[asyncio.AbstractEventLoop] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AnyProxyManager, cls).__new__(cls)
        return cls._instance

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    @property
    def current_log_file(self) -> Optional[str]:
        return self._current_log_file

    async def start(self, filename: str) -> None:
        if self.is_running:
            raise RuntimeError("AnyProxy is already running")

        # Capture the running loop to use for thread-safe callbacks
        self._loop = asyncio.get_running_loop()

        if not filename.endswith('.json'):
            filename += '.json'
        
        base_dir = Path.cwd()
        # rule path is now backend/anyproxy/rule.js relative to project root (casb)
        # or relative to backend/utils/anyproxy_manager.py if resolving via file
        
        # Determine rule path robustly
        # Option 1: Relative to casb/ (cwd)
        rule_path = base_dir / "backend" / "anyproxy" / "rule.js"
        
        if not rule_path.exists():
            # Option 2: Maybe cwd is backend/ ?
            rule_path = base_dir / "anyproxy" / "rule.js"
        
        if not rule_path.exists():
             raise FileNotFoundError(f"Rule file not found. Searched at {rule_path} and associated paths.")

        logger.info(f"Starting AnyProxy with rule: {rule_path}, logging to: {filename}")

        env = os.environ.copy()
        env["LOG_FILE_NAME"] = filename
        
        # Construct command
        # On Windows, using shell=True with npx is often easier, or npx.cmd
        npx_cmd = "npx.cmd" if sys.platform == "win32" else "npx"
        
        cmd = [
            npx_cmd,
            "anyproxy",
            "--port", "8001",
            "--rule", str(rule_path),
            "--intercept"
        ]
        
        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                shell=(sys.platform == "win32"), # shell=True helps resolve npx on windows sometimes
                text=True, # Auto-decode bytes to string
                bufsize=1  # Line buffered
            )
            self._current_log_file = filename
            self._stop_event.clear()
            
            # Start monitor thread
            self._monitor_thread = threading.Thread(target=self._monitor_output)
            self._monitor_thread.daemon = True
            self._monitor_thread.start()
            
        except Exception as e:
            logger.error(f"Failed to start process: {e}")
            if self._process:
                self._process.kill()
            self._process = None
            raise e

    async def stop(self) -> None:
        if not self.is_running:
            return

        self._stop_event.set()
        try:
            # Graceful termination
            self._process.terminate()
            # Give it a moment (async sleep to not block loop)
            for _ in range(10):
                if self._process.poll() is not None:
                    break
                await asyncio.sleep(0.5)
            
            if self._process.poll() is None:
                self._process.kill()
                
        except Exception as e:
            logger.error(f"Error stopping process: {e}")
        finally:
            self._process = None
            self._current_log_file = None

    def _monitor_output(self):
        """Runs in a separate thread to read stdout."""
        if not self._process or not self._process.stdout:
            return

        try:
            # Iterate over stdout lines
            for line in self._process.stdout:
                if self._stop_event.is_set():
                    break
                
                line = line.strip()
                if not line:
                    continue
                
                # Check for JSON structure
                if line.startswith('{') and line.endswith('}'):
                    try:
                        # Quick validation it's proper JSON before broadcasting
                        # This optional, but good to filter noise
                        # json.loads(line) 
                        self._broadcast_threadsafe(line)
                    except:
                        pass
        except Exception as e:
            logger.error(f"Monitor thread error: {e}")

    def _broadcast_threadsafe(self, log_data: str):
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._dispatch_to_listeners, log_data)

    def _dispatch_to_listeners(self, log_data: str):
        # This runs in the async loop
        for queue in list(self._listeners):
            try:
                queue.put_nowait(log_data)
            except Exception:
                pass

    async def stream_logs(self) -> AsyncGenerator[str, None]:
        queue = asyncio.Queue()
        self._listeners.add(queue)
        try:
            while True:
                # Wait for new log
                data = await queue.get()
                yield f"data: {data}\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            self._listeners.remove(queue)

anyproxy_manager = AnyProxyManager()
