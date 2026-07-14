import logging

logger = logging.getLogger("ThinkLM.Config")

class ConfigManager:
    """
    ConfigManager manages system settings and configuration flags for ThinkLM.
    
    Responsibilities:
    - Maintains the runtime mode (Normal Mode vs. Plan Mode).
    - Manages configuration flags like pending_plan_mode.
    """
    def __init__(self):
        self._mode = "Normal Mode"
        self.pending_plan_mode = False
        logger.info("ConfigManager initialized. Default mode: 'Normal Mode'.")

    @property
    def mode(self) -> str:
        return self._mode

    @mode.setter
    def mode(self, value: str):
        if value not in ["Normal Mode", "Plan Mode"]:
            raise ValueError("Invalid mode. Must be 'Normal Mode' or 'Plan Mode'.")
        self._mode = value
        self.pending_plan_mode = (value == "Plan Mode")
        logger.info(f"System mode updated to: {self._mode} (pending_plan_mode={self.pending_plan_mode})")

    def toggle_mode(self) -> str:
        """Toggles the system mode between Normal Mode and Plan Mode."""
        if self._mode == "Normal Mode":
            self.mode = "Plan Mode"
        else:
            self.mode = "Normal Mode"
        return self._mode
