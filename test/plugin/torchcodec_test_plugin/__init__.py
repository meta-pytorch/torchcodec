"""
This is a device backend extension used for testing.
"""

import os


def _autoload():
    # Set the environment variable to true in this entrypoint
    os.environ["IS_CUSTOM_DEVICE_BACKEND_IMPORTED"] = "1"
