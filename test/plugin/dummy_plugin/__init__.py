"""
This is a device backend extension used for testing.
"""

import os


def _autoload():
    os.environ["IS_DUMMY_PLUGIN_LOADED"] = "1"
