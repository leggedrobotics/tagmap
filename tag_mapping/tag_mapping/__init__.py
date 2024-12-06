from .tag_map import TagMap, TagMapEntry

from .pose_graph import PoseGraph

import os

TAG_MAPPING_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
"""Absolute path to the tag mapping root dir."""

TAG_MAPPING_CONFIG_DIR = os.path.join(TAG_MAPPING_ROOT_DIR, 'config')