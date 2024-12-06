import os
import pickle
import numpy as np
import uuid

from dataclasses import dataclass
from typing import Dict, Any


@dataclass(frozen=True)
class TagMapEntry:
    pose: np.ndarray
    uuid: uuid.UUID
    extras: Dict[str, Any] = None


@dataclass(frozen=True)
class TagDBEntry:
    entry_uuid: uuid.UUID
    extras: Dict[str, Any] = None


class TagMap:
    def __init__(self, metadata: Dict[str, Any]):
        self._metadata = metadata
        self._tags_db = {}
        self._entry_db = {}

    def add_entry(self, entry: TagMapEntry):
        """
        Add entry (i.e. observed frame) to the tag map.

        NOTE: This method raises a ValueError if the entry's uuid is
        already in the entry database.
        """
        if entry.uuid in self._entry_db:
            raise ValueError("uuid already in the entry database")
        self._entry_db[entry.uuid] = entry

    def add_tag(self, tag: str, entry_uuid: uuid.UUID, extras: Dict[str, Any] = None):
        """
        Associates a tag with an entry in the tag map.

        Args:
            tag: The tag to associate with the entry.
            entry_uuid: The uuid of the entry to associate with the tag.
            extras: Any extra data to associate with the tag.
        """
        if entry_uuid not in self._entry_db:
            raise ValueError("uuid not in the entry database")

        tag_db_entry = TagDBEntry(entry_uuid, extras)
        if tag not in self._tags_db:
            self._tags_db[tag] = [tag_db_entry]
        else:
            self._tags_db[tag].append(tag_db_entry)

    def add_extra(self, extra_name: str, extra_data: Any, overwrite: bool = False):
        """
        Add extra data to the tag map (e.g. pose graph).

        Args:
            extra_name: The name of the extra data.
            extra_data: The extra data.
            overwrite: Whether to overwrite the extra data if it already exists.
        """
        if not hasattr(self, "_extras"):
            self._extras = {extra_name: extra_data}
        else:
            if extra_name in self._extras and not overwrite:
                raise ValueError(
                    "Extra {} already stored in tag map, set overwrite=True to overwrite".format(
                        extra_name
                    )
                )
            else:
                self._extras[extra_name] = extra_data

    def query(self, tag: str, return_uuids: bool = False):
        """
        Query the tag map for all entries associated with a tag.

        Returns:
            A list of TagMapEntry objects associated with the tag
                or None if the tag is not in the tag map.
        """
        if tag not in self._tags_db:
            print("{} not in the tag map".format(tag))
            return None

        entry_uuids = [e.entry_uuid for e in self._tags_db[tag]]
        entries = [self._entry_db[id] for id in entry_uuids]
        tag_extras = [e.extras for e in self._tags_db[tag]]

        # pack tag extras into entry.extras
        for entry, tag_extra in zip(entries, tag_extras):
            if entry.extras is None:
                entry.extras = {}

            if tag_extra is not None:
                for key, value in tag_extra.items():
                    entry.extras[key] = value

        if return_uuids:
            return entries, entry_uuids
        else:
            return entries

    def save(self, save_path):
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(save_path, "wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, save_path):
        with open(save_path, "rb") as file:
            obj = pickle.load(file)
        if isinstance(obj, cls):
            return obj
        else:
            raise ValueError("Loaded object is not an instance of TagMap")

    @property
    def metadata(self):
        return self._metadata

    @property
    def unique_objects(self):
        return self._tags_db.keys()

    @property
    def num_entries(self):
        return len(self._entry_db)

    @property
    def num_tags(self):
        return len(self._tags_db)

    @property
    def extras(self):
        return self._extras
    
    def __contains__(self, tag: str):
        return tag in self.unique_objects
