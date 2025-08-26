# src/cache.py

import json
import os


class Cache:
    def __init__(self, cache_file: str):
        self.cache_file = cache_file
        if os.path.exists(cache_file):
            with open(cache_file, "r") as f:
                self.cache = json.load(f)
        else:
            self.cache = {}

    def contains(self, key: str) -> bool:
        return key in self.cache

    def get(self, key: str):
        return self.cache.get(key, None)

    def set(self, key: str, value):
        self.cache[key] = value
        self._save()

    def _save(self):
        with open(self.cache_file, "w") as f:
            json.dump(self.cache, f, indent=2)
