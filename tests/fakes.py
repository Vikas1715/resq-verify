"""A minimal in-memory stand-in for pymongo (just the calls MongoStore makes), so tests need no database."""
from __future__ import annotations

import copy


class _Cursor:
    def __init__(self, docs): self.docs = docs
    def sort(self, key, direction=1):
        if isinstance(key, list):
            key, direction = key[0]
        self.docs = sorted(self.docs, key=lambda d: (d.get(key) is None, d.get(key)), reverse=direction == -1)
        return self
    def limit(self, n): self.docs = self.docs[:n]; return self
    def __iter__(self): return iter(self.docs)


def _project(d, proj):
    d = copy.deepcopy(d)
    if proj and proj.get("_id") == 0:
        d.pop("_id", None)
    return d


class FakeCollection:
    def __init__(self): self.docs = []; self.unique = []
    def create_index(self, spec, unique=False, **kw):
        if unique: self.unique.append(spec)
    def _match(self, d, flt): return all(d.get(k) == v for k, v in (flt or {}).items())
    def insert_one(self, doc):
        for u in self.unique:
            if any(d.get(u) == doc.get(u) for d in self.docs):
                raise RuntimeError(f"duplicate key on {u}")
        doc = dict(doc); doc["_id"] = len(self.docs) + 1; self.docs.append(copy.deepcopy(doc))
    def find_one(self, flt=None, proj=None):
        for d in self.docs:
            if self._match(d, flt): return _project(d, proj)
        return None
    def find(self, flt=None, proj=None): return _Cursor([_project(d, proj) for d in self.docs if self._match(d, flt)])
    def replace_one(self, flt, doc):
        for i, d in enumerate(self.docs):
            if self._match(d, flt):
                self.docs[i] = {**copy.deepcopy(doc), "_id": d["_id"]}
                return


class _Admin:
    def command(self, name): return {"ok": 1}


class FakeDB:
    def __init__(self, name): self.name = name; self.cols = {}
    def __getitem__(self, k): return self.cols.setdefault(k, FakeCollection())


class FakeMongoClient:
    def __init__(self): self.dbs = {}; self.admin = _Admin()
    def __getitem__(self, k): return self.dbs.setdefault(k, FakeDB(k))
