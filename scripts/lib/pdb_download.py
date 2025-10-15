from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import requests
import threading
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter


_tls = threading.local()


def _get_session() -> requests.Session:
    sess = getattr(_tls, "session", None)
    if sess is None:
        sess = requests.Session()
        sess.headers.update({"User-Agent": "LDB-collector/1.0"})
        retry = Retry(
            total=5,
            connect=5,
            read=5,
            status=5,
            backoff_factor=0.5,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(["GET", "POST"]),
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=16, pool_maxsize=16)
        sess.mount("https://", adapter)
        sess.mount("http://", adapter)
        _tls.session = sess
    return sess


RCSB_CORE_URL = "https://data.rcsb.org/rest/v1/core"
RCSB_DOWNLOAD_URL = "https://files.rcsb.org/download"


def get_json(path: str, timeout: int = 30) -> dict:
    url = f"{RCSB_CORE_URL}/{path.lstrip('/')}"
    resp = _get_session().get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def fetch_entry_json(pdbid: str) -> dict:
    return get_json(f"/entry/{pdbid}")


def fetch_assembly_json(pdbid: str, assembly_id: str) -> dict:
    return get_json(f"/assembly/{pdbid}/{assembly_id}")


def fetch_polymer_entity_json(pdbid: str, entity_id: str | int) -> dict:
    return get_json(f"/polymer_entity/{pdbid}/{entity_id}")


def fetch_nonpolymer_entity_json(pdbid: str, entity_id: str | int) -> dict:
    return get_json(f"/nonpolymer_entity/{pdbid}/{entity_id}")


def fetch_chemcomp_json(comp_id: str) -> dict:
    return get_json(f"/chemcomp/{comp_id}")


def download_mmcif_assembly(pdbid: str, assembly_id: str, out_path: Path, timeout: int = 60) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"{RCSB_DOWNLOAD_URL}/{pdbid}.cif?assembly_id={assembly_id}"
    resp = _get_session().get(url, timeout=timeout)
    resp.raise_for_status()
    out_path.write_bytes(resp.content)
    return out_path


def download_mmcif_asym_unit(pdbid: str, out_path: Path, timeout: int = 60) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"{RCSB_DOWNLOAD_URL}/{pdbid}.cif"
    resp = _get_session().get(url, timeout=timeout)
    resp.raise_for_status()
    out_path.write_bytes(resp.content)
    return out_path
