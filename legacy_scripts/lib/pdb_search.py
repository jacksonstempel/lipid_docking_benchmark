from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable, Iterator, List, Optional

import requests
import threading
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter


RCSB_SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"


@dataclass(frozen=True)
class SearchConfig:
    method: str = "X-RAY DIFFRACTION"
    res_max: float = 3.0
    page_size: int = 1000


def _base_query(cfg: SearchConfig) -> dict:
    # RCSB Search API JSON query for entries with X-ray method, resolution cutoff,
    # and at least one nonpolymer entity (to later check lipid status).
    # Keyword-based prefilter to narrow to lipid-relevant entries before local filtering.
    lipid_kw_nodes = [
        {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "struct_keywords.text",
                "operator": "contains_words",
                "value": kw,
            },
        }
        for kw in [
            "lipid",
            "fatty acid",
            "sterol",
            "cholesterol",
            "phospholipid",
            "sphingolipid",
            "ceramide",
            "diacylglycerol",
            "triacylglycerol",
            "phosphatidyl",
        ]
    ]

    q = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "exptl.method",
                        "operator": "exact_match",
                        "negation": False,
                        "value": cfg.method,
                    },
                },
                {
                    # ensure at least one nonpolymer present
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.nonpolymer_entity_count",
                        "operator": "greater",
                        "negation": False,
                        "value": 0,
                    },
                },
                {
                    "type": "group",
                    "logical_operator": "or",
                    "nodes": lipid_kw_nodes,
                },
            ],
        },
        "return_type": "entry",
        "request_options": {
            "results_verbosity": "minimal",
            "paginate": {"start": 0, "rows": cfg.page_size},
        },
    }
    return q


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


def search_candidates(cfg: Optional[SearchConfig] = None) -> List[str]:
    """Return a list of entry IDs meeting broad criteria.

    This does not ensure monomeric assembly or single-lipid-only. Those
    are enforced downstream.
    """
    cfg = cfg or SearchConfig()
    query = _base_query(cfg)
    ids: list[str] = []
    start = 0
    while True:
        query["request_options"]["paginate"]["start"] = start
        resp = _get_session().post(RCSB_SEARCH_URL, json=query, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("result_set", [])
        if not results:
            break
        ids.extend([r["identifier"] for r in results if "identifier" in r])
        if len(results) < cfg.page_size:
            break
        start += cfg.page_size
    # De-duplicate while preserving order
    seen = set()
    out: list[str] = []
    for i in ids:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out
