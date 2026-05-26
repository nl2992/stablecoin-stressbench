"""Etherscan V2 on-chain data loader.

Fetches ERC-20 token transfer events, large mint/burn events, and gas proxies
for USDC, USDT, and DAI on Ethereum mainnet.

Reference:
    https://docs.etherscan.io/v/etherscan-v2
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import polars as pl
import requests

from stressbench.common.config import bronze_root, get_env, load_token_addresses
from stressbench.common.logging import get_logger

logger = get_logger(__name__)

_ETHERSCAN_BASE = "https://api.etherscan.io/v2/api"
_CHAIN_ID = 1  # Ethereum mainnet
_MAX_RESULTS = 10000
_RATE_LIMIT_SLEEP = 0.25  # seconds between requests (free tier: 5 req/s)


def _api_key() -> str:
    key = get_env("ETHERSCAN_API_KEY", "")
    if not key:
        logger.warning("ETHERSCAN_API_KEY not set; requests may be rate-limited.")
    return key


def _get(params: dict[str, Any]) -> dict[str, Any] | None:
    """Make a GET request to the Etherscan V2 API."""
    params["chainid"] = _CHAIN_ID
    params["apikey"] = _api_key()
    try:
        resp = requests.get(_ETHERSCAN_BASE, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") != "1":
            logger.warning("Etherscan API error: %s", data.get("message"))
            return None
        return data
    except requests.RequestException as exc:
        logger.error("Etherscan request failed: %s", exc)
        return None
    finally:
        time.sleep(_RATE_LIMIT_SLEEP)


def fetch_token_transfers(
    token_symbol: str,
    start_block: int,
    end_block: int,
    page: int = 1,
    offset: int = _MAX_RESULTS,
) -> list[dict[str, Any]]:
    """Fetch ERC-20 transfer events for a stablecoin.

    Args:
        token_symbol: One of ``"USDC"``, ``"USDT"``, ``"DAI"``.
        start_block: Starting Ethereum block number.
        end_block: Ending Ethereum block number.
        page: Page number for pagination.
        offset: Number of results per page (max 10000).

    Returns:
        List of transfer event dicts from Etherscan.
    """
    addresses = load_token_addresses()
    token_info = addresses.get(token_symbol)
    if not token_info:
        logger.error("Unknown token symbol: %s", token_symbol)
        return []

    contract_address = token_info.get("ethereum", {}).get("address")
    if not contract_address:
        logger.error("No Ethereum address for %s", token_symbol)
        return []

    params = {
        "module": "account",
        "action": "tokentx",
        "contractaddress": contract_address,
        "startblock": start_block,
        "endblock": end_block,
        "page": page,
        "offset": offset,
        "sort": "asc",
    }
    data = _get(params)
    if data is None:
        return []
    return data.get("result", [])


def fetch_block_by_timestamp(timestamp_utc: int) -> int | None:
    """Return the closest Ethereum block number for a Unix timestamp.

    Args:
        timestamp_utc: Unix timestamp in seconds.

    Returns:
        Block number, or ``None`` on failure.
    """
    params = {
        "module": "block",
        "action": "getblocknobytime",
        "timestamp": timestamp_utc,
        "closest": "before",
    }
    data = _get(params)
    if data is None:
        return None
    try:
        return int(data["result"])
    except (KeyError, ValueError):
        return None


def save_transfers_to_bronze(
    transfers: list[dict[str, Any]],
    token_symbol: str,
    date: str,
    root: Path | None = None,
) -> list[Path]:
    """Save a list of transfer events to canonical Bronze Parquet.

    Writes to the same Hive-partitioned layout used by live collectors::

        venue=etherscan/
          channel=transfer/
            symbol=<TOKEN>/
              date=YYYY-MM-DD/
                hour=HH/
                  part-0.parquet

    The Silver builder's ``is_etherscan`` path reads from
    ``venue=etherscan / channel=transfer`` at ``hour=*`` granularity.

    Args:
        transfers: List of Etherscan transfer event dicts.
        token_symbol: Token symbol for Hive partitioning (e.g. ``"USDC"``).
        date: ISO date string ``YYYY-MM-DD`` for partitioning.
        root: Bronze root override.

    Returns:
        List of paths to written Parquet files (one per hour bucket).
    """
    import json
    from datetime import datetime, timezone

    if not transfers:
        return []

    root = root or bronze_root()

    # Bucket transfers by hour using the Etherscan ``timeStamp`` field (Unix s)
    hour_buckets: dict[int, list[dict[str, Any]]] = {}
    for tx in transfers:
        try:
            ts_s = int(tx.get("timeStamp", 0))
            hour = datetime.fromtimestamp(ts_s, tz=timezone.utc).hour
        except (ValueError, TypeError):
            hour = 0
        hour_buckets.setdefault(hour, []).append(tx)

    written_paths: list[Path] = []
    for hour, hour_transfers in sorted(hour_buckets.items()):
        out_dir = (
            root
            / "venue=etherscan"
            / "channel=transfer"
            / f"symbol={token_symbol}"
            / f"date={date}"
            / f"hour={hour:02d}"
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "part-0.parquet"

        # Wrap each transfer event as a Bronze-schema record so the Silver
        # normalizer receives a consistent payload column
        bronze_rows = []
        for tx in hour_transfers:
            try:
                ts_s = int(tx.get("timeStamp", 0))
                ts_ns = ts_s * 1_000_000_000
            except (ValueError, TypeError):
                ts_ns = 0
            bronze_rows.append({
                "source": "etherscan",
                "channel": "transfer",
                "symbol": token_symbol,
                "ts_exchange": tx.get("timeStamp", ""),
                "ts_receive_ns": ts_ns,
                "payload": json.dumps(tx, sort_keys=True),
                "payload_hash": "",
                "schema_version": "raw.v1",
                "ingest_batch_id": "etherscan_api",
            })

        pl.DataFrame(bronze_rows).write_parquet(out_file)
        written_paths.append(out_file)
        logger.info(
            "Saved %d transfers to %s", len(hour_transfers), out_file
        )

    return written_paths
