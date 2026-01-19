#!/usr/bin/env python3
"""
OSM Power Infrastructure Fetcher & Converter

Fetches power infrastructure data (nodes and ways) from Overpass API and
converts directly to GeoJSON without intermediate disk writes.
Optimized for GitHub Actions free tier.

Features:
- Multi-endpoint retry with failover
- Quadtree spatial tiling for large queries that timeout
- Pre-emptive tiling for known-large tags (skips wasteful global attempts)
- Shorter timeouts for faster failover
- Deduplication of elements crossing tile boundaries
- Memory-efficient processing

Changes from v1:
- Removed osm.jp endpoint (always 403)
- Added PRE_TILE_TAGS to skip global query for line/minor_line/substation
- Reduced WAY_TIMEOUT from 3600s to 240s (below Azure NAT ~280s limit)
- Added elapsed time tracking for diagnostics
- Endpoints: private.coffee (quality) → mail.ru (reliable) → overpass-api.de (fast-fail)
"""

import os
import json
import requests
from datetime import datetime, timezone
from typing import Any
from collections import defaultdict
from osm2geojson import json2geojson

# === CONFIG ===============================================================

OUTPUT_DIR = "downloads"

# Node-primary tags (no transformation needed)
NODE_POWER_TAGS = [
    "transformer",
    "switch",
    "terminal",
    "converter",
    "connection",
    "transition",
    "compensator",
    "inverter",
    "cable_distribution",
    "cable_distribution_cabinet",
]

# Way-primary tags (transformation needed)
WAY_POWER_TAGS = [
    "line",
    "minor_line",
    "cable",
    "switchgear",
    "substation",
]

# Tags that are known to timeout on global queries - skip straight to tiling
# Based on empirical testing: these always fail globally and waste ~10 min each
PRE_TILE_TAGS = {"line", "minor_line", "substation"}

# Endpoint order: quality first, reliable fallback, fast-fail last
# Removed osm.jp - always returns 403
OVERPASS_ENDPOINTS = [
    "https://overpass.private.coffee/api/interpreter",  # Best data quality
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",  # Most reliable for large queries
    "https://overpass-api.de/api/interpreter",  # Fast 504 on overload (good for quick failover)
]

# Timeouts: (connect_timeout, read_timeout)
#
# CRITICAL: GitHub Actions runs on Azure infrastructure with ~280s idle connection timeout.
# Overpass processes queries entirely server-side before sending ANY response data,
# meaning the connection sits idle during processing. If processing exceeds ~280s,
# Azure NAT kills the connection with "RemoteDisconnected" before we receive data.
#
# Strategy:
# - Set timeout to 240s (below 280s NAT limit, but allows legitimate queries time)
# - Queries completing in <240s succeed
# - Queries that would take >280s fail faster and trigger tiling sooner
# - The 40s buffer (240→280) accounts for network variance
#
# From home IPs this limit doesn't exist, but we optimize for GHA where it matters.
NODE_TIMEOUT = (30, 300)   # 5 min read for smaller node queries (rarely hit limits)
WAY_TIMEOUT = (30, 240)    # 4 min read for way queries (below Azure NAT ~280s limit)

# Quadtree tiling config
WORLD_BBOX = (-90, -180, 90, 180)  # (south, west, north, east)
MAX_TILE_DEPTH = 3  # Max recursion: 1→4→16→64 tiles

# === UTILITIES ============================================================


def log(msg: str) -> None:
    """Print a timestamped log message."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{now}] {msg}", flush=True)


def split_bbox(bbox: tuple[float, float, float, float]) -> list[tuple[float, float, float, float]]:
    """Split bbox into 4 quadrants (SW, SE, NW, NE)."""
    south, west, north, east = bbox
    mid_lat = (south + north) / 2
    mid_lon = (west + east) / 2
    return [
        (south, west, mid_lat, mid_lon),      # SW
        (south, mid_lon, mid_lat, east),      # SE
        (mid_lat, west, north, mid_lon),      # NW
        (mid_lat, mid_lon, north, east),      # NE
    ]


def bbox_to_str(bbox: tuple[float, float, float, float]) -> str:
    """Format bbox for logging."""
    return f"({bbox[0]:.1f},{bbox[1]:.1f},{bbox[2]:.1f},{bbox[3]:.1f})"


def is_retriable_error(error_msg: str) -> bool:
    """Check if error should trigger retry/tiling."""
    error_lower = error_msg.lower()
    retriable_patterns = [
        "remotedisconnected",
        "connection aborted",
        "connection reset",
        "timeout",
        "timed out",
        "expecting value",  # Empty JSON response = server overloaded
    ]
    return any(pattern in error_lower for pattern in retriable_patterns)


def deduplicate_elements(elements: list[dict], verbose: bool = False) -> list[dict]:
    """
    Remove duplicate elements by (type, id).
    Ways crossing tile boundaries may appear in multiple tiles.
    """
    seen = {}
    duplicate_count = 0

    for elem in elements:
        key = (elem.get("type"), elem.get("id"))
        if key in seen:
            duplicate_count += 1
        else:
            seen[key] = elem

    if verbose and duplicate_count > 0:
        log(f"    🔄 Deduplicated: removed {duplicate_count} duplicates, {len(seen):,} unique")

    return list(seen.values())


# === QUERY BUILDERS =======================================================


def build_node_query(tag: str, timeout: int = 300) -> str:
    """Build Overpass QL query for node elements."""
    return f'[out:json][timeout:{timeout}];node["power"="{tag}"];out meta;'


def build_way_query(tag: str, bbox: tuple[float, float, float, float] | None = None, timeout: int = 180) -> str:
    """Build Overpass QL query for way elements."""
    bbox_str = f"({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]})" if bbox else ""
    return f'[out:json][timeout:{timeout}];way["power"="{tag}"]{bbox_str};out geom;'


# === FETCH LOGIC ==========================================================


def fetch_single_endpoint(
    query: str,
    endpoint: str,
    timeout: tuple[int, int],
) -> tuple[list[dict] | None, str | None, bool]:
    """
    Fetch from a single endpoint.

    Returns: (elements, error_message, is_retriable)
    - elements: list of OSM elements if successful
    - error_message: error description if failed
    - is_retriable: True if failure should trigger tiling
    """
    import time
    start_time = time.time()

    try:
        response = requests.post(
            endpoint,
            data={"data": query},
            timeout=timeout,
        )
        elapsed = time.time() - start_time

        # HTTP error codes
        if response.status_code == 429:
            return None, f"HTTP 429 (rate limited) [{elapsed:.1f}s]", True

        if response.status_code >= 500:
            return None, f"HTTP {response.status_code} (server error) [{elapsed:.1f}s]", True

        if response.status_code != 200:
            return None, f"HTTP {response.status_code} [{elapsed:.1f}s]", False

        # Parse JSON
        try:
            data = response.json()
        except json.JSONDecodeError as e:
            # Empty response typically means server timeout/overload
            return None, f"JSON decode error [{elapsed:.1f}s]: {e}", True

        elements = data.get("elements", [])
        return elements, f"[{elapsed:.1f}s]", False  # Include timing in "error" field for logging

    except requests.exceptions.Timeout as e:
        elapsed = time.time() - start_time
        return None, f"Timeout [{elapsed:.1f}s]: {e}", True

    except requests.exceptions.ConnectionError as e:
        elapsed = time.time() - start_time
        error_str = str(e)
        return None, f"Connection error [{elapsed:.1f}s]: {e}", is_retriable_error(error_str)

    except requests.exceptions.RequestException as e:
        elapsed = time.time() - start_time
        return None, f"Request error [{elapsed:.1f}s]: {e}", False


def fetch_with_retry(query: str, label: str, timeout: tuple[int, int]) -> tuple[list[dict] | None, bool]:
    """
    Try all endpoints in sequence.

    Returns: (elements, should_tile)
    - elements: list of OSM elements if any endpoint succeeded
    - should_tile: True if all failures were retriable (should attempt tiling)
    """
    any_retriable = False

    for endpoint in OVERPASS_ENDPOINTS:
        log(f"    🔗 Trying {endpoint}...")

        elements, error_or_timing, is_retriable = fetch_single_endpoint(query, endpoint, timeout)

        if elements is not None:
            # Success - error_or_timing contains timing info on success
            log(f"      ✅ {len(elements):,} elements {error_or_timing}")
            return elements, False

        # Log the error
        symbol = "⚡" if is_retriable else "❌"
        log(f"      {symbol} {error_or_timing}")

        if is_retriable:
            any_retriable = True

    log(f"    ❌ All endpoints failed for '{label}'")
    return None, any_retriable


# === NODE FETCHING (simple, no tiling needed) =============================


def fetch_nodes(tag: str) -> list[dict] | None:
    """Fetch node data with multi-endpoint retry."""
    query = build_node_query(tag, timeout=NODE_TIMEOUT[1])
    elements, _ = fetch_with_retry(query, f"node:{tag}", NODE_TIMEOUT)
    return elements


# === WAY FETCHING (with tiling support) ===================================


def fetch_ways_with_tiling(
    tag: str,
    bbox: tuple[float, float, float, float] = WORLD_BBOX,
    depth: int = 0,
) -> list[dict] | None:
    """
    Fetch way data with automatic quadtree tiling on failure.

    On retriable failures (timeout, disconnect, 429, 5xx):
    - Split bbox into 4 quadrants
    - Recursively fetch each quadrant
    - Merge and deduplicate results

    For PRE_TILE_TAGS: Skip global query entirely to save time.

    Returns: list of OSM elements, or None if unrecoverable failure
    """
    indent = "  " * depth
    label = f"way:{tag}"

    # Check if we should skip global query (pre-emptive tiling)
    should_skip_global = (depth == 0) and (tag in PRE_TILE_TAGS)

    if should_skip_global:
        log(f"{indent}⏭️ PRE-EMPTIVE TILING: Skipping global query for '{tag}' (known to timeout)")
        log(f"{indent}  🔀 Splitting into 4 quadrants (depth 0 → 1)...")
        should_tile = True
        elements = None
    else:
        # Normal fetch attempt
        if depth == 0:
            # Global query (no bbox)
            query = build_way_query(tag, None, timeout=WAY_TIMEOUT[1])
        else:
            log(f"{indent}📦 Tile depth={depth} bbox={bbox_to_str(bbox)}")
            query = build_way_query(tag, bbox, timeout=WAY_TIMEOUT[1])

        # Try all endpoints
        elements, should_tile = fetch_with_retry(query, label, WAY_TIMEOUT)

    if elements is not None:
        return elements

    # Check if we should attempt tiling
    if not should_tile and not should_skip_global:
        log(f"{indent}  ❌ Non-retriable failure, cannot tile")
        return None

    if depth >= MAX_TILE_DEPTH:
        log(f"{indent}  ❌ Max tile depth ({MAX_TILE_DEPTH}) reached")
        return None

    # Quadtree subdivision
    if not should_skip_global:
        log(f"{indent}  🔀 Splitting into 4 quadrants (depth {depth} → {depth + 1})...")

    quadrants = split_bbox(bbox)
    quad_names = ["SW", "SE", "NW", "NE"]
    all_elements = []

    for i, quad_bbox in enumerate(quadrants):
        log(f"{indent}  [{quad_names[i]}] ───────────────────")

        quad_elements = fetch_ways_with_tiling(tag, quad_bbox, depth + 1)

        if quad_elements is None:
            log(f"{indent}  [{quad_names[i]}] ❌ Failed - aborting")
            return None

        log(f"{indent}  [{quad_names[i]}] ✅ {len(quad_elements):,} elements")
        all_elements.extend(quad_elements)

    # Deduplicate elements crossing tile boundaries
    before_count = len(all_elements)
    all_elements = deduplicate_elements(all_elements, verbose=True)
    after_count = len(all_elements)

    if before_count == after_count:
        log(f"{indent}  📊 Merged: {after_count:,} elements (no duplicates)")
    # else: deduplicate_elements already logged

    return all_elements


# === CONVERTERS ===========================================================


def convert_to_geojson(osm_data: dict[str, Any]) -> dict[str, Any]:
    """Convert raw OSM JSON to GeoJSON using osm2geojson."""
    return json2geojson(osm_data)


def transform_geojson(geojson: dict[str, Any]) -> dict[str, Any]:
    """
    Clean up GeoJSON properties (for ways only):
      - Remove 'nodes' key (internal OSM reference)
      - Remove 'type' key (redundant with geometry type)
    """
    features = []
    for feature in geojson.get("features", []):
        props = feature.get("properties", {}).copy()
        props.pop("nodes", None)
        props.pop("type", None)

        features.append({
            "type": "Feature",
            "properties": props,
            "geometry": feature.get("geometry"),
        })

    return {"type": "FeatureCollection", "features": features}


def save_geojson(geojson: dict[str, Any], filepath: str) -> None:
    """Write GeoJSON to disk with compact formatting."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(geojson, f, ensure_ascii=False, separators=(",", ":"))


# === PROCESSORS ===========================================================


def process_node_tag(tag: str) -> bool:
    """
    Pipeline for a node power tag (3 steps):
      1. Fetch from Overpass API
      2. Convert to GeoJSON
      3. Save to disk

    Returns True on success, False on failure.
    """
    label = f"node:{tag}"
    output_file = os.path.join(OUTPUT_DIR, f"osm_power_node_{tag}.geojson")

    log(f"{'=' * 60}")
    log(f"Processing '{label}'")
    log(f"{'=' * 60}")

    # Step 1: Fetch
    log(f"[1/3] Fetching '{label}' from Overpass API...")
    elements = fetch_nodes(tag)

    if elements is None:
        log(f"  ❌ Failed to fetch '{label}'\n")
        return False

    if not elements:
        log(f"  ⚠️ No elements found for '{label}'\n")
        return False

    # Wrap in OSM JSON structure
    osm_data = {"version": 0.6, "elements": elements}
    del elements

    # Step 2: Convert
    log(f"[2/3] Converting to GeoJSON...")
    geojson = convert_to_geojson(osm_data)
    del osm_data

    # Step 3: Save
    feature_count = len(geojson.get("features", []))
    log(f"[3/3] Saving {feature_count:,} features → {output_file}")
    save_geojson(geojson, output_file)
    del geojson

    log(f"  ✅ Done\n")
    return True


def process_way_tag(tag: str) -> bool:
    """
    Pipeline for a way power tag (4 steps):
      1. Fetch from Overpass API (with tiling fallback)
      2. Convert to GeoJSON
      3. Transform (clean properties)
      4. Save to disk

    Returns True on success, False on failure.
    """
    label = f"way:{tag}"
    output_file = os.path.join(OUTPUT_DIR, f"osm_power_way_{tag}.geojson")

    log(f"{'=' * 60}")
    log(f"Processing '{label}'")
    log(f"{'=' * 60}")

    # Step 1: Fetch (with tiling fallback)
    log(f"[1/4] Fetching '{label}' from Overpass API...")
    elements = fetch_ways_with_tiling(tag)

    if elements is None:
        log(f"  ❌ Failed to fetch '{label}' (all endpoints and tiling failed)\n")
        return False

    if not elements:
        log(f"  ⚠️ No elements found for '{label}'\n")
        return False

    log(f"  📊 Total: {len(elements):,} elements")

    # Wrap in OSM JSON structure
    osm_data = {"version": 0.6, "elements": elements}
    del elements

    # Step 2: Convert
    log(f"[2/4] Converting to GeoJSON...")
    geojson = convert_to_geojson(osm_data)
    del osm_data

    # Step 3: Transform
    log(f"[3/4] Transforming (cleaning properties)...")
    geojson = transform_geojson(geojson)

    # Step 4: Save
    feature_count = len(geojson.get("features", []))
    log(f"[4/4] Saving {feature_count:,} features → {output_file}")
    save_geojson(geojson, output_file)
    del geojson

    log(f"  ✅ Done\n")
    return True


# === MAIN =================================================================


def main() -> None:
    import time
    total_start = time.time()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    total_tags = len(NODE_POWER_TAGS) + len(WAY_POWER_TAGS)
    log("Starting OSM Power Infrastructure Pipeline")
    log(f"Node tags: {NODE_POWER_TAGS}")
    log(f"Way tags: {WAY_POWER_TAGS}")
    log(f"Pre-tile tags (skip global): {PRE_TILE_TAGS}")
    log(f"Total: {total_tags} tags to process")
    log(f"Max tile depth: {MAX_TILE_DEPTH}")
    log(f"Node timeout: {NODE_TIMEOUT[1]}s | Way timeout: {WAY_TIMEOUT[1]}s")
    log(f"Endpoints ({len(OVERPASS_ENDPOINTS)}):")
    for ep in OVERPASS_ENDPOINTS:
        log(f"  - {ep}")
    log("")

    results = {"success": [], "failed": []}

    # Process node-primary tags first (smaller datasets, no tiling needed)
    log("=" * 60)
    log("PHASE 1: NODE-PRIMARY TAGS")
    log("=" * 60 + "\n")

    for tag in NODE_POWER_TAGS:
        if process_node_tag(tag):
            results["success"].append(f"node:{tag}")
        else:
            results["failed"].append(f"node:{tag}")

    # Process way-primary tags (larger datasets, may need tiling)
    log("=" * 60)
    log("PHASE 2: WAY-PRIMARY TAGS")
    log("=" * 60 + "\n")

    for tag in WAY_POWER_TAGS:
        if process_way_tag(tag):
            results["success"].append(f"way:{tag}")
        else:
            results["failed"].append(f"way:{tag}")

    # Summary
    total_elapsed = time.time() - total_start
    log("=" * 60)
    log("SUMMARY")
    log("=" * 60)
    log(f"✅ Success: {len(results['success'])}/{total_tags}")
    if results["failed"]:
        log(f"❌ Failed: {results['failed']}")
    else:
        log("🎉 All tags processed successfully!")
    log(f"⏱️ Total elapsed: {total_elapsed/60:.1f} minutes ({total_elapsed/3600:.2f} hours)")


if __name__ == "__main__":
    main()
