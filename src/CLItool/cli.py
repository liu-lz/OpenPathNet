"""
CLI entry point for running the scene generation functionality.

Users can define the scene location by a rectangle in two ways:
1) Specifying four GPS corners directly, or
2) Providing one reference point plus width/height in meters and a corner/center position.

scenegenerationpipe \
    --data-dir test123 \
    --bbox -71.06025695800783 42.35128145107633 -71.04841232299806 42.35917815419112
"""

import logging

from argparse import ArgumentParser
from .core import Scene
from .utils import rect_from_point_and_size

try:
    from importlib.metadata import version as pkg_version, PackageNotFoundError
except ImportError:
    # For Python < 3.8, use importlib_metadata backport
    from importlib_metadata import version as pkg_version, PackageNotFoundError

PACKAGE_NAME = "scenegenerationpipe"


def get_package_version() -> str:
    """
    Attempt to retrieve the installed package version from metadata.
    Falls back to a default if the package isn't found (not installed).
    """
    try:
        return pkg_version(PACKAGE_NAME)
    except PackageNotFoundError:
        return "0.0.0.dev (uninstalled)"


def setup_logging(log_file=None):
    """Configure logging; file output is optional (disabled by default)."""
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s - %(name)s - [%(levelname)s] - %(message)s")
    console_formatter = logging.Formatter("[%(levelname)s] %(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)

    file_handler = None
    if log_file:
        file_handler = logging.FileHandler(log_file, mode="w")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)

    if logger.hasHandlers():
        logger.handlers.clear()

    logger.addHandler(console_handler)
    if file_handler:
        logger.addHandler(file_handler)


#!/usr/bin/env python3
"""
CLI entry point for Scene Generation Pipeline.
"""

import sys
from argparse import ArgumentParser, RawTextHelpFormatter


def main():
    """
    Main function to parse arguments and dispatch subcommands.
    """

    parser = ArgumentParser(
        description="Scene Generation CLI.\n\n"
        "You can define the scene location (a rectangle) in two ways:\n"
        "  1) 'bbox' subcommand: specify four GPS corners (min_lon, min_lat, max_lon, max_lat).\n"
        "  2) 'point' subcommand: specify one GPS point, indicate its corner/center position, "
        "and give width/height in meters.\n",
        formatter_class=RawTextHelpFormatter,
    )

    # --version/-v: we'll handle printing version info ourselves after parse_args()
    parser.add_argument(
        "--version",
        "-v",
        action="store_true",
        help="Show version information and exit.",
    )

    # Create a "parent" parser to hold common optional arguments.
    # Use add_help=False so we don’t duplicate the --help in child parsers.
    common_parser = ArgumentParser(add_help=False)
    common_parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose output."
    )
    common_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate actions without executing anything.",
    )
    common_parser.add_argument(
        "--data-dir",
        required=True,
        help="Directory where scene file will be saved.",
    )
    common_parser.add_argument(
        "--osm-server-addr",
        default="https://overpass-api.de/api/interpreter",
        help="OSM server address (optional).",
    )
    common_parser.add_argument(
        "--enable-building-map",
        action="store_true",
        help="Enable 2D building map output.",
    )
    common_parser.add_argument(
        "--debug",
        action="store_true",
        help=(
            "If passed, sets console logging to DEBUG (file logging is always DEBUG). "
            "This overrides the default console level of INFO."
        ),
    )

    # Create subparsers for different subcommands
    subparsers = parser.add_subparsers(
        title="Subcommands", dest="command", help="Available subcommands."
    )

    # Subcommand 'bbox': define a bounding box by four float coordinates
    parser_bbox = subparsers.add_parser(
        "bbox",
        parents=[common_parser],
        help=(
            "Define a bounding box using four GPS coordinates in the order: "
            "min_lon, min_lat, max_lon, max_lat."
        ),
    )
    parser_bbox.add_argument("min_lon", type=float, help="Minimum longitude.")
    parser_bbox.add_argument("min_lat", type=float, help="Minimum latitude.")
    parser_bbox.add_argument("max_lon", type=float, help="Maximum longitude.")
    parser_bbox.add_argument("max_lat", type=float, help="Maximum latitude.")

    # Subcommand 'point': define a reference point and rectangle size
    parser_point = subparsers.add_parser(
        "point",
        parents=[common_parser],
        help="Work with a single point and a rectangle size.",
    )
    parser_point.add_argument("lon", type=float, help="Latitude.")
    parser_point.add_argument("lat", type=float, help="Longitude.")
    parser_point.add_argument(
        "position",
        choices=["top-left", "top-right", "bottom-left", "bottom-right", "center"],
        help="Relative position inside a rectangle.",
    )
    parser_point.add_argument("width", type=float, help="Width in meters.")
    parser_point.add_argument("height", type=float, help="Height in meters.")

    # Parse the full command line
    args = parser.parse_args()

    # Handle --version or no subcommand
    if args.version:
        print(f"{PACKAGE_NAME} version {get_package_version()}")
        sys.exit(0)

    if not args.command:
        # No subcommand provided: show help and exit
        parser.print_help()
        sys.exit(1)

    # -------------------------------------------------------------------------
    # 1) Set up logging by default
    #    - debug.log file captures all logs at DEBUG level
    #    - console sees INFO+ by default
    # -------------------------------------------------------------------------
    setup_logging(log_file=None)

    try:
        import osmnx as ox
        ox.settings.use_cache = False
        ox.settings.cache_folder = None
    except Exception:
        pass

    # If user wants console debug output too, adjust console handler level.
    if args.debug:
        console_logger = logging.getLogger()  # root logger
        for handler in console_logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.setLevel(logging.DEBUG)

    logger = logging.getLogger(__name__)

    # Dispatch subcommands
    if args.command == "bbox":
        min_lon = args.min_lon
        min_lat = args.min_lat
        max_lon = args.max_lon
        max_lat = args.max_lat

        logger.info(
            f"Check the bbox at http://bboxfinder.com/#{min_lat:.{4}f},{min_lon:.{4}f},{max_lat:.{4}f},{max_lon:.{4}f}"
        )
        scene_instance = Scene()
        scene_instance(
            [
                [min_lon, min_lat],
                [min_lon, max_lat],
                [max_lon, max_lat],
                [max_lon, min_lat],
                [min_lon, min_lat],
            ],
            args.data_dir,
            None,
            osm_server_addr=args.osm_server_addr,
            lidar_calibration=False,
            generate_building_map=args.enable_building_map,
        )
    elif args.command == "point":
        polygon_points_gps = rect_from_point_and_size(
            args.lon, args.lat, args.position, args.width, args.height
        )
        min_lon, min_lat = polygon_points_gps[0]
        max_lon, max_lat = polygon_points_gps[2]
        logger.info(
            f"Check the bbox at http://bboxfinder.com/#{min_lat:.{4}f},{min_lon:.{4}f},{max_lat:.{4}f},{max_lon:.{4}f}"
        )
        scene_instance = Scene()
        scene_instance(
            polygon_points_gps,
            args.data_dir,
            None,
            osm_server_addr=args.osm_server_addr,
            lidar_calibration=False,
            generate_building_map=args.enable_building_map,
        )
    else:
        # Should never happen if we covered all subcommands
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()


