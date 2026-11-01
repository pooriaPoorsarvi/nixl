#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import os
import signal
import sys
import time

from nixl import nixl_telemetry_buffer_reader

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Global flag for graceful shutdown
running = True


def signal_handler(signum, _):
    """Signal handler for Ctrl+C"""
    global running
    if signum == signal.SIGINT:
        logger.info("\nReceived Ctrl+C, shutting down...")
        running = False


def format_bytes(bytes_val):
    """Format bytes to human readable format"""
    units = ["B", "KB", "MB", "GB", "TB"]
    unit_index = 0
    value = float(bytes_val)

    while value >= 1024.0 and unit_index < 4:
        value /= 1024.0
        unit_index += 1

    return f"{value:.2f} {units[unit_index]}"


def print_telemetry_event(event):
    """Print telemetry event in a formatted way"""
    logger.info("\n=== NIXL Telemetry Event ===")
    logger.info("Event: %s", event.event_type)
    logger.info("Value: %s", event.value)
    logger.info("===========================")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="NIXL Telemetry Reader")
    parser.add_argument(
        "--telemetry_path", help="Path to the telemetry file", required=True
    )

    args = parser.parse_args()

    logger.info("Telemetry path: %s", args.telemetry_path)
    telemetry_file_name = args.telemetry_path
    if not os.path.exists(telemetry_file_name):
        logger.error("Telemetry file %s does not exist", telemetry_file_name)
        return 1

    signal.signal(signal.SIGINT, signal_handler)

    try:
        logger.info("Opening telemetry buffer: %s", telemetry_file_name)
        logger.info("Press Ctrl+C to stop reading telemetry...")

        buffer = nixl_telemetry_buffer_reader(telemetry_file_name)

        logger.info(
            "Successfully opened telemetry buffer (version: %d)", buffer.version()
        )
        logger.info("Buffer capacity: %d events", buffer.capacity())
        logger.info("Current events in buffer: %d", buffer.size())

        event_count = 0

        while running:
            # Try to read an event from the buffer
            event = buffer.pop()
            if event:
                event_count += 1
                print_telemetry_event(event)
            else:
                # No events available, sleep briefly
                time.sleep(0.5)

        logger.info("\nTotal events read: %d", event_count)
        logger.info("Final buffer size: %d events", buffer.size())

    except Exception as e:
        logger.error("Error: %s", e)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
