#!/bin/bash
# This script installs the Playwright Chromium browser.
# NOTE: This may still fail if Node.js < 14 is installed.
echo "Running install_browsers.sh ..."
npx playwright install chromium
echo "Done installing browsers!"
