#!/bin/bash
# This script installs the Playwright Chromium browser.
# NOTE: This may still fail if Node.js < 14 is installed.

NODE_VERSION=$(node -v)
NODE_MAJOR_VERSION=$(echo $NODE_VERSION | cut -d'.' -f1 | sed 's/[^0-9]*//g')

if [ "$NODE_MAJOR_VERSION" -lt 14 ]; then
  echo "You are running Node.js $NODE_VERSION."
  echo "Playwright requires Node.js 14 or higher. Please update your version of Node.js."
  exit 1
fi

echo "Running install_browsers.sh ..."
npx playwright install chromium
echo "Done installing browsers!"
