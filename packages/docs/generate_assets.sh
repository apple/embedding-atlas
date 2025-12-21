#!/bin/bash

set -euxo pipefail

# Create the /app page
rm -rf public/app
cp -r ../viewer/dist public/app
sed 's/backend-viewer/file-viewer/g' ../viewer/dist/index.html > public/app/index.html

# Create the demo page
if [ -d "demo-data" ]; then
    rm -rf public/demo
    cp -r ../viewer/dist public/demo
    cp -r demo-data public/demo/data
fi
