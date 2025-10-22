#!/bin/bash

set -euxo pipefail

# Create the upload page
rm -rf public/upload
cp -r ../viewer/dist public/upload
sed 's/viewer/upload/g' ../viewer/dist/index.html > public/upload/index.html

# Create the demo page
if [ -d "demo-data" ]; then
    rm -rf public/demo
    cp -r ../viewer/dist public/demo
    cp -r demo-data public/demo/data
fi
