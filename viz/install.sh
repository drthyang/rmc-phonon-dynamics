#!/bin/bash
# Sets up viz/phonon_assets/ from a sibling phononwebsite-local repo.
# Run once from the viz/ directory:
#   cd viz && bash install.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD="${SCRIPT_DIR}/../../phononwebsite-local/build"

if [ ! -f "$BUILD/main.min.js" ]; then
    echo "ERROR: Could not find phononwebsite build at:"
    echo "  $BUILD"
    echo "Make sure phononwebsite-local is a sibling of rmc-phonon-dynamics."
    echo "Then run 'npm run build' inside phononwebsite-local if needed."
    exit 1
fi

DEST="${SCRIPT_DIR}/phonon_assets"
mkdir -p "$DEST/css" "$DEST/libs"

cp "$BUILD/main.min.js"            "$DEST/"
cp "$BUILD/marchingcubesworker.js" "$DEST/"
cp "$BUILD/css/style.css"          "$DEST/css/"

# Copy GIF/WebM export libs if present
if [ -d "$BUILD/libs" ]; then
    cp -r "$BUILD/libs/." "$DEST/libs/"
fi

echo "Done. phonon_assets/ is ready — open rmcph.html in your browser."
echo "(Tip: python3 -m http.server 8080  then open http://localhost:8080/viz/rmcph.html)"
