#!/bin/bash

uv run embedding-atlas spawn99/wine-reviews --text description --split train --static ../viewer/dist "$@"
