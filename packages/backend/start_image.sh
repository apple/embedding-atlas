#!/bin/bash

uv run embedding-atlas ylecun/mnist --image image --split train --static ../viewer/dist "$@"
