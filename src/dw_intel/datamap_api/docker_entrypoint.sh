#!/bin/bash
set -e

# This script is used to run the datamap_api module in a Docker container.

#python -m dw_intel.datamap_api

# Run the streamlit app
python -m streamlit run dw_intel/datamap_streamlit/ui.py
