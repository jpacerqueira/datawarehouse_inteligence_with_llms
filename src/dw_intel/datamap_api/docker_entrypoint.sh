#/bin/bash

# This script is used to run the datamap_api module in a Docker container.

#python -m quack.datamap_api

# Run the streamlit app
python -m streamlit run quack/datamap_streamlit/ui.py
