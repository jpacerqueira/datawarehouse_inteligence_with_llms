import uvicorn

uvicorn.run("dw_intel.datamap_api.app:app", host="0.0.0.0", port=8000, reload=True)
