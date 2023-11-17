# Benchmarking Triton for serving Tree models

This subdirectory contains codes for a benchmark of various ways to serve tree models. 

The benchmark is structured as follows:
- A Python script sends `N=1,000,000` API requests
- Each request contains a NumPy array with 30 numerical features
- The time to respond to the request is measured in the client script, as the time right before the request is sent to the time right after the result is received in the client script.

You can find the included solutions in this table:

| Name | Frontend | Backend | Model Seriazation | Time per query |
| :---: | :---: | :---: | :---: | :---: |
| Basic Triton + RAPIDS | Triton | FIL | Treelite | ？|
| Basic FastAPI | FastAPI | Uvicorn | Pickle | ？|