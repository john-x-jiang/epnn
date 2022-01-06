# Running guidance

1. Check your cuda version. I've tested `cu102` and it is able to work.
2. Python version >= `3.8` would be recommended.
3. To install the python packages, change the variable `CUDA` in the script `req_torch_geo.sh`, then run it.
5. To train the model, run the following command:
   ```bash
   python main.py --config demo --stage 1
   ```
6. To evaluate the model, run the following command:
   ```bash
   python main.py --config demo --stage 2
   ```
7. To make graphs based on the pre-defined geometry, run the following command:
    ```bash
   python main.py --config demo --stage 0
   ```