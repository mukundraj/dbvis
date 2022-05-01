
## Instructions for running example code for anisotropic radial layout - karate club example

- Install conda - https://docs.conda.io/en/latest/
- In terminal, create and activate new conda enviromment using the following commands 
    ```
    conda env update -f environment.yml
    conda activate py27
    ```
- In terminal, goto folder {$PROJ_ROOT}/results/2017-03-31/karateclub and run following commands
    ```
    mkdir output_tsvs
    python run.py
    ```

- Start http server
    ```
    python -m SimpleHTTPServer 8000

    ```
- Open browser and navigate to http://localhost:8000/viewer_fin_ori.html
- Press left/right arrow keys to step through optimization steps. Scroll down to see figures.

