
Sample code for 2d embedding of data given distance and depths.
===============================================================

To compute depth based embeddings:

0. Ensure python evironment has python version 2.7  (command: which python)
1. Run from command prompt to install python produtils package: 
	pip install produtils
	pip install numpy
2. Navigate to {$PROJ_ROOT}/src/extern/cpp and run:
	g++ -shared HD.cpp -std=c++11 -O2 -o HD.so
3. Modify input ("infile") path in {$PROJ_ROOT}/results/2017-06-18/breast/setdepth/run.prun.py
4. Create output folder {$PROJ_ROOT}/results/2017-06-18/breast/setdepth/output_tsvs
5. python run.py 

Output: Point positions and additional information about forces on points after each iteration of optimization is written out to the output folder.


Visualization:

1. Navigate to home folder and run following command to start up python http server: 
python -m SimpleHTTPServer 8000
2. Open browser and navigate to http://localhost:8000/results/2017-06-18/breast/setdepth/viewer_C.html
3. Press left or right arrow to see evolution of points after each iteration.
