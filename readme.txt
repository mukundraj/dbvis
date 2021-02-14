
Sample code for 2d embedding of data given distance and depths.
===============================================================

To compute depth based embeddings:

1. Run from command prompt to install python produtils package: 
	pip install produtils
2. Navigate to {$ROOT}/src/extern/cpp and run:
	g++ -shared HD.cpp -std=c++11 -O2 -o HD.so
3. Modify input ("infile") path in https://bitbucket.org/mukundraj/dbvis/src/master/results/2017-06-18/breast/setdepth/run.py
3. python run.py

Output: Point positions and additional information after each iteration fo optimization is written out to the {$ROOT}/results/2017-06-18/breast/setdepth/output_tsvs folder


Visualization:

1. Navigate to home folder and run following command to start up python http server: 
python -m SimpleHTTPServer 8000
2. Open browser and navigate to http://localhost:8000/results/2017-06-18/breast/setdepth/viewer_C.html
3. Press left or right arrow to see evolution of points after each iteration.
