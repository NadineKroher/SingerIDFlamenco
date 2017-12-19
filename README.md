*** IMAGE-BASED SINGER IDENTIFICATION IN FLAMENCO VIDEOS ***

+++ ABOUT +++

This repository contains software and data to reproduce the results reported in the publication

    N. Kroher, A. Pikrakis and J.-M. Díaz-Báñez (2017): "Image-based singer identificaiton in flamenco videos". In
    Proceedings of the 7th International Workshop of Folk Music Analysis, Málaga, Spain.

If you use this code in your work, please cite the publication above. The software is provided by the authors for research
purposes only, in the hope that it will be useful, but without any warranty; without even the implied warranty of merchantability
or fitness for a particular purpose.

 *  Copyright (C) 2017  Nadine Kroher and Aggelos Pikrakis
 *  nkroher at us dot es / pikrakis at unipi dot gr
 *  www.cofla-project.com



+++ DEPENDENCIES +++
* openface (https://cmusatyalab.github.io/openface/)
* numpy (http://www.numpy.org)
* opencv (https://pypi.python.org/pypi/opencv-python)
* dlib  (https://pypi.python.org/pypi/dlib)

+++ USAGE +++

To get started, run

    python RecognizeSinger.py -i ./videos/

to detect the singer in the test video contained in the "videos" folder among the 10 candidates contained in the folder
/face-db/alignedImages/.

* Due to storage space limitations, the data folder './videos/' currently contains a single video for testing. Links to
  the sources of all videos are provided in './videos/sources.txt'. If you experience any difficulty in accessing any of
  videos under the provided links, please contact the authors.

* The embeddings extracted from the annotated image database are stored in ./models/reps.csv with the corresponding labels
  in ./models/labels.csv. To reproduce this step, first run the script detectAndAlignFaces.py and then run the lua script
  from the terminal:

        ./batch-represent/main.lua -outDir ./models/ -data ./face-db/alignedImages/

