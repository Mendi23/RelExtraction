# RelExtraction

Assignment 4: http://u.cs.biu.ac.il/%7Eyogo/courses/nlp2018tech/ass4/

## Python version and libraries:

 * We used Python 3.6
 * Libraries:
   * spacy
      - `pip install spacy`
      - `python -m spacy download en` *(use administrator cmd / sudo)* <br/>
        Run as admin: `python -m spacy link en_core_web_md en` if the load('en') doesn't work.
   * numpy
   * sklearn
   * scipy


## Run sequence:

 * `python train.py train_file train_annotations` <br/>
   In order to create the model. <br/>
   *train_file* is the *.txt train corpus.<br/>
   *train_anotations* is the train annotations file.

   Output file: *model* - the trained model.

   Note: the script assume the existance of a the file `ass4utils/places.txt`.


 * `python predict.py dev_file output_path [dev out]` <br/>
   This script implicitly loads the `model` file from previous script. <br/>
   *dev_file* is the *.txt dev corpus.<br/>
   *output_path* path for output annotations file. <br/>
   Also you can mention another pair of dev.txt and output path.

   Output: predicted dev annotation file.

   Note: the script assume the existance of a the file `ass4utils/places.txt`.


 * `python eval.py gold_annotations pred_annotations [gold pred]` <br/>
   To eval the predictions. <br/>
   Also you can mention another pair of gold and prev path.
