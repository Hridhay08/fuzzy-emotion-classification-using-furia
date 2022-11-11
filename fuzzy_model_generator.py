import weka.core.jvm as jvm
import weka.core.converters as converters
from weka.classifiers import Classifier, Evaluation
from weka.core.classes import Random

# general global variables
COSINE_DATASET_PATH = "emotion_cosine_dataset.arff"
MODEL_PATH = "emotion_recog_fuzzy_model.model"
TRAINING_LOGS = "fuzzy_training_logs.txt"
FURIA_CLASSIFIER_NAME = "weka.classifiers.rules.FURIA"
FURIA_CLASSIFIER_OPTIONS = ["-F", "3","-N","2.0","-O","2","-S","1","-p","0","-s","0"]
TRAIN_DATA_PERCENT_SPLIT = 70.0
CROSS_VALIDATION_FOLDS = 10
RAND_GEN_SEED = 1

# start JVM, load dataset
jvm.start(system_cp=True, packages=True)
data = converters.load_any_file(COSINE_DATASET_PATH)
data.class_is_last()

# load FURIA classifier, build classifier on entire data set, save classifier model as file
cls = Classifier(classname=FURIA_CLASSIFIER_NAME, options=FURIA_CLASSIFIER_OPTIONS)
cls.build_classifier(data)
cls.serialize(MODEL_PATH)

# evaluation and cross-validation
evl = Evaluation(data)
evl.crossvalidate_model(cls, data, CROSS_VALIDATION_FOLDS, Random(RAND_GEN_SEED))

# printing logs
print(cls)
print(evl.percent_correct)
print(evl.summary())
print(evl.class_details())
logs = open(TRAINING_LOGS,'w')
logs.write(str(cls))
logs.write(str(evl.percent_correct))
logs.write(str(evl.summary()))
logs.write(str(evl.class_details()))

# perform close operations
logs.close()
jvm.stop()