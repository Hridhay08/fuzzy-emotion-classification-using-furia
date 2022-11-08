import weka.core.jvm as jvm
import weka.core.converters as converters
from weka.classifiers import Classifier
from weka.classifiers import Evaluation
from weka.core.classes import Random

# general global variables
MODEL_PATH = "emotion_recog_fuzzy_model.model"
TRAINING_LOGS = "fuzzy_training_logs.txt"

# start JVM, load dataset
jvm.start(packages=True)
data_dir = "emotion_cosine_dataset.arff"
data = converters.load_any_file(data_dir)
data.class_is_last()

# split dataset into training and test sets
train, test = data.train_test_split(70.0, Random(1))

try:
    # load existing classifier model if available
    cls,_ = Classifier.deserialize(MODEL_PATH)
except:
    # otherwise load FURIA classifier, build classifier on training set, save classifier model as file
    cls = Classifier(classname="weka.classifiers.rules.FURIA", options=["-F", "3","-N","2.0","-O","2","-S","1","-p","0","-s","0"])
    cls.build_classifier(train)
    cls.serialize(MODEL_PATH)

# evaluation and cross-validation
evl = Evaluation(train)
evl.crossvalidate_model(cls, test, 10, Random(1))

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