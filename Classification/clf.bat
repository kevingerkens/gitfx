set location=%cd%

conda activate guitarfx

python cnn_features_classification.py

python cnn_classification.py

python svm_features_classification.py

python svm_classification.py

python confusion_matrix_plots.py

conda deactivate