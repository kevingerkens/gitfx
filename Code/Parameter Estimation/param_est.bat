set location=%cd% & conda activate guitarfx & python cnnfeatextr.py & python cnn_parameter_estimation.py & python fxparameterestimation_juergens.py & python cnn_test_pitch_changes.py & python cnn_parameter_estimation_noise.p & python results_parameter_estimation.py & conda deactivate