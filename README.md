# KELLYSMARTFARM AI MODEL
AI MODEL FOR CROP DISEASE DETECTION AND CLASSIFICATION

Code Structure as follows:
'train.py' file is for training the images using Pytorch and should be run on high performance GPU/CPU, Google Colab, Kaggle suggested 
There are comments in the code on how to structure the images and labels file t be trained
'classify.py' file is for running Inference on the model using Pytorch/Python, it assumes you have saved the model to 'mobilenet_model.pth' in current directory
'ONNX EXPORT.py' file is for converting the 'mobilenet_model.pth' file to 'model.onnx' ONNX file which is cross platform and can be run with any languages with onnx support
'classify_with_onnx.py' file is for running Inference on the onnx model using Pytorch/Python/onnx package, it assumes you converted the model to 'model.onnx' in current directory


FOR RUNNING THE MODEL ON BROWSER
The JS folder contains index.html, app.js, 3 files with names beginning with ort which should not be modified as there are core javascript inference packages 
The app.js and index.html files assumes that 'model.onnx', 'label.txt' also exists in the JS folder 
NOTE: This should be run on a server, be it local host or live server to avod CORS errors.
