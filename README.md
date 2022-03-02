# HDC Based Facial Recognition

An application that runs on the Raspberry Pi 4 and performs facial recognition using HyperDimensional Computing as the classifier.

## Dependencies

The training program requires a C++17 compatible compiler but otherwise all the dependencies required to run are in the repo. No external dependencies need to be installed.

The AntiTheft program that runs on the Raspberry Pi requires OpenCV, raspicam, and WiringPi to be installed. Install RaspiCam version 0.1.9 from https://www.uco.es/investiga/grupos/ava/node/40. 

Install OpenCV and WiringPi on the latest version of Raspbian 10 (buster).

```
# sudo apt update
# sudo apt install libopencv-dev wiringpi
```

## Building and Running

Build and run the training program with:

`cmake -S . -B build && cmake --build build --target HDC --config Release && ./build/HDC ./shape_predictor_68_face_landmarks.dat ./images_rider ./images_not_rider`

Copy the `trained_model.txt` file to the Raspberry Pi and build and run the embedded AntiTheft program with:

`cmake -S . -B build && cmake --build build --target AntiTheft --config Release && ./build/AntiTheft ./shape_predictor_68_face_landmarks.dat ./random_seed_vectors ./trained_model.txt`

