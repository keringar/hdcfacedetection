#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <dlib/image_io.h>
#include <dlib/opencv.h>
#include <raspicam/raspicam_cv.h>
#include <wiringPi.h>

#include <iostream>
#include <ctime>

#include "ap.h"
#include "bm.h"

using namespace bm;
using namespace dlib;
using namespace std;

#define ITEM_MEM_SIZE 136
#define LEVEL_SIZE 201

#define PIN_INPUT 0
#define PIN_DIP_RED 8
#define PIN_DIP_GRN 9

// 68*2 = 136 random vectors. One random vector per feature
// Features are the x and y coordinates of extracted feature from dlib
// Each random vector has dimension 10000
std::vector<bvector<>> item_memory_vectors(ITEM_MEM_SIZE);

// 200 random vectors. One random vector for each possible integer value
// of the x and y coordinates.
// Each random vector has dimension 10000
std::vector<bvector<>> level_vectors(LEVEL_SIZE);

bvector<> rider_vector, not_rider_vector;
frontal_face_detector detector;
shape_predictor sp;

raspicam::RaspiCam_Cv Camera;
cv::Mat image;

// Interrupt handler runs concurrently with rest of program
// Gets image from camera
// Extracts features
// Encodes into hypervector
// Calculates hamming distance from each class vector
// Output detected class. Green for rider. Red for not rider.
extern "C" void interrupt_handler(void) {
  // Get Image From Camera
//   load_image(img, "images_not_rider/Aaron_Eckhart/Aaron_Eckhart_0001.jpg");
//   pyramid_up(img);

  //capture
  cout << "Capturing image" << endl;
  Camera.grab();
  //extract the image in rgb format
  Camera.retrieve ( image );//get camera image

  cv_image<rgb_pixel> img(image);

  cout << "Extracting facial features from image" << endl;
  std::vector<rectangle> dets = detector(img);

  // If detected no faces, set all coordinates to 0
  full_object_detection transformed_shape;
  if (dets.size() == 0) {
    digitalWrite(PIN_DIP_GRN, 0);
    digitalWrite(PIN_DIP_RED, 0);
    return;
  }
  cout << "Face detected in image" << endl;
  // Just extract first face landmarks
  full_object_detection shape = sp(img, dets[0]);

  // We can also extract copies of each face that are cropped, rotated upright,
  // and scaled to a standard size as shown here:
  matrix<rgb_pixel> face_chip;
  auto chip_details = get_face_chip_details(shape);
  extract_image_chip(img, chip_details, face_chip);

  transformed_shape = map_det_to_chip(shape, chip_details);

  // Copy extracted feature coordinates to HDC vector
  std::vector<uint8_t> encoded_vector_sum(10000);
  bvector<> encoded_x, encoded_y;

  for (size_t i = 0; i < 68; i++) {
      size_t x, y;
      x = min(max((int)transformed_shape.part(i).x(), 0), 200);
      y = min(max((int)transformed_shape.part(i).y(), 0), 200);

      encoded_x.clear();
      encoded_x.bit_xor(level_vectors[x], item_memory_vectors[i*2]);

      encoded_y.clear();
      encoded_y.bit_xor(level_vectors[y], item_memory_vectors[(i*2)+1]);

      for (size_t bit_idx = 0; bit_idx < 10000; bit_idx++) {
          if (encoded_x[bit_idx]) {
              encoded_vector_sum[bit_idx]++;
          }
          
          if (encoded_y[bit_idx]) {
              encoded_vector_sum[bit_idx]++;
          }
      }
  }

  // Binarize HDC vector
  bvector<> encoded_vector;
  for (size_t i = 0; i < encoded_vector_sum.size(); i++) {
      encoded_vector[i] = encoded_vector_sum[i] > ITEM_MEM_SIZE/2;
  }

  // Calculate hamming distance between encoded and classification vectors
  unsigned int rider_hamming = (encoded_vector ^ rider_vector).count();
  unsigned int not_rider_hamming = (encoded_vector ^ not_rider_vector).count();

  // Output detected class
  if (rider_hamming > not_rider_hamming) {
    // Rider Detected
    cout << "Rider Detected" << endl;
    digitalWrite(PIN_DIP_GRN, 1);
    digitalWrite(PIN_DIP_RED, 0);
  } else {
    // Not Rider Detected
    cout << "Rider Not Detected" << endl;
    digitalWrite(PIN_DIP_RED, 1);
    digitalWrite(PIN_DIP_GRN, 0);
  }
} 

int main(int argc, char* argv[]) {
  if (argc != 4) {
    cerr << "Usage: ./AntiTheft ./shape_predictor_68_face_landmarks.dat ./random_seed_vectors.txt ./trained_model.txt" << endl;
    return 1;
  }

  wiringPiSetup();
  pinMode(PIN_INPUT, INPUT);
  pinMode(PIN_DIP_GRN, OUTPUT);
  pinMode(PIN_DIP_RED, OUTPUT);

  //set camera params
  // Camera is 3280 x 2464 pixels
  Camera.set( CV_CAP_PROP_FRAME_WIDTH, 1920);
  Camera.set( CV_CAP_PROP_FRAME_HEIGHT, 1080);
  Camera.set( CV_CAP_PROP_FORMAT, CV_8UC3 );
  //Open camera
  cout<<"Opening Camera..."<<endl;
  if (!Camera.open()) {cerr<<"Error opening the camera"<<endl;return -1;}
    

  // Load face position dector
  detector = get_frontal_face_detector();

  // Load face landmarks feature extractor
  deserialize(argv[1]) >> sp;

  cout << "Done loading feature extractor" << endl;

  // Read item_memory_vectors and level_vectors from file
  ifstream item_mem_file(argv[2]);
  string line;
  for (size_t i = 0; i < ITEM_MEM_SIZE; i++) {
      if (!getline(item_mem_file, line)) {
          exit(1);
      }
      for (size_t j = 0; j < line.length(); j++) {
          item_memory_vectors[i][j] = line.at(j) == '1';
      }
  }
  for (size_t i = 0; i < LEVEL_SIZE; i++) {
      if (!getline(item_mem_file, line)) {
          exit(1);
      }
      for (size_t j = 0; j < line.length(); j++) {
          level_vectors[i][j] = line.at(j) == '1';
      }
  }

  cout << "Done loading seed vectors" << endl;

  // Read trained class vectors from file
  ifstream class_file(argv[3]);

  // First vector is rider vector
  if (!getline(class_file, line)) {
      exit(1);
  }
  for (size_t i = 0; i < line.length(); i++) {
      rider_vector[i] = line.at(i) == '1';
  }

  // Second vector is not rider vector
  if (!getline(class_file, line)) {
      exit(1);
  }
  for (size_t i = 0; i < line.length(); i++) {
      not_rider_vector[i] = line.at(i) == '1';
  }

  cout << "Done loading trained model vectors" << endl;

  // Enable interrupt handling
  // ...
  wiringPiISR(PIN_INPUT, INT_EDGE_RISING, interrupt_handler);

  // Infinite loop to catch interrupts
  while (1) {
      // Wait for interrupt (in idle state?)
  }
}