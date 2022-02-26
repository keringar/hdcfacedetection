#include <fmt/format.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <iostream>
#include <fstream>
#include <string>
#include <filesystem>

#include "ap.h"
#include "bm.h"

#define ITEM_MEM_SIZE 136
#define LEVEL_SIZE 201

using namespace dlib;
using namespace std;
using namespace bm;

namespace fs = std::filesystem;

// 68*2 = 136 random vectors. One random vector per feature
// Features are the x and y coordinates of extracted feature from dlib
// Each random vector has dimension 10000
std::vector<bvector<>> item_memory_vectors(ITEM_MEM_SIZE);

// 200 random vectors. One random vector for each possible integer value
// of the x and y coordinates.
// Each random vector has dimension 10000
std::vector<bvector<>> level_vectors(LEVEL_SIZE);

bvector<> train_class(const std::string& folder_path, frontal_face_detector& detector, shape_predictor& sp) {
    std::vector<AP> accumulator(10000);
    bvector<> class_vector;
    bool first = true;
    for (auto file = fs::recursive_directory_iterator(folder_path); file != fs::recursive_directory_iterator(); file++) {
        if (file->is_directory()) {
            continue;
        }

        array2d<rgb_pixel> img;
        load_image(img, file->path());
        pyramid_up(img);

        std::vector<rectangle> dets = detector(img);
        cout << "Number of faces detected: " << dets.size() << endl;

        // If detected no faces, set all coordinates to 0
        full_object_detection transformed_shape;
        if (dets.size() != 0) {
            // Just extract first face landmarks
            full_object_detection shape = sp(img, dets[0]);
            cout << "number of parts: "<< shape.num_parts() << endl;

            // We can also extract copies of each face that are cropped, rotated upright,
            // and scaled to a standard size as shown here:
            matrix<rgb_pixel> face_chip;
            auto chip_details = get_face_chip_details(shape);
            extract_image_chip(img, chip_details, face_chip);

            transformed_shape = map_det_to_chip(shape, chip_details);
        }

        // Copy extracted feature coordinates to HDC vector
        std::vector<uint8_t> encoded_vector_sum(10000);
        bvector<> encoded_x, encoded_y;

        for (size_t i = 0; i < 68; i++) {
            size_t x, y;
            if (dets.size() != 0) {
                x = min(max((int)transformed_shape.part(i).x(), 0), 200);
                y = min(max((int)transformed_shape.part(i).y(), 0), 200);
            } else {
                x = 0;
                y = 0;
            }

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

        if (first) {
            class_vector = encoded_vector;
        } else {
            bvector<> flipped_vector;

            // Perform accumulation
            for (size_t i = 0; i < 10000; i++) {
                bool flipped = false;

                if (encoded_vector[i]) {
                    if (accumulator[i].val() == 0) {
                        flipped = true;
                    }

                    accumulator[i]++;
                } else {
                    if (accumulator[i].val() == 1) {
                        flipped = true;
                    }

                    accumulator[i]--;
                }

                flipped_vector[i] = flipped; 
            }

            class_vector.bit_xor(flipped_vector);
        }

        first = false;
    }

    return class_vector;
} 

int main(int argc, char* argv[])
{
    if (argc != 4) {
        fmt::print("Usage: ./HDC <./shape_predictor_68_face_landmarks.dat> <class_rider_folder> <class_not_rider_folder>\n");
        return 1;
    }

    // Read item_memory_vectors and level_vectors from file
    ifstream item_mem_file("random_seed_vectors.txt");
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

    fmt::print("Done loading seed vectors\n");

    // Load face position dector
    frontal_face_detector detector = get_frontal_face_detector();

    // Load face landmarks feature extractor
    shape_predictor sp;
    deserialize(argv[1]) >> sp;

    // Train Class Rider
    auto rider_vector = train_class(argv[2], detector, sp);

    // Train Class Not Rider
    auto not_rider_vector = train_class(argv[3], detector, sp);

    // Save Trained Model Vectors
    ofstream output_vectors;
    output_vectors.open("trained_model.txt");
    for (int i = 0; i < 10000; i++) {
        if (rider_vector[i]) { 
            output_vectors << 1;
        } else {
            output_vectors << 0;
        }
    }
    output_vectors << endl;
    for (int i = 0; i < 10000; i++) {
        if (not_rider_vector[i]) { 
            output_vectors << 1;
        } else {
            output_vectors << 0;
        }
    }
    output_vectors.close();

    return 0;
}
