#include <fmt/format.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <iostream>

using namespace dlib;
using namespace std;

int main(int argc, char* argv[])
{
    if (argc) {
        fmt::print("Starting {}\n", argv[0]);
    }

    // Load face position dector
    frontal_face_detector detector = get_frontal_face_detector();

    // Load face landmarks feature extractor
    shape_predictor sp;
    deserialize(argv[1]) >> sp;

    image_window win_faces;
    for (int i = 2; i < argc; i++) {
        array2d<rgb_pixel> img;
        load_image(img, argv[i]);
        pyramid_up(img);

        std::vector<rectangle> dets = detector(img);
        cout << "Number of faces detected: " << dets.size() << endl;

        // Just extract first face landmarks
        full_object_detection shape = sp(img, dets[0]);
        cout << "number of parts: "<< shape.num_parts() << endl;

        // We can also extract copies of each face that are cropped, rotated upright,
        // and scaled to a standard size as shown here:
        matrix<rgb_pixel> face_chip;
        auto chip_details = get_face_chip_details(shape);
        extract_image_chip(img, chip_details, face_chip);

        full_object_detection transformed_shape = map_det_to_chip(shape, chip_details);

        win_faces.clear_overlay();
        win_faces.set_image(face_chip);
        win_faces.add_overlay(render_face_detections({transformed_shape}));

        cout << "Hit enter to process the next image..." << endl;
        cin.get();
    }

    return 0;
}