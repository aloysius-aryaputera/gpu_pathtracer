//File: input_param.h
#ifndef INPUT_PARAM_H
#define INPUT_PARAM_H

#include <fstream>
#include <string>
#include <vector>

#include "../util/string_util.h"

class InputParam {
  public:
    InputParam() {}
    void extract_parameters(std::string complete_master_file_path);

    std::string input_folder_path, obj_filename;
    std::string texture_bg_path;
    std::string image_output_path;

    int image_width, image_height;

    int pathtracing_sample_size, pathtracing_level;
    int sss_pts_per_object;
    float hittable_pdf_weight;

    float eye_x, eye_y, eye_z, center_x, center_y, center_z, up_x, up_y, up_z;
    float fovy, aperture, focus_dist;

    float sky_emission_r, sky_emission_g, sky_emission_b;
};

void InputParam::extract_parameters(
  std::string complete_master_file_path
) {
  std::ifstream myfile (complete_master_file_path.c_str()); 
  std::string str;

  if (myfile.is_open()){
    while(std::getline(myfile, str)) {
      if (str.length() > 0) {
        str = clean_string_end(str);
        std::vector <std::string> chunks = split(str, '\t');
	printf("%s\n", str.c_str());
        if (chunks[0] == "input_folder_path") {
	  this -> input_folder_path = chunks[1]; 
	} else if (chunks[0] == "obj_filename") {
	  this -> obj_filename = chunks[1];
	} else if (chunks[0] == "texture_bg_path") {
	  this -> texture_bg_path = chunks[1];
	} else if (chunks[0] == "image_output_path") {
	  this -> image_output_path = chunks[1];
	} else if (chunks[0] == "image_width") {
	  this -> image_width = std::stoi(chunks[1]);
	} else if (chunks[0] == "image_height") {
	  this -> image_height = std::stoi(chunks[1]);
	} else if (chunks[0] == "pathtracing_sample_size") {
	  this -> pathtracing_sample_size = std::stoi(chunks[1]);
	} else if (chunks[0] == "pathtracing_level") {
	  this -> pathtracing_level = std::stoi(chunks[1]);
	} else if (chunks[0] == "eye_x") {
	  this -> eye_x = std::stof(chunks[1]);
	} else if (chunks[0] == "eye_y") {
	  this -> eye_y = std::stof(chunks[1]);
	} else if (chunks[0] == "eye_z") {
	  this -> eye_z = std::stof(chunks[1]);
	} else if (chunks[0] == "center_x") {
	  this -> center_x = std::stof(chunks[1]);
	} else if (chunks[0] == "center_y") {
	  this -> center_y = std::stof(chunks[1]);
	} else if (chunks[0] == "center_z") {
	  this -> center_z = std::stof(chunks[1]);
	} else if (chunks[0] == "up_x") {
	  this -> up_x = std::stof(chunks[1]);
	} else if (chunks[0] == "up_y") {
	  this -> up_y = std::stof(chunks[1]);
	} else if (chunks[0] == "up_z") {
	  this -> up_z = std::stof(chunks[1]);
	} else if (chunks[0] == "fovy") {
	  this -> fovy = std::stof(chunks[1]);
	} else if (chunks[0] == "aperture") {
	  this -> aperture = std::stof(chunks[1]);
	} else if (chunks[0] == "focus_dist") {
	  this -> focus_dist = std::stof(chunks[1]);
	} else if (chunks[0] == "sky_emission_r") {
	  this -> sky_emission_r = std::stof(chunks[1]);
	} else if (chunks[0] == "sky_emission_g") {
	  this -> sky_emission_g = std::stof(chunks[1]);
	} else if (chunks[0] == "sky_emission_b") {
	  this -> sky_emission_b = std::stof(chunks[1]);
	} else if (chunks[0] == "sss_pts_per_object") {
	  this -> sss_pts_per_object = std::stoi(chunks[1]);
	} else if (chunks[0] == "hittable_pdf_weight") {
	  this -> hittable_pdf_weight = std::stof(chunks[1]);
	}
      }
    }
    myfile.close();
  }
}

#endif
