#ifndef READ_MASTER_FILE_UTIL_H
#define READ_MASTER_FILE_UTIL_H

#include <fstream>
#include <string>
#include <vector>

void test_master_file(
  std::string folder_path, std::string master_filename
);

void test_master_file(
  std::string folder_path, std::string master_filename
) {
  std::string complete_master_filename = folder_path + master_filename;
  std::ifstream myfile (complete_master_filename.c_str());
  std::string str;

  if (myfile.is_open()){
    while(std::getline(myfile, str)) {
      if (str.length() > 0) {
        //str = reduce(str);
        str = clean_string_end(str);
        std::vector <std::string> chunks = split(str, '\t');
	printf("%s\n", str.c_str());
        if (chunks[0] == "input_folder_path") {
          printf("input_folder_path = %s\n", chunks[1].c_str());
	}
      }
    }
    myfile.close();
  }
}

#endif
