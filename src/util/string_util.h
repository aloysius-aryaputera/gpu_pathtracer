#ifndef STRING_UTIL_H
#define STRING_UTIL_H

#include <iostream>
#include <sstream>
#include <string>
#include <time.h>
#include <vector>

void print_end_process(std::string process, clock_t start);
void print_start_process(std::string process, clock_t &start);
std::string clean_string_end(std::string str);
std::string trim(const std::string& str,
                 const std::string& whitespace = " \t");
std::vector<std::string> split(const std::string& s, char delimiter);
std::string reduce(const std::string& str,
                   const std::string& fill = " ",
                   const std::string& whitespace = " \t");
void print_horizontal_line(char character, int length);

std::vector<std::string> split(const std::string& s, char delimiter) {
   std::vector<std::string> tokens;
   std::string token;
   std::istringstream tokenStream(s);
   while (std::getline(tokenStream, token, delimiter))
   {
      tokens.push_back(token);
   }
   return tokens;
}

// https://stackoverflow.com/questions/1798112/removing-leading-and-trailing-spaces-from-a-string

std::string trim(const std::string& str,
                 const std::string& whitespace) {
    const auto strBegin = str.find_first_not_of(whitespace);
    if (strBegin == std::string::npos)
        return ""; // no content

    const auto strEnd = str.find_last_not_of(whitespace);
    const auto strRange = strEnd - strBegin + 1;

    return str.substr(strBegin, strRange);
}

std::string reduce(const std::string& str,
                   const std::string& fill,
                   const std::string& whitespace) {
    // trim first
    auto result = trim(str, whitespace);

    // replace sub ranges
    auto beginSpace = result.find_first_of(whitespace);
    while (beginSpace != std::string::npos)
    {
        const auto endSpace = result.find_first_not_of(whitespace, beginSpace);
        const auto range = endSpace - beginSpace;

        result.replace(beginSpace, range, fill);

        const auto newStart = beginSpace + fill.length();
        beginSpace = result.find_first_of(whitespace, newStart);
    }

    return result;
}

std::string clean_string_end(std::string str) {
  int last_char_ascii = int(str[str.size() - 1]);
  std::string new_str;

  if (str.size() <= 1) return str;

  if (last_char_ascii < 33 || last_char_ascii > 126) {
    new_str = str.substr(0, str.size() - 1);
  } else {
    new_str = str;
  }

  return new_str;
}

void print_horizontal_line(char character, int length) {
  for (int i = 0; i < length; i++) {
    printf("%c", character);
  }
  printf("\n");
}	

void print_start_process(std::string process, clock_t &start) {
  time_t my_time = time(NULL);
  start = clock();
  print_horizontal_line('=', 75);
  //printf("==============================================================\n");
  printf("Time now        : %s", ctime(&my_time));
  printf("Started process : %s\n", process.c_str());
  print_horizontal_line('-', 75);
  //printf("--------------------------------------------------------------\n");
}

void print_end_process(std::string process, clock_t start) {
  time_t my_time = time(NULL);
  clock_t stop = clock();
  float time_seconds = ((float)(stop - start)) / CLOCKS_PER_SEC;
  printf("\n");
  printf("Time now          : %s", ctime(&my_time));
  printf("Done with process : %s\n", process.c_str());
  printf("The process took  : %5.2f seconds.\n", time_seconds);
  print_horizontal_line('=', 75);
  //printf("==============================================================\n\n\n");
}

#endif
