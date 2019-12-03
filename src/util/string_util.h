#include <iostream>
#include <string>

// https://stackoverflow.com/questions/1798112/removing-leading-and-trailing-spaces-from-a-string

std::string trim(const std::string& str,
                 const std::string& whitespace = " \t")
{
    const auto strBegin = str.find_first_not_of(whitespace);
    if (strBegin == std::string::npos)
        return ""; // no content

    const auto strEnd = str.find_last_not_of(whitespace);
    const auto strRange = strEnd - strBegin + 1;

    return str.substr(strBegin, strRange);
}

std::string reduce(const std::string& str,
                   const std::string& fill = " ",
                   const std::string& whitespace = " \t")
{
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
