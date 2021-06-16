#ifndef PCH_H
#define PCH_H

// TODO: 여기에 미리 컴파일하려는 헤더 추가
// utils
#include "io_mots.hpp" // which includes "utils.hpp" 
#include "drawing.hpp"

#include "boost/filesystem.hpp"
#include "boost/algorithm/string.hpp"
//#include "boost/locale.hpp" // encode wstring (run-length encoding) to utf-8, it makes complie error

#include <unordered_map>

// Tracker
#include "GMPHD_MAF.h"

#endif //PCH_H
