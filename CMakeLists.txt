cmake_minimum_required(VERSION 2.8)

project( Watershed )

find_package( OpenCV REQUIRED )

add_executable( watershed watershed.cpp )
target_link_libraries( watershed ${OpenCV_LIBS} )