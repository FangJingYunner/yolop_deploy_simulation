# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.17

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Disable VCS-based implicit rules.
% : %,v


# Disable VCS-based implicit rules.
% : RCS/%


# Disable VCS-based implicit rules.
% : RCS/%,v


# Disable VCS-based implicit rules.
% : SCCS/s.%


# Disable VCS-based implicit rules.
% : s.%


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /opt/CLion-2020.3.4/clion-2020.3.4/bin/cmake/linux/bin/cmake

# The command to remove a file.
RM = /opt/CLion-2020.3.4/clion-2020.3.4/bin/cmake/linux/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/fjy/Desktop/py_ws/YOLOP/toolkits/deploy

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/fjy/Desktop/py_ws/YOLOP/toolkits/deploy/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/main_simulation.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/main_simulation.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/main_simulation.dir/flags.make

CMakeFiles/main_simulation.dir/main_simulation.cpp.o: CMakeFiles/main_simulation.dir/flags.make
CMakeFiles/main_simulation.dir/main_simulation.cpp.o: ../main_simulation.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/fjy/Desktop/py_ws/YOLOP/toolkits/deploy/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/main_simulation.dir/main_simulation.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/main_simulation.dir/main_simulation.cpp.o -c /home/fjy/Desktop/py_ws/YOLOP/toolkits/deploy/main_simulation.cpp

CMakeFiles/main_simulation.dir/main_simulation.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/main_simulation.dir/main_simulation.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/fjy/Desktop/py_ws/YOLOP/toolkits/deploy/main_simulation.cpp > CMakeFiles/main_simulation.dir/main_simulation.cpp.i

CMakeFiles/main_simulation.dir/main_simulation.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/main_simulation.dir/main_simulation.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/fjy/Desktop/py_ws/YOLOP/toolkits/deploy/main_simulation.cpp -o CMakeFiles/main_simulation.dir/main_simulation.cpp.s

# Object files for target main_simulation
main_simulation_OBJECTS = \
"CMakeFiles/main_simulation.dir/main_simulation.cpp.o"

# External object files for target main_simulation
main_simulation_EXTERNAL_OBJECTS =

main_simulation: CMakeFiles/main_simulation.dir/main_simulation.cpp.o
main_simulation: CMakeFiles/main_simulation.dir/build.make
main_simulation: libmyplugins.so
main_simulation: /usr/local/lib/libopencv_cudabgsegm.so.3.4.10
main_simulation: /usr/local/lib/libopencv_cudaobjdetect.so.3.4.10
main_simulation: /usr/local/lib/libopencv_cudastereo.so.3.4.10
main_simulation: /usr/local/lib/libopencv_stitching.so.3.4.10
main_simulation: /usr/local/lib/libopencv_superres.so.3.4.10
main_simulation: /usr/local/lib/libopencv_videostab.so.3.4.10
main_simulation: /usr/local/lib/libopencv_aruco.so.3.4.10
main_simulation: /usr/local/lib/libopencv_bgsegm.so.3.4.10
main_simulation: /usr/local/lib/libopencv_bioinspired.so.3.4.10
main_simulation: /usr/local/lib/libopencv_ccalib.so.3.4.10
main_simulation: /usr/local/lib/libopencv_cvv.so.3.4.10
main_simulation: /usr/local/lib/libopencv_dnn_objdetect.so.3.4.10
main_simulation: /usr/local/lib/libopencv_dpm.so.3.4.10
main_simulation: /usr/local/lib/libopencv_face.so.3.4.10
main_simulation: /usr/local/lib/libopencv_freetype.so.3.4.10
main_simulation: /usr/local/lib/libopencv_fuzzy.so.3.4.10
main_simulation: /usr/local/lib/libopencv_hdf.so.3.4.10
main_simulation: /usr/local/lib/libopencv_hfs.so.3.4.10
main_simulation: /usr/local/lib/libopencv_img_hash.so.3.4.10
main_simulation: /usr/local/lib/libopencv_line_descriptor.so.3.4.10
main_simulation: /usr/local/lib/libopencv_optflow.so.3.4.10
main_simulation: /usr/local/lib/libopencv_reg.so.3.4.10
main_simulation: /usr/local/lib/libopencv_rgbd.so.3.4.10
main_simulation: /usr/local/lib/libopencv_saliency.so.3.4.10
main_simulation: /usr/local/lib/libopencv_stereo.so.3.4.10
main_simulation: /usr/local/lib/libopencv_structured_light.so.3.4.10
main_simulation: /usr/local/lib/libopencv_surface_matching.so.3.4.10
main_simulation: /usr/local/lib/libopencv_tracking.so.3.4.10
main_simulation: /usr/local/lib/libopencv_xfeatures2d.so.3.4.10
main_simulation: /usr/local/lib/libopencv_ximgproc.so.3.4.10
main_simulation: /usr/local/lib/libopencv_xobjdetect.so.3.4.10
main_simulation: /usr/local/lib/libopencv_xphoto.so.3.4.10
main_simulation: /usr/local/cuda-11.4/lib64/libcudart.so
main_simulation: /usr/local/lib/libopencv_cudafeatures2d.so.3.4.10
main_simulation: /usr/local/lib/libopencv_shape.so.3.4.10
main_simulation: /usr/local/lib/libopencv_cudaoptflow.so.3.4.10
main_simulation: /usr/local/lib/libopencv_cudalegacy.so.3.4.10
main_simulation: /usr/local/lib/libopencv_cudawarping.so.3.4.10
main_simulation: /usr/local/lib/libopencv_highgui.so.3.4.10
main_simulation: /usr/local/lib/libopencv_videoio.so.3.4.10
main_simulation: /usr/local/lib/libopencv_viz.so.3.4.10
main_simulation: /usr/local/lib/libopencv_phase_unwrapping.so.3.4.10
main_simulation: /usr/local/lib/libopencv_video.so.3.4.10
main_simulation: /usr/local/lib/libopencv_datasets.so.3.4.10
main_simulation: /usr/local/lib/libopencv_plot.so.3.4.10
main_simulation: /usr/local/lib/libopencv_text.so.3.4.10
main_simulation: /usr/local/lib/libopencv_dnn.so.3.4.10
main_simulation: /usr/local/lib/libopencv_ml.so.3.4.10
main_simulation: /usr/local/lib/libopencv_imgcodecs.so.3.4.10
main_simulation: /usr/local/lib/libopencv_objdetect.so.3.4.10
main_simulation: /usr/local/lib/libopencv_calib3d.so.3.4.10
main_simulation: /usr/local/lib/libopencv_features2d.so.3.4.10
main_simulation: /usr/local/lib/libopencv_flann.so.3.4.10
main_simulation: /usr/local/lib/libopencv_photo.so.3.4.10
main_simulation: /usr/local/lib/libopencv_cudaimgproc.so.3.4.10
main_simulation: /usr/local/lib/libopencv_cudafilters.so.3.4.10
main_simulation: /usr/local/lib/libopencv_cudaarithm.so.3.4.10
main_simulation: /usr/local/lib/libopencv_imgproc.so.3.4.10
main_simulation: /usr/local/lib/libopencv_core.so.3.4.10
main_simulation: /usr/local/lib/libopencv_cudev.so.3.4.10
main_simulation: CMakeFiles/main_simulation.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/fjy/Desktop/py_ws/YOLOP/toolkits/deploy/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable main_simulation"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/main_simulation.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/main_simulation.dir/build: main_simulation

.PHONY : CMakeFiles/main_simulation.dir/build

CMakeFiles/main_simulation.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/main_simulation.dir/cmake_clean.cmake
.PHONY : CMakeFiles/main_simulation.dir/clean

CMakeFiles/main_simulation.dir/depend:
	cd /home/fjy/Desktop/py_ws/YOLOP/toolkits/deploy/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/fjy/Desktop/py_ws/YOLOP/toolkits/deploy /home/fjy/Desktop/py_ws/YOLOP/toolkits/deploy /home/fjy/Desktop/py_ws/YOLOP/toolkits/deploy/cmake-build-debug /home/fjy/Desktop/py_ws/YOLOP/toolkits/deploy/cmake-build-debug /home/fjy/Desktop/py_ws/YOLOP/toolkits/deploy/cmake-build-debug/CMakeFiles/main_simulation.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/main_simulation.dir/depend

