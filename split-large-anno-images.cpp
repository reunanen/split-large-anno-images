#include <dlib/dir_nav/dir_nav_extensions.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include "cxxopts/include/cxxopts.hpp"
#include "tiling/opencv-wrapper.h"

int main(int argc, char** argv)
{
    if (argc == 1) {
        std::cout << "Usage: " << std::endl;
        std::cout << "> split-large-anno-images -i /path/to/images -o /path/to/results" << std::endl;
        return 1;
    }

    cxxopts::Options options("split-large-anno-images", "Split large training images, so that generating mini-batches is more convenient");

    options.add_options()
        ("i,input-directory", "The directory where to search for input files", cxxopts::value<std::string>())
        ("o,output-directory", "The directory where to write the output", cxxopts::value<std::string>())
        //("i,interpolation-method", "Try 0 for nearest neighbor, 1 for bilinear, 2 for bicubic, 3 for pixel area, or 4 for Lanczos", cxxopts::value<int>()->default_value("1"))
        ("w,max-tile-width", "Max tile height", cxxopts::value<int>()->default_value("1024"))
        ("h,max-tile-height", "Max tile width", cxxopts::value<int>()->default_value("1024"))
        ("x,overlap-x", "Overlap in X", cxxopts::value<int>()->default_value("257"))
        ("y,overlap-y", "Overlap in Y", cxxopts::value<int>()->default_value("257"))
        ;

    try {
        options.parse(argc, argv);

        cxxopts::check_required(options, { "input-directory", "output-directory" });

        const std::string input_directory = options["input-directory"].as<std::string>();
        const std::string output_directory = options["output-directory"].as<std::string>();
        //const double scaling_factor = options["scaling-factor"].as<double>();
        //const int interpolation_method = options["interpolation-method"].as<int>();

        tiling::parameters tiling_parameters;
        tiling_parameters.max_tile_width = options["max-tile-width"].as<int>();
        tiling_parameters.max_tile_height = options["max-tile-height"].as<int>();
        tiling_parameters.overlap_x = options["overlap-x"].as<int>();
        tiling_parameters.overlap_y = options["overlap-y"].as<int>();

        std::cout << "Input directory  : " << input_directory << std::endl;
        std::cout << "Output directory : " << output_directory << std::endl;

        if (input_directory == output_directory) {
            throw std::runtime_error("The input directory shouldn't equal the output directory");
        }

        const std::vector<dlib::file> files = dlib::get_files_in_directory_tree(input_directory,
            [](const dlib::file& name) {
                if (dlib::match_ending("_mask.png")(name)) {
                    return false;
                }
                if (dlib::match_ending("_result.png")(name)) {
                    return false;
                }
                return dlib::match_ending(".jpeg")(name)
                    || dlib::match_ending(".jpg")(name)
                    || dlib::match_ending(".png")(name);
            });

        std::cout << "Found " << files.size() << " files, now splitting ..." << std::endl;

        const std::string mask_file_suffix = "_mask.png";

        for (const auto& file : files) {

            const std::string& full_name = file.full_name();
            const std::string& name = file.name();

            const std::string mask_file_full_name = file.full_name() + mask_file_suffix;
            const std::string mask_file_name = file.name() + mask_file_suffix;

            std::cout << "Processing " << full_name;

            const cv::Mat mask_image = cv::imread(mask_file_full_name, cv::IMREAD_UNCHANGED);

            if (!mask_image.empty()) {

                const cv::Mat image = cv::imread(full_name, cv::IMREAD_UNCHANGED);

                if (!image.data) {
                    std::cout << " - unable to read image, skipping...";
                }
                else {
                    std::cout
                        << ", width = " << image.cols
                        << ", height = " << image.rows
                        << ", channels = " << image.channels()
                        << ", type = 0x" << std::hex << image.type();

                    DLIB_CASSERT(image.size() == mask_image.size());

                    std::vector<tiling::opencv_tile> tiles = tiling::get_tiles(image.cols, image.rows, tiling_parameters);

                    std::cout << ", tiles = " << std::dec << tiles.size();

                    const auto dot_pos = name.find('.');
                    DLIB_CASSERT(dot_pos != std::string::npos);

                    const auto base_name = name.substr(0, dot_pos);
                    const auto extension = name.substr(dot_pos);

                    const auto dot_pos_mask = mask_file_name.find('.');
                    DLIB_CASSERT(dot_pos_mask != std::string::npos);

                    const auto mask_base_name = mask_file_name.substr(0, dot_pos_mask);
                    const auto mask_extension = mask_file_name.substr(dot_pos_mask);

                    DLIB_CASSERT(base_name == mask_base_name);

                    int i = 0;

                    for (const auto& tile : tiles) {
                        const cv::Mat m = mask_image(tile.full_rect);

                        const cv::Scalar mean = cv::mean(m);

                        if (mean[0] > 0 || mean[1] > 0 || mean[2] > 0 || mean[3] > 0) {
                            const cv::Mat t = image(tile.full_rect);

                            DLIB_CASSERT(t.size() == m.size());

                            std::ostringstream output_filename;
                            output_filename << output_directory << "/" << base_name << "_" << i << extension;
                            cv::imwrite(output_filename.str(), t);

                            std::ostringstream mask_output_filename;
                            mask_output_filename << output_directory << "/" << mask_base_name << "_" << i << mask_extension;
                            cv::imwrite(mask_output_filename.str(), m);

                            ++i;
                        }
                    }
                }
            }
            else {
                std::cout << " - unable to read mask, skipping...";
            }
            std::cout << std::endl;
        }

        return 0;
    }
    catch (std::exception& e) {
        std::cerr << std::endl << "Error: " << e.what() << std::endl;
        std::cerr << std::endl << options.help() << std::endl;
        return 255;
    }
}
