#include "coverage_planner.h"
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <ostream>
#include <string>
#define DENSE_PATH
#include <fstream>
#include <cmath>
#include <vector>
#include <utility>

#define PARAMETER_FILE_PATH "../config/params.config"
#define WAYPOINT_COORDINATE_FILE_PATH "../result/waypoints.txt"
#define EXTERNAL_POLYGON_FILE_PATH "../result/ext_polygon_coord.txt"

std::string image_path;
uint robot_width;
uint robot_height;
uint open_kernel_width;
uint open_kernel_height;
uint dilate_kernel_width;
uint dilate_kernel_height;
int sweep_step;
bool show_cells;
bool mouse_select_start;
bool manual_orientation;
uint start_x;
uint start_y;
uint subdivision_dist;
std:: vector<cv::Point> points;
cv::Mat img_copy;

bool LoadParameters() {
  // Load parameters from config file
  std::ifstream in(PARAMETER_FILE_PATH);

  std::string param;

  while (!in.eof()) {
    in >> param;

    if (param == "IMAGE_PATH") {
      in >> image_path;
    } else if (param == "ROBOT_SIZE") {
      in >> robot_width;
      in >> robot_height;
    } else if (param == "MORPH_SIZE") {
      in >> open_kernel_width;
      in >> open_kernel_height;
    } else if (param == "OBSTACLE_INFLATION") {
      in >> dilate_kernel_width;
      in >> dilate_kernel_height;
    } else if (param == "SWEEP_STEP") {
      in >> sweep_step;
    } else if (param == "SHOW_CELLS") {
      std::string show_cells_str;
      in >> show_cells;
    } else if (param == "MOUSE_SELECT_START") {
      std::string mouse_select_start_str;
      in >> mouse_select_start;
    } else if (param == "START_POS") {
      in >> start_x;
      in >> start_y;
    } else if (param == "SUBDIVISION_DIST") { // Ensure finer waypoints for ros navstack
      in >> subdivision_dist;
    } else if (param == "MANUAL_ORIENTATION") {
      // Allow user to define the orientation for each polygon 
      in >> manual_orientation;
    }
  }
  in.close();

  // Log the loaded parameters
  std::cout << "Parameters Loaded:" << std::endl;
  std::cout << "image_path: " << image_path << std::endl;
  std::cout << "robot_width, robot_height: " << robot_width << " "
            << robot_height << std::endl;
  std::cout << "open_kernel_width, open_kernel_height: " << open_kernel_width
            << " " << open_kernel_height << std::endl;
  std::cout << "sweep_step: " << sweep_step << std::endl;
  std::cout << "show_cells: " << show_cells << std::endl;
  std::cout << "mouse_select_start: " << mouse_select_start << std::endl;

  return true;
}

void mouseCallback(int event, int x, int y, int flags, void* param) {
    if (event == cv::EVENT_LBUTTONDOWN && points.size() < 4) {
        points.push_back(cv::Point(x, y));
        std::cout << "Point" << points.size() << ": " << x << "," << y << std::endl;
        cv::circle(img_copy, cv::Point(x, y), 2, cv::Scalar(0, 0, 255), -1);
        cv::imshow("Select 4 points", img_copy);
    }
}

// Function to crop and transform the image based on selected points
std::pair<cv::Mat, cv::Point> cropAndTransform(const cv::Mat& img) {
    // Find the bounding rectangle for the selected points
    std::vector<int> x_coords, y_coords;
    for (const auto& p : points) {
        x_coords.push_back(p.x);
        y_coords.push_back(p.y);
    }

    // Determine the top-left and bottom-right corners
    int x_min = *std::min_element(x_coords.begin(), x_coords.end());
    int x_max = *std::max_element(x_coords.begin(), x_coords.end());
    int y_min = *std::min_element(y_coords.begin(), y_coords.end());
    int y_max = *std::max_element(y_coords.begin(), y_coords.end());

    cv::Mat cropped_img = img(cv::Rect(x_min, y_min, x_max - x_min, y_max - y_min)).clone();

    return std::make_pair(cropped_img, cv::Point(x_min,y_min));
}

int main() {
  // Load parameters from config file
  if (!LoadParameters()) {
    return EXIT_FAILURE;
  }

  // Read image to be processed
  cv::Mat original_img = cv::imread(image_path);
  cv::Mat img = cv::imread(image_path);

    img_copy = img.clone();
  cv::imshow("Select 4 points", img_copy);

  //Set mouse callback
  cv::setMouseCallback("Select 4 points", mouseCallback, nullptr);
  cv::waitKey(0);

  auto crop = cropAndTransform(img);
  cv::Mat result = crop.first;
  cv::Point top_left=crop.second;
  if (!result.empty()) {
      cv::imshow("Cropped Image", result);
      cv::waitKey(0);
  }

  img=result;

  std::cout << "Read map" << std::endl;
  std::cout << "Pre-Processing map image" << std::endl;

  // Image Pre-processing (Reduce noise of image)
  cv::Mat gray;
  cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

  cv::Mat img_ = gray.clone();

  // Takes every pixel and declare to be only black or white
  // Binarizes the image (Making contrast clear)
  cv::threshold(img_, img_, 250, 255, 0);

  // Robot radius (Size) is defined here
  std::cout << "--Applying morphological operations onto image--" << std::endl;

  // Makes kernel in an ellipse shape of a certain size
  // And runs through the entire image and sets each kernel batch, all pixels
  // in the kernel, to the minimum value of that kernel (0 for black)
  cv::Mat erode_kernel = cv::getStructuringElement(
      cv::MORPH_ELLIPSE, cv::Size(robot_width, robot_height),
      cv::Point(-1, -1)); // size: robot radius
  cv::morphologyEx(img_, img_, cv::MORPH_ERODE, erode_kernel);
  std::cout << "Erosion Kernel for robot size applied" << std::endl;

  //  Applied after the above erosion kernel to enhance image
  //  Can use MORPH_RECT, MORPH_ELLIPSE
  cv::Mat open_kernel = cv::getStructuringElement(
      cv::MORPH_RECT, cv::Size(open_kernel_width, open_kernel_height),
      cv::Point(-1, -1));
  cv::morphologyEx(img_, img_, cv::MORPH_OPEN, open_kernel);
  std::cout << "Open Kernel applied" << std::endl;

  // To inflate the obstacles on the map
  // Invert the image so that black walls become white
  std::cout << "--Inverting the image to apply dilation on black walls--" << std::endl;
  cv::bitwise_not(img_, img_);  // Invert the image

  // Inflate the walls (now white) by dilating them
  std::cout << "--Inflating walls by dilating the obstacles--" << std::endl;
  cv::Mat dilation_kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(dilate_kernel_width, dilate_kernel_height), cv::Point(-1, -1));
  cv::dilate(img_, img_, dilation_kernel);
  std::cout << "Dilation applied to inflate the walls" << std::endl;

  // Invert the image back to original (black walls)
  std::cout << "--Reverting the image back to original (black walls)--" << std::endl;
  cv::bitwise_not(img_, img_);  // Invert the image back

  // TODO: SECOND RUN OF Preprocessing if needed

  /*// Applied after the above erosion kernel to enhance image*/
  /*// Can use MORPH_RECT, MORPH_ELLIPSE*/
  /*open_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(10, 10),*/
  /*                                        cv::Point(-1, -1));*/
  /*cv::morphologyEx(img_, img_, cv::MORPH_OPEN, open_kernel);*/
  /*std::cout << "Open Kernel applied" << std::endl;*/
  /**/

  cv::imshow("preprocess", img_);
  cv::waitKey();
  cv::imwrite("preprocess_img.png", img_);

  std::cout << std::string(50, '-') << std::endl;

  std::vector<std::vector<cv::Point>> cnts;
  std::vector<cv::Vec4i> hierarchy; // index: next, prev, first_child, parent
  cv::findContours(img_, cnts, hierarchy, cv::RETR_TREE,
                    cv::CHAIN_APPROX_SIMPLE);

  std::vector<int> cnt_indices(cnts.size());
  std::iota(cnt_indices.begin(), cnt_indices.end(), 0);
  std::sort(cnt_indices.begin(), cnt_indices.end(), [&cnts](int lhs, int rhs) {
    return cv::contourArea(cnts[lhs]) > cv::contourArea(cnts[rhs]);
  });
  int ext_cnt_idx = cnt_indices.front();

  cv::Mat cnt_canvas = img.clone();
  cv::drawContours(cnt_canvas, cnts, ext_cnt_idx, cv::Scalar(0, 0, 255));
  std::vector<std::vector<cv::Point>> contours;
  contours.emplace_back(cnts[ext_cnt_idx]);

  // find all the contours of obstacle
  for (int i = 0; i < hierarchy.size(); i++) {
    if (hierarchy[i][3] == ext_cnt_idx) { // parent contour's index equals to
                                          // external contour's index
      contours.emplace_back(cnts[i]);
      cv::drawContours(cnt_canvas, cnts, i, cv::Scalar(255, 0, 0));
    }
  }
  //    cv::imshow("contours", cnt_canvas);

  cv::Mat cnt_img = cv::Mat(img.rows, img.cols, CV_8UC3);
  cnt_img.setTo(255);
  for (int i = 0; i < contours.size(); i++) {
    cv::drawContours(cnt_img, contours, i, cv::Scalar(0, 0, 0));
  }
  //    cv::imshow("only contours", cnt_img);

  cv::Mat poly_canvas = original_img.clone();
  std::vector<cv::Point> poly;
  std::vector<std::vector<cv::Point>> polys;
  for (auto &contour : contours) {
    cv::approxPolyDP(contour, poly, 3, true);
    std::vector<cv::Point> translated_poly;
    for (const auto& point : poly) {
        translated_poly.push_back(point + top_left);
    }
    polys.emplace_back(translated_poly);
    poly.clear();
  }
  for (int i = 0; i < polys.size(); i++) {
    cv::drawContours(poly_canvas, std::vector<std::vector<cv::Point>>{polys[i]}, -1, cv::Scalar(255, 0, 255));
  }

  cv::imshow("polygons", poly_canvas);
  cv::waitKey();

  cv::Mat poly_img = cv::Mat(img.rows, img.cols, CV_8UC3);
  poly_img.setTo(255);
  for (int i = 0; i < polys.size(); i++) {
    cv::drawContours(poly_img, polys, i, cv::Scalar(0, 0, 0));
  }

  // Extract the vertices of the external Polygons

  //    cv::imshow("only polygons", poly_img);
  //    cv::waitKey();/your-docusaurus-site.example.com

  // compute main direction

  // [0,180)
  std::vector<int> line_deg_histogram(180);
  double line_len; // weight
  double line_deg;
  int line_deg_idx;

  cv::Mat line_canvas = img.clone();
  auto ext_poly = polys.front();
  ext_poly.emplace_back(ext_poly.front());
  for (int i = 1; i < ext_poly.size(); i++) {
    line_len = std::sqrt(std::pow((ext_poly[i].x - ext_poly[i - 1].x), 2) +
                         std::pow((ext_poly[i].y - ext_poly[i - 1].y), 2));
    // y-axis towards up, x-axis towards right, theta is from x-axis to y-axis
    line_deg = std::round(atan2(-(ext_poly[i].y - ext_poly[i - 1].y),
                                (ext_poly[i].x) - ext_poly[i - 1].x) /
                          M_PI * 180.0);        // atan2: (-180, 180]
    line_deg_idx = (int(line_deg) + 180) % 180; // [0, 180)
    line_deg_histogram[line_deg_idx] += int(line_len);

    //          std::cout<<"deg: "<<line_deg_idx<<std::endl;
    //          cv::line(line_canvas, ext_poly[i], ext_poly[i-1],
    //          cv::Scalar(255,255,0)); cv::imshow("lines",line_canvas);
    //          cv::waitKey();
  }

  //    cv::waitKey();

  auto it =
      std::max_element(line_deg_histogram.begin(), line_deg_histogram.end());
  int main_deg = (it - line_deg_histogram.begin());
  std::cout << "main deg: " << main_deg << std::endl;

  // file stream to write external polygon vertices to
  std::ofstream out_ext_poly(EXTERNAL_POLYGON_FILE_PATH);

  // construct polygon with holes

  std::vector<cv::Point> outer_poly = polys.front();
  polys.erase(polys.begin());
  std::vector<std::vector<cv::Point>> inner_polys = polys;

  Polygon_2 outer_polygon;
  out_ext_poly << outer_poly.size() << std::endl;

  for (const auto &point : outer_poly) {
    outer_polygon.push_back(Point_2(point.x, point.y));
    out_ext_poly << point.x << " " << point.y << std::endl;
  }

  // close the file stream
  out_ext_poly.close();

  int num_holes = inner_polys.size();
  std::vector<Polygon_2> holes(num_holes);
  for (int i = 0; i < inner_polys.size(); i++) {
    for (const auto &point : inner_polys[i]) {
      holes[i].push_back(Point_2(point.x, point.y));
    }
  }

  PolygonWithHoles pwh(outer_polygon, holes.begin(), holes.end());

  std::cout << "constructed polygons" << std::endl;

  // cell decomposition
  // TODO: Bottleneck for memory space

  std::cout << "Performing cell decomposition" << std::endl;

  // To measure the time it takes to execute cell decomposition
  auto start_time = std::chrono::high_resolution_clock::now();
  std::vector<Polygon_2> bcd_cells;

  //    polygon_coverage_planning::computeBestTCDFromPolygonWithHoles(pwh,
  //    &bcd_cells);
  polygon_coverage_planning::computeBestBCDFromPolygonWithHoles(pwh,
                                                                &bcd_cells);

  auto end_time = std::chrono::high_resolution_clock::now();
  auto execution_time = std::chrono::duration_cast<std::chrono::microseconds>(
      end_time - start_time);
  long double ms = execution_time.count();
  long double s = ms / 1000000;
  std::cout << "Cell decomposition complete in " << s << "s" << std::endl;

  // test decomposition
  if (show_cells) {
    std::vector<std::vector<cv::Point>> bcd_polys;
    std::vector<cv::Point> bcd_poly;

    for (const auto &cell : bcd_cells) {
      for (int i = 0; i < cell.size(); i++) {
        bcd_poly.emplace_back(cv::Point(CGAL::to_double(cell[i].x()),
                                        CGAL::to_double(cell[i].y())));
      }
      bcd_polys.emplace_back(bcd_poly);
      bcd_poly.clear();
    }

    for (int i = 0; i < bcd_polys.size(); i++) {
      cv::drawContours(poly_img, bcd_polys, i, cv::Scalar(255, 0, 255));
      cv::imshow("bcd", poly_img);
      cv::waitKey();
    }
    cv::imshow("bcd", poly_img);
    cv::waitKey();
  }

  auto cell_graph = calculateDecompositionAdjacency(bcd_cells);

  // Get starting point from mouse click
  Point_2 start;
  if (mouse_select_start) {
    std::cout << "Select starting point" << std::endl;
    start = getStartingPoint(original_img);
  } else {
    start = Point_2(start_x, start_y);
    std::cout << "Starting point configured: (" << start.x() << ", " << start.y() << ")" << std::endl;
  }

  int starting_cell_idx = getCellIndexOfPoint(bcd_cells, start);
  auto cell_idx_path = getTravellingPath(cell_graph, starting_cell_idx) ;
  std::cout << "path length: " << cell_idx_path.size() << std::endl;
  std::cout << "start";
  for (auto &cell_idx : cell_idx_path) {
    std::cout << "->" << cell_idx;
  }
  std::cout << std::endl;

  // sweep_step (distance per step in sweep),
  // int sweep_step = 5;
  std::vector<std::vector<Point_2>> cells_sweeps;
  
  if (manual_orientation) {
    // Store user-defined angles for sweep direction
    std::vector<double> polygon_sweep_directions;

    // Create a named window to show the polygon
    cv::namedWindow("Selected Polygon", cv::WINDOW_NORMAL);

    for (size_t i = 0; i < bcd_cells.size(); ++i) {
      // Display the polygon to the user using OpenCV as before.
      cv::Mat img_copy = original_img.clone();  // Create a copy of the image
      std::vector<std::vector<cv::Point>> poly_contours;

      // Extract the points of the current polygon
      std::vector<cv::Point> current_polygon;
      for (int j = 0; j < bcd_cells[i].size(); ++j) {
          current_polygon.push_back(cv::Point(CGAL::to_double(bcd_cells[i][j].x()), 
                                              CGAL::to_double(bcd_cells[i][j].y())));
      }
      poly_contours.push_back(current_polygon);
      
      // Draw the current polygon on the copied image
      cv::drawContours(img_copy, poly_contours, -1, cv::Scalar(0, 255, 0), 2);
      cv::imshow("Polygon Selection", img_copy);
      cv::waitKey(500);  // Allow the user to see the polygon

      // Compute best sweep direction
      Direction_2 best_sweep_dir;
      double min_altitude = polygon_coverage_planning::findBestSweepDir(bcd_cells[i], &best_sweep_dir);

      // Ensure valid sweep direction
      if (std::isnan(CGAL::to_double(best_sweep_dir.dx())) || std::isnan(CGAL::to_double(best_sweep_dir.dy()))) {
        std::cerr << "Invalid best sweep direction detected for polygon " << i << std::endl;
        continue;  // Skip this polygon if sweep direction is invalid
      }

      // Convert best sweep direction to degrees
      double best_sweep_angle = std::atan2(CGAL::to_double(best_sweep_dir.dy()), CGAL::to_double(best_sweep_dir.dx())) * 180.0 / M_PI;
      std::cout << "Best sweep direction for polygon " << i + 1 << " is: " << best_sweep_angle << " degrees" << std::endl;

      // Prompt user to enter custom angle or use the best one
      std::cout << "Enter sweep direction (degrees) for polygon " << i + 1
                << " (or press Enter to use best sweep direction): ";

      // Capture the user input, expecting a newline after entry
      std::string input;
      std::getline(std::cin, input);  // Get the user input for the sweep direction

      double user_angle;
      try {
        if (input.empty()) {
            user_angle = best_sweep_angle;  // Use best sweep direction if no input
        } else {
            user_angle = std::stod(input);  // Use user input
        }
      } catch (const std::invalid_argument& e) {
        std::cerr << "Invalid input for angle. Using best sweep angle for polygon " << i << std::endl;
        user_angle = best_sweep_angle;  // Fallback to best sweep angle
      }

      polygon_sweep_directions.push_back(user_angle);
    }

    // Execute sweep for each polygon using the user-defined or best direction
    for (size_t i = 0; i < bcd_cells.size(); ++i) {
      std::vector<Point_2> cell_sweep;

      // Convert the user-defined angle to radians
      double angle_in_radians = polygon_sweep_directions[i] * (M_PI / 180.0);

      // Ensure valid direction vectors
      if (std::isnan(std::cos(angle_in_radians)) || std::isnan(std::sin(angle_in_radians))) {
        std::cerr << "Invalid sweep direction for polygon " << i << ". Using default direction." << std::endl;
        continue;  // Skip this polygon if the direction is invalid
      }

      // Create direction from the angle
      Direction_2 user_defined_dir(std::cos(angle_in_radians), std::sin(angle_in_radians));

      // Perform sweep using the user-defined direction
      polygon_coverage_planning::visibility_graph::VisibilityGraph vis_graph(bcd_cells[i]);

      try {
        polygon_coverage_planning::computeSweep(bcd_cells[i], vis_graph, sweep_step, user_defined_dir, true, &cell_sweep);

        if (cell_sweep.empty()) {
            std::cerr << "Warning: Sweep for polygon " << i + 1 << " returned no points." << std::endl;
        } else {
            std::cout << "Successfully constructed sweep for polygon " << i + 1 << std::endl;
        }

        cells_sweeps.emplace_back(cell_sweep);

      } catch (const std::exception& e) {
        std::cerr << "Error constructing sweep for polygon " << i + 1 << ": " << e.what() << std::endl;
      }
    }
  } else {
    for (size_t i = 0; i < bcd_cells.size(); ++i) {
      // Compute all cluster sweeps.
      std::vector<Point_2> cell_sweep;
      Direction_2 best_dir;
      polygon_coverage_planning::findBestSweepDir(bcd_cells[i], &best_dir);
      polygon_coverage_planning::visibility_graph::VisibilityGraph vis_graph(
          bcd_cells[i]);

      bool counter_clockwise = true;
      polygon_coverage_planning::computeSweep(bcd_cells[i], vis_graph, sweep_step,
                                              best_dir, counter_clockwise,
                                              &cell_sweep);
      cells_sweeps.emplace_back(cell_sweep);
    }
  }

  auto cell_intersections = calculateCellIntersections(bcd_cells, cell_graph);

  std::vector<Point_2> way_points;

#ifdef DENSE_PATH
  Point_2 point = start;
  std::list<Point_2> next_candidates;
  Point_2 next_point;
  std::vector<Point_2> shortest_path;

  if (doReverseNextSweep(start, cells_sweeps[cell_idx_path.front()])) {
    shortest_path = getShortestPath(bcd_cells[cell_idx_path.front()], start,
                                    cells_sweeps[cell_idx_path.front()].back());
    way_points.insert(way_points.end(), shortest_path.begin(),
                      std::prev(shortest_path.end()));
  } else {
    shortest_path =
        getShortestPath(bcd_cells[cell_idx_path.front()], start,
                        cells_sweeps[cell_idx_path.front()].front());
    way_points.insert(way_points.end(), shortest_path.begin(),
                      std::prev(shortest_path.end()));
  }

  point = way_points.back();

  for (size_t i = 0; i < cell_idx_path.size(); ++i) {
    // has been cleaned?
    if (!cell_graph[cell_idx_path[i]].isCleaned) {
      // need to reverse?
      if (doReverseNextSweep(point, cells_sweeps[cell_idx_path[i]])) {
        way_points.insert(way_points.end(),
                          cells_sweeps[cell_idx_path[i]].rbegin(),
                          cells_sweeps[cell_idx_path[i]].rend());
      } else {
        way_points.insert(way_points.end(),
                          cells_sweeps[cell_idx_path[i]].begin(),
                          cells_sweeps[cell_idx_path[i]].end());
      }
      // now cleaned
      cell_graph[cell_idx_path[i]].isCleaned = true;
      // update current point
      point = way_points.back();
      // find shortest path to next cell
      if ((i + 1) < cell_idx_path.size()) {
        next_candidates =
            cell_intersections[cell_idx_path[i]][cell_idx_path[i + 1]];
        if (doReverseNextSweep(point, cells_sweeps[cell_idx_path[i + 1]])) {
          next_point =
              findNextGoal(point, cells_sweeps[cell_idx_path[i + 1]].back(),
                           next_candidates);
          shortest_path =
              getShortestPath(bcd_cells[cell_idx_path[i]], point, next_point);
          way_points.insert(way_points.end(), std::next(shortest_path.begin()),
                            std::prev(shortest_path.end()));
          shortest_path =
              getShortestPath(bcd_cells[cell_idx_path[i + 1]], next_point,
                              cells_sweeps[cell_idx_path[i + 1]].back());
        } else {
          next_point =
              findNextGoal(point, cells_sweeps[cell_idx_path[i + 1]].front(),
                           next_candidates);
          shortest_path =
              getShortestPath(bcd_cells[cell_idx_path[i]], point, next_point);
          way_points.insert(way_points.end(), std::next(shortest_path.begin()),
                            std::prev(shortest_path.end()));
          shortest_path =
              getShortestPath(bcd_cells[cell_idx_path[i + 1]], next_point,
                              cells_sweeps[cell_idx_path[i + 1]].front());
        }
        way_points.insert(way_points.end(), shortest_path.begin(),
                          std::prev(shortest_path.end()));
        point = way_points.back();
      }
    } else {
      shortest_path = getShortestPath(bcd_cells[cell_idx_path[i]],
                                      cells_sweeps[cell_idx_path[i]].front(),
                                      cells_sweeps[cell_idx_path[i]].back());
      if (doReverseNextSweep(point, cells_sweeps[cell_idx_path[i]])) {
        way_points.insert(way_points.end(), shortest_path.rbegin(),
                          shortest_path.rend());
      } else {
        way_points.insert(way_points.end(), shortest_path.begin(),
                          shortest_path.end());
      }
      point = way_points.back();

      if ((i + 1) < cell_idx_path.size()) {
        next_candidates =
            cell_intersections[cell_idx_path[i]][cell_idx_path[i + 1]];
        if (doReverseNextSweep(point, cells_sweeps[cell_idx_path[i + 1]])) {
          next_point =
              findNextGoal(point, cells_sweeps[cell_idx_path[i + 1]].back(),
                           next_candidates);
          shortest_path =
              getShortestPath(bcd_cells[cell_idx_path[i]], point, next_point);
          way_points.insert(way_points.end(), std::next(shortest_path.begin()),
                            std::prev(shortest_path.end()));
          shortest_path =
              getShortestPath(bcd_cells[cell_idx_path[i + 1]], next_point,
                              cells_sweeps[cell_idx_path[i + 1]].back());
        } else {
          next_point =
              findNextGoal(point, cells_sweeps[cell_idx_path[i + 1]].front(),
                           next_candidates);
          shortest_path =
              getShortestPath(bcd_cells[cell_idx_path[i]], point, next_point);
          way_points.insert(way_points.end(), std::next(shortest_path.begin()),
                            std::prev(shortest_path.end()));
          shortest_path =
              getShortestPath(bcd_cells[cell_idx_path[i + 1]], next_point,
                              cells_sweeps[cell_idx_path[i + 1]].front());
        }
        way_points.insert(way_points.end(), shortest_path.begin(),
                          std::prev(shortest_path.end()));
        point = way_points.back();
      }
    }
  }

  cv::Point p1, p2;
  cv::namedWindow("cover", cv::WINDOW_NORMAL);
  cv::imshow("cover", original_img);
  cv::waitKey();

  // Open waypoint file to write coordinates
  std::ofstream out(WAYPOINT_COORDINATE_FILE_PATH);

for (size_t i = 1; i < way_points.size(); ++i) {
    cv::Point p1 = cv::Point(std::round(CGAL::to_double(way_points[i - 1].x())),
                                 std::round(CGAL::to_double(way_points[i - 1].y())));
    cv::Point p2 = cv::Point(std::round(CGAL::to_double(way_points[i].x())),
                                 std::round(CGAL::to_double(way_points[i].y())));

    // Ensure p1 and p2 are within valid bounds
    if (std::isnan(p1.x) || std::isnan(p1.y) || std::isnan(p2.x) || std::isnan(p2.y)) {
        std::cerr << "Invalid points detected: p1(" << p1.x << ", " << p1.y 
                  << ") p2(" << p2.x << ", " << p2.y << ")" << std::endl;
        continue;  // Skip this iteration if invalid points are found
    }

    std::vector<cv::Point> newPoints;

    // get number of subdivisions
    if (subdivision_dist > 0) {
      double euclidean_dist = std::sqrt(std::pow(p2.x - p1.x, 2) + std::pow(p2.y - p1.y, 2));
      // std::cout << "euclidean_dist: " << euclidean_dist << std::endl;
      double number_of_subdivisions = std::round(euclidean_dist / subdivision_dist);
      // std::cout << "# of subdivisions: " << number_of_subdivisions << std::endl;


      if (number_of_subdivisions > 0) {
        // Compute the step increments based on the number of subdivisions
        double stepX = (p2.x - p1.x) / static_cast<double>(number_of_subdivisions + 1);
        double stepY = (p2.y - p1.y) / static_cast<double>(number_of_subdivisions + 1);

        // Add intermediate points
        for (int i = 1; i <= number_of_subdivisions; ++i) {
            cv::Point intermediatePoint;
            intermediatePoint.x = std::round(p1.x + stepX * i);
            intermediatePoint.y = std::round(p1.y + stepY * i);
            newPoints.push_back(intermediatePoint);
        }

        // Draw the initial line segment from p1 to the first interpolated point
        cv::line(original_img, p1, newPoints[0], cv::Scalar(0, 64, 255));
        for (size_t j = 0; j < newPoints.size() - 1; ++j) {
            cv::line(original_img, newPoints[j], newPoints[j + 1], cv::Scalar(0, 64, 255));  // Draw between subdivided points
        }
        cv::line(original_img, newPoints.back(), p2, cv::Scalar(0, 64, 255));  // Draw final segment to p2
      } else {
        // If subdivisions == 0, directly draw the line between p1 and p2
          cv::line(original_img, p1, p2, cv::Scalar(0, 64, 255));
      }
    }

    cv::namedWindow("cover", cv::WINDOW_NORMAL);
    cv::imshow("cover", original_img);
    //        cv::waitKey(50);
    cv::line(original_img, p1, p2, cv::Scalar(200, 200, 200));

    // Write waypoints to a file (to be fed as coordinates for robot)
    if (i == 1) {
        out << p1.x << " " << p1.y << std::endl;
    }
    for (const auto& point : newPoints) {
        out << point.x << " " << point.y << std::endl;
    }

    // For all other points we will just use p2,
    // we do not pass both p1 and p2 as it would duplicate the points
    out << p2.x << " " << p2.y << std::endl;
  }
  out.close();

  cv::waitKey();
  cv::imwrite("image_result.png", original_img);
#else

  cv::Point p1, p2;
  cv::Mat temp_img;
  cv::namedWindow("cover", cv::WINDOW_NORMAL);
  cv::imshow("cover", img);
  cv::waitKey();

  Point_2 point = start;
  way_points.emplace_back(point);

  for (auto &idx : cell_idx_path) {
    if (!cell_graph[idx].isCleaned) {
      if (doReverseNextSweep(point, cells_sweeps[idx])) {
        way_points.insert(way_points.end(), cells_sweeps[idx].rbegin(),
                          cells_sweeps[idx].rend());

        temp_img = img.clone();
        cv::line(
            img,
            cv::Point(CGAL::to_double(point.x()), CGAL::to_double(point.y())),
            cv::Point(CGAL::to_double(cells_sweeps[idx].back().x()),
                      CGAL::to_double(cells_sweeps[idx].back().y())),
            cv::Scalar(255, 0, 0), 1);
        cv::namedWindow("cover", cv::WINDOW_NORMAL);
        cv::imshow("cover", img);
        //                cv::waitKey(500);
        img = temp_img.clone();

        for (size_t i = (cells_sweeps[idx].size() - 1); i > 0; --i) {
          p1 = cv::Point(CGAL::to_double(cells_sweeps[idx][i].x()),
                         CGAL::to_double(cells_sweeps[idx][i].y()));
          p2 = cv::Point(CGAL::to_double(cells_sweeps[idx][i - 1].x()),
                         CGAL::to_double(cells_sweeps[idx][i - 1].y()));
          cv::line(img, p1, p2, cv::Scalar(0, 64, 255));
          cv::namedWindow("cover", cv::WINDOW_NORMAL);
          cv::imshow("cover", img);
          //                    cv::waitKey(50);
          cv::line(img, p1, p2, cv::Scalar(200, 200, 200));
        }

      } else {
        way_points.insert(way_points.end(), cells_sweeps[idx].begin(),
                          cells_sweeps[idx].end());

        temp_img = img.clone();
        cv::line(
            img,
            cv::Point(CGAL::to_double(point.x()), CGAL::to_double(point.y())),
            cv::Point(CGAL::to_double(cells_sweeps[idx].front().x()),
                      CGAL::to_double(cells_sweeps[idx].front().y())),
            cv::Scalar(255, 0, 0), 1);
        cv::namedWindow("cover", cv::WINDOW_NORMAL);
        cv::imshow("cover", img);
        //                cv::waitKey(500);
        img = temp_img.clone();

        for (size_t i = 1; i < cells_sweeps[idx].size(); ++i) {
          p1 = cv::Point(CGAL::to_double(cells_sweeps[idx][i - 1].x()),
                         CGAL::to_double(cells_sweeps[idx][i - 1].y()));
          p2 = cv::Point(CGAL::to_double(cells_sweeps[idx][i].x()),
                         CGAL::to_double(cells_sweeps[idx][i].y()));
          cv::line(img, p1, p2, cv::Scalar(0, 64, 255));
          cv::namedWindow("cover", cv::WINDOW_NORMAL);
          cv::imshow("cover", img);
          //                    cv::waitKey(50);
          cv::line(img, p1, p2, cv::Scalar(200, 200, 200));
        }
      }

      cell_graph[idx].isCleaned = true;
      point = way_points.back();
    }
  }

  cv::waitKey(1000);

#endif

  return 0;
}
