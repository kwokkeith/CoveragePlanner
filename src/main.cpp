#include "coverage_planner.h"
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <ostream>
#include <string>
#define DENSE_PATH
#include <fstream>

#define PARAMETER_FILE_PATH "../config/params.config"
#define WAYPOINT_COORDINATE_FILE_PATH "../result/waypoints.txt"
#define EXTERNAL_POLYGON_FILE_PATH "../result/ext_polygon_coord.txt"

std::string image_path;
uint robot_width;
uint robot_height;
uint open_kernel_width;
uint open_kernel_height;
int sweep_step;
bool show_cells;
bool mouse_select_start;
uint start_x;
uint start_y;

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

int main() {
  // Load parameters from config file
  if (!LoadParameters()) {
    return EXIT_FAILURE;
  }

  // Read image to be processed
  cv::Mat img = cv::imread(image_path);

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
  // Seems to have a minimum size of (4, 4)

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

  cv::Mat poly_canvas = img.clone();
  std::vector<cv::Point> poly;
  std::vector<std::vector<cv::Point>> polys;
  for (auto &contour : contours) {
    cv::approxPolyDP(contour, poly, 3, true);
    polys.emplace_back(poly);
    poly.clear();
  }
  for (int i = 0; i < polys.size(); i++) {
    cv::drawContours(poly_canvas, polys, i, cv::Scalar(255, 0, 255));
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
  // construct adjacent graph
  //  std::map<size_t, std::set<size_t>> cell_graph;
  //  bool succeed = calculateDecompositionAdjacency(bcd_cells, &cell_graph);

  //  if(succeed){
  //      std::cout<<"cell graph was constructed"<<std::endl;
  //  }else{
  //      std::cout<<"cell graph wasn't constructed"<<std::endl;
  //  }

  //  for(auto cell_it=cell_graph.begin(); cell_it!=cell_graph.end();
  //  cell_it++){
  //      std::cout<<"cell "<<cell_it->first<<" 's neighbors: "<<std::endl;
  //      for(auto it=cell_it->second.begin(); it!=cell_it->second.end();
  //      it++){
  //          std::cout<<*it<<" ";
  //      }
  //      std::cout<<std::endl<<std::endl;
  //  }

  auto cell_graph = calculateDecompositionAdjacency(bcd_cells);
  //  for(auto& cell:cell_graph){
  //      std::cout<<"cell "<<cell.cellIndex<<" 's neighbors: "<<std::endl;
  //      for(auto& neighbor:cell.neighbor_indices){
  //          std::cout<<neighbor<<" "<<std::endl;
  //      }
  //      std::cout<<std::endl;
  //  }

  

  // Get starting point from mouse click
  Point_2 start;
  if (mouse_select_start) {
    std::cout << "Select starting point" << std::endl;
    start = getStartingPoint(img);
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

  // sweep_step (distance per step in sweep),
  // int sweep_step = 5;

  std::vector<std::vector<Point_2>> cells_sweeps;

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

  //    cv::Point prev, curr;
  //    cv::Mat poly_img_ = poly_img.clone();
  //    for(size_t i = 0; i < cells_sweeps.size(); ++i){
  //        for(size_t j = 1; j < cells_sweeps[i].size(); ++j){
  //            prev =
  //            cv::Point(CGAL::to_double(cells_sweeps[i][j-1].x()),CGAL::to_double(cells_sweeps[i][j-1].y()));
  //            curr =
  //            cv::Point(CGAL::to_double(cells_sweeps[i][j].x()),CGAL::to_double(cells_sweeps[i][j].y()));
  //            cv::line(poly_img_, prev, curr, cv::Scalar(0, 0, 255));
  //            cv::namedWindow("way",cv::WINDOW_NORMAL);
  //            cv::imshow("way", poly_img_);
  //            cv::waitKey(0);
  //        }
  //        poly_img_ = poly_img.clone();
  //    }
  //    cv::waitKey(0);

  //    for(int i = 0; i < cell_path.size(); i++){
  //        cv::drawContours(poly_img, bcd_polys, cell_path[i].cellIndex,
  //        cv::Scalar(0, 255, 255)); cv::namedWindow("path cell",
  //        cv::WINDOW_NORMAL); cv::imshow("path cell", poly_img);
  //        cv::waitKey(500);
  //        cv::drawContours(poly_img, bcd_polys, cell_path[i].cellIndex,
  //        cv::Scalar(0, 0, 255));
  //    }
  //    cv::waitKey();

  auto cell_intersections = calculateCellIntersections(bcd_cells, cell_graph);

  //    for(size_t i = 0; i < cell_intersections.size(); ++i){
  //        for(auto j = cell_intersections[i].begin(); j !=
  //        cell_intersections[i].end(); ++j){
  //            std::cout<<"cell "<<i<<" intersect with "<<"cell
  //            "<<j->first<<":
  //            "; for(auto k = j->second.begin(); k != j->second.end(); ++k){
  //                std::cout<<"("<<CGAL::to_double(k->x())<<",
  //                "<<CGAL::to_double(k->y())<<")"<<" ";
  //            }
  //            std::cout<<std::endl;
  //        }
  //        std::cout<<std::endl<<std::endl;
  //    }

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
  cv::imshow("cover", img);
  cv::waitKey();

  // Open waypoint file to write coordinates
  std::ofstream out(WAYPOINT_COORDINATE_FILE_PATH);

  for (size_t i = 1; i < way_points.size(); ++i) {
    p1 = cv::Point(CGAL::to_double(way_points[i - 1].x()),
                   CGAL::to_double(way_points[i - 1].y()));
    p2 = cv::Point(CGAL::to_double(way_points[i].x()),
                   CGAL::to_double(way_points[i].y()));
    cv::line(img, p1, p2, cv::Scalar(0, 64, 255));
    cv::namedWindow("cover", cv::WINDOW_NORMAL);
    cv::imshow("cover", img);
    //        cv::waitKey(50);
    cv::line(img, p1, p2, cv::Scalar(200, 200, 200));

    // Write waypoints to a file (to be fed as coordinates for robot)
    if (i == 1) {
      out << p1.x << " " << p1.y << std::endl;
    }
    // For all other points we will just use p2,
    // we do not pass both p1 and p2 as it would duplicate the points
    out << p2.x << " " << p2.y << std::endl;
  }
  out.close();

  cv::waitKey();
  cv::imwrite("image_result.png", img);
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
