/*
 * polygon_coverage_planning implements algorithms for coverage planning in
 * general polygons with holes. Copyright (C) 2019, Rik Bähnemann, Autonomous
 * Systems Lab, ETH Zürich
 *
 * This program is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "decomposition.h"
#include "bcd.h"
#include "tcd.h"
#include "weakly_monotone.h"
#include <thread>
#include <atomic>

namespace polygon_coverage_planning {

std::vector<Direction_2> findEdgeDirections(const PolygonWithHoles &pwh) {
  // Get all possible polygon directions.
  std::vector<Direction_2> directions;
  for (size_t i = 0; i < pwh.outer_boundary().size(); ++i) {
    directions.push_back(pwh.outer_boundary().edge(i).direction());
  }
  
  for (PolygonWithHoles::Hole_const_iterator hit = pwh.holes_begin();
       hit != pwh.holes_end(); ++hit) {
    for (size_t i = 0; i < hit->size(); i++) {
      directions.push_back(hit->edge(i).direction());
    }
  }

  // Remove redundant directions.
  std::set<size_t> to_remove;
  for (size_t i = 0; i < directions.size() - 1; ++i) {
    for (size_t j = i + 1;j < directions.size(); ++j) {
      if (CGAL::orientation(directions[i].vector(), directions[j].vector()) ==
          CGAL::COLLINEAR)
        to_remove.insert(j);
    }
  }
  for (std::set<size_t>::reverse_iterator rit = to_remove.rbegin();
      rit != to_remove.rend(); ++rit) {
    directions.erase(std::next(directions.begin(), *rit));
  }

  // Add opposite directions.
  std::vector<Direction_2> temp_directions = directions;
  for (size_t i = 0; i < temp_directions.size(); ++i) {
    directions.push_back(-temp_directions[i]);
  }

  return directions;
}

std::vector<Direction_2> findPerpEdgeDirections(const PolygonWithHoles &pwh) {
  std::vector<Direction_2> directions = findEdgeDirections(pwh);
  for (auto &d : directions) {
    d = Direction_2(-d.dy(), d.dx());
  }

  return directions;
}

double findBestSweepDir(const Polygon_2 &cell, Direction_2 *best_dir) {
  // Get all sweepable edges.
  PolygonWithHoles pwh(cell);
  std::vector<Direction_2> edge_dirs = getAllSweepableEdgeDirections(cell);

  // Find minimum altitude.
  double min_altitude = std::numeric_limits<double>::max();
  for (const auto &dir : edge_dirs) {
    auto s = findSouth(cell, Line_2(Point_2(CGAL::ORIGIN), dir));
    auto n = findNorth(cell, Line_2(Point_2(CGAL::ORIGIN), dir));
    auto orthogonal_vec =
        dir.vector().perpendicular(CGAL::Orientation::POSITIVE);
    Line_2 line_through_n(*n, orthogonal_vec.direction());
    auto s_proj = line_through_n.projection(*s);
    double altitude =
        std::sqrt(CGAL::to_double(CGAL::squared_distance(*n, s_proj)));

    if (altitude < min_altitude) {
      min_altitude = altitude;
      if (best_dir)
        *best_dir = dir;
    }
  }

  return min_altitude;
}

// TODO: This function uses too much memory
bool computeBestBCDFromPolygonWithHoles(const PolygonWithHoles &pwh,
                                        std::vector<Polygon_2> *bcd_polygons) {
  bcd_polygons->clear();
  std::atomic<double> min_altitude_sum{std::numeric_limits<double>::max()};
  std::atomic_flag update_flag = ATOMIC_FLAG_INIT;
  std::vector<Polygon_2> best_bcd_polygons;

  // Get all possible decomposition directions.
  std::vector<Direction_2> directions = findPerpEdgeDirections(pwh);

  // Get number of CPU cores
  unsigned int num_threads = std::thread::hardware_concurrency();
  std::cout << "Number of threads detected: " << num_threads << std::endl;
  if (num_threads == 0) num_threads = 1; // Default

  // If directions.size() < num_threads then spawn only directions.size() threads
  num_threads = std::min(num_threads, static_cast<unsigned int>(directions.size()));
  size_t chunk_size = directions.size() / num_threads;
  std::vector<std::thread> threads;

  // Threading to solve for chunks
  auto computeChunk = [&](size_t start, size_t end) {
      // Each thread computes a chunk of directions
      double local_min_altitude_sum = std::numeric_limits<double>::max();
      std::vector<Polygon_2> local_best_bcd;

      for (size_t i = start; i < end; ++i) {
          const Direction_2& dir = directions[i];
          std::vector<Polygon_2> cells = computeBCD(pwh, dir);

          double min_altitude_sum_tmp = 0.0;
          for (const auto& cell : cells) {
              min_altitude_sum_tmp += findBestSweepDir(cell);
          }

          // Update the local best result for this thread
          if (min_altitude_sum_tmp < local_min_altitude_sum) {
              local_min_altitude_sum = min_altitude_sum_tmp;
              local_best_bcd = std::move(cells);
          }
      }

      // Atomic update global result using shared variable 
      double expected = min_altitude_sum.load(std::memory_order_relaxed);
      while (local_min_altitude_sum < expected &&
              !min_altitude_sum.compare_exchange_weak(expected, local_min_altitude_sum,
                                                      std::memory_order_acq_rel)) {
          expected = min_altitude_sum.load(std::memory_order_relaxed);
      }

      if (local_min_altitude_sum == min_altitude_sum.load(std::memory_order_relaxed)) {
          while (update_flag.test_and_set(std::memory_order_acquire));
          *bcd_polygons = std::move(local_best_bcd);
          update_flag.clear(std::memory_order_release);
      }
  };

  //Create threads to compute chunks of directions
  for (size_t i = 0; i < num_threads; ++i) {
      size_t start = i * chunk_size;
      // To handle case where directions.size() not divisible by number of threads
      // Last thread would handle to direction.size()
      size_t end = (i == num_threads - 1) ? directions.size() : start + chunk_size;

      threads.emplace_back(computeChunk, start, end);
  }

  // Join all the threads
  for (auto& thread : threads) {
      thread.join();
  }
  
  if (bcd_polygons -> empty())
    return false;
  else
    return true;

  // // For all possible rotations:
  // for (const auto &dir : directions) {
  //   // std::cout << "Performing calculation for " << dir << std::endl;

  //   // Calculate decomposition.
  //   std::vector<Polygon_2> cells = computeBCD(pwh, dir);

  //   // Calculate minimum altitude sum for each cell.
  //   double min_altitude_sum_tmp = 0.0;
  //   for (const auto &cell : cells) {
  //     min_altitude_sum_tmp += findBestSweepDir(cell);
  //   }

  //   // Update best decomposition.
  //   if (min_altitude_sum_tmp < min_altitude_sum) {
  //     min_altitude_sum = min_altitude_sum_tmp;
  //     *bcd_polygons = cells;
  //   }
  // }

  // if (bcd_polygons->empty())
  //   return false;
  // else
  //   return true;
}

bool computeBestTCDFromPolygonWithHoles(const PolygonWithHoles &pwh,
                                        std::vector<Polygon_2> *tcd_polygons) {
  tcd_polygons->clear();
  double min_altitude_sum = std::numeric_limits<double>::max();

  // Get all possible decomposition directions.
  std::vector<Direction_2> directions = findPerpEdgeDirections(pwh);

  // For all possible rotations:
  for (const auto &dir : directions) {
    // Calculate decomposition.
    std::vector<Polygon_2> cells = computeTCD(pwh, dir);

    // Calculate minimum altitude sum for each cell.
    double min_altitude_sum_tmp = 0.0;
    for (const auto &cell : cells) {
      min_altitude_sum_tmp += findBestSweepDir(cell);
    }

    // Update best decomposition.
    if (min_altitude_sum_tmp < min_altitude_sum) {
      min_altitude_sum = min_altitude_sum_tmp;
      *tcd_polygons = cells;
    }
  }

  if (tcd_polygons->empty())
    return false;
  else
    return true;
}

} // namespace polygon_coverage_planning