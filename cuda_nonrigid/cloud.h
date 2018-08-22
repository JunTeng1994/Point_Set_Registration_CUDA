#pragma once

#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/PolygonMesh.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
//#include <opencv2/core/core.hpp>
//#include <opencv2/core/eigen.hpp>
#include "kernel.cuh"

using namespace std;
using namespace Eigen;

typedef pcl::PointXYZRGB PointT;
typedef pcl::PointCloud<PointT> PointCloudT;

void VoxelGridFilter(PointCloudT::Ptr& cloudin, PointCloudT::Ptr& cloudout, float param);
/*void WriteMatrixToCV(Eigen::Isometry3d Matrix, string filename);
vector<vector<double>> ReadOriginalPointsFromfile(string filename);
Eigen::Matrix4d ReadTransMatrix(string filename);
std::vector<double> SolvePTR(Eigen::Isometry3d TrailMatrix, std::vector<double> original_points);
void GenerateTrail(string surfilename, string transfilename, const char* savefilename);*/