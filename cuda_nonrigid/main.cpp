#include "cloud.h"
#include "kernel.cuh"
#include <vector> 

using namespace std;

int main()
{
	myMatrix* source;
	myMatrix* target;
	source = (myMatrix*)malloc(sizeof(myMatrix));
	target = (myMatrix*)malloc(sizeof(myMatrix));

	PointCloudT::Ptr cloud_ori(new PointCloudT);
	PointCloudT::Ptr cloud_trans(new PointCloudT);

	pcl::io::loadPCDFile<PointT>("test_pcd_EX_ds_deform.pcd", *cloud_ori);
	pcl::io::loadPCDFile<PointT>("test_pcd_EX_ds_deform_temp.pcd", *cloud_trans);
	VoxelGridFilter(cloud_ori, cloud_ori, 0.001);
	VoxelGridFilter(cloud_trans, cloud_trans, 0.001);
	cout << cloud_ori->size() << endl;
	cout << cloud_trans->size() << endl;

	source->cols = 3;
	source->rows = cloud_ori->size();
	target->cols = 3;
	target->rows = cloud_trans->size();

	source->Mat = (double*)malloc(sizeof(double)* 3 * source->rows);
	target->Mat = (double*)malloc(sizeof(double)* 3 * target->rows);

	for (int i = 0; i < cloud_ori->size(); ++i)
	{
		source->Mat[3 * i] = cloud_ori->points[i].x;
		source->Mat[3 * i + 1] = cloud_ori->points[i].y;
		source->Mat[3 * i + 2] = cloud_ori->points[i].z;
	}
	for (int i = 0; i < cloud_trans->size(); ++i)
	{
		target->Mat[3 * i] = cloud_trans->points[i].x;
		target->Mat[3 * i + 1] = cloud_trans->points[i].y;
		target->Mat[3 * i + 2] = cloud_trans->points[i].z;
	}

	double m_beta = 3.0;
	double m_lambda = 3.0;
	double sigma2 = 0.278141;
	double outliers = 0.1;
	double tolerance = 1e-5;
	int max_iter = 150;

	double* result = (double*)malloc(source->rows * 3 * sizeof(double));
	double* transform = (double*)malloc(3 * 3 * sizeof(double));
	double* translation = (double*)malloc(3 * sizeof(double));

	ResultCompute(target, source, result, sigma2, outliers, tolerance, max_iter, m_beta, m_lambda);

	PointCloudT::Ptr cloud_result(new PointCloudT);
	PointCloudT &cloud = *cloud_result;
	cloud.height = 1;
	cloud.width = source->rows;
	cloud.is_dense = false;
	cloud.points.resize(cloud.width * cloud.height);

	for (int i = 0; i < cloud.points.size(); ++i)
	{
		cloud.points[i].x = result[i * 3];
		cloud.points[i].y = result[i * 3 + 1];
		cloud.points[i].z = result[i * 3 + 2];
		cloud.points[i].r = 255.0f;
		cloud.points[i].g = 0.0f;
		cloud.points[i].b = 0.0f;
	}

	pcl::io::savePCDFileBinary("result.pcd", *cloud_result);
	pcl::io::savePCDFileBinary("target_filtered.pcd", *cloud_trans);

	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
	pcl::visualization::PointCloudColorHandlerCustom<PointT> cloud_result_color_h(cloud_result, 255, 0, 0);
	viewer->addPointCloud<PointT>(cloud_result, cloud_result_color_h, "cloud result");
	//pcl::visualization::PointCloudColorHandlerCustom<PointT> cloud1_color_h(cloud1_filtered, 20, 180, 20);
	//viewer->addPointCloud<PointT>(cloud1_filtered, cloud1_color_h, "cloud ori");
	pcl::visualization::PointCloudColorHandlerCustom<PointT> cloud_trans_color_h(cloud_trans, 0, 0, 0);
	viewer->addPointCloud<PointT>(cloud_trans, cloud_trans_color_h, "cloud whole");
	viewer->setBackgroundColor(255, 255, 255);
	//viewer->addCoordinateSystem(1.0);
	viewer->initCameraParameters();

	while (!viewer->wasStopped())
	{
		viewer->spinOnce(100);
	}

	free(source);
	free(target);
	system("pause");
	return 0;
}