#include "cloud.h"

void VoxelGridFilter(PointCloudT::Ptr& cloudin, PointCloudT::Ptr& cloudout, float param)
{
	pcl::VoxelGrid<PointT> voxel;
	voxel.setLeafSize(param, param, param);
	voxel.setInputCloud(cloudin);
	voxel.filter(*cloudout);
}

/*void WriteMatrixToCV(Eigen::Isometry3d Matrix, string filename)
{
cv::Mat mat;
cv::eigen2cv(Matrix.matrix(), mat);
cv::FileStorage fs(filename, cv::FileStorage::WRITE);
fs << "TransformMatrix" << mat;
fs.release();
}
vector<vector<double>> ReadOriginalPointsFromfile(string filename)
{
vector<std::vector<double>> trailpoints;
ifstream fin(filename, std::ios::in);
istringstream istr;
string str;
vector<double> tmpvec;
double tmp;
int i = 0;
while (getline(fin, str))
{
istr.str(str);
while (istr >> tmp)
{
if (i < 6)
{
if (tmp > 3.1415926)
{
tmp = tmp - 2 * 3.1415926;
}
if (tmp < -3.1415926)
{
tmp = tmp + 2 * 3.1415926;
}
}
i++;
if (i == 8)
{
i = 0;
}
tmpvec.push_back(tmp);
}
trailpoints.push_back(tmpvec);
tmpvec.clear();
istr.clear();
str.clear();
}
fin.close();
return trailpoints;
}

Eigen::Matrix4d ReadTransMatrix(string filename)
{
cv::FileStorage Trans(filename, cv::FileStorage::READ);
cv::Mat Mat;
Trans["TransformMatrix"] >> Mat;
Eigen::Matrix4d TransMatrix;
cv::cv2eigen(Mat, TransMatrix);
return TransMatrix;
}

std::vector<double> SolvePTR(Eigen::Isometry3d TrailMatrix, std::vector<double> original_points)
{
std::vector<double> result_points;
double p1, p2, p, t, r;
double s1, s2;

result_points.push_back(TrailMatrix(0, 3));
result_points.push_back(TrailMatrix(1, 3));
result_points.push_back(TrailMatrix(2, 3));

p1 = atan2(TrailMatrix(1, 0), TrailMatrix(0, 0));
p2 = atan2(-TrailMatrix(1, 0), -TrailMatrix(0, 0));
s1 = abs(p1 - original_points[3]);
s2 = abs(p2 - original_points[3]);
if (s1 < s2)
{
p = p1;
}
else
{
p = p2;
}
t = atan2(-TrailMatrix(2, 0), TrailMatrix(0, 0)*cos(p) + TrailMatrix(1, 0)*sin(p));
r = atan2(TrailMatrix(0, 2)*sin(p) - TrailMatrix(1, 2)*cos(p), TrailMatrix(1, 1)*cos(p) - TrailMatrix(0, 1)*sin(p));
result_points.push_back(p);
result_points.push_back(t);
result_points.push_back(r);

return result_points;
}

void GenerateTrail(string surfilename, string transfilename, const char* savefilename)
{
vector<vector<double>> trailpoints;
trailpoints = ReadOriginalPointsFromfile(surfilename);

vector<Eigen::Isometry3d> TrailMatrix;
Eigen::Isometry3d transtemp;
Eigen::Vector3d  translation;
for (int i = 1; i < trailpoints.size(); i++)
{
transtemp = Eigen::Isometry3d::Identity();
translation[0] = trailpoints[i][0];
translation[1] = trailpoints[i][1];
translation[2] = trailpoints[i][2];
transtemp.translate(translation);
transtemp.rotate(Eigen::AngleAxisd(trailpoints[i][3], Eigen::Vector3d::UnitZ()));
transtemp.rotate(Eigen::AngleAxisd(trailpoints[i][4], Eigen::Vector3d::UnitY()));
transtemp.rotate(Eigen::AngleAxisd(trailpoints[i][5], Eigen::Vector3d::UnitX()));
TrailMatrix.push_back(transtemp);
}

Eigen::Matrix4d IcpTrans;
IcpTrans = ReadTransMatrix(transfilename);
cout << IcpTrans << endl;

for (int i = 0; i < TrailMatrix.size(); i++)
{
TrailMatrix[i].matrix() = IcpTrans*TrailMatrix[i].matrix();
}

vector<vector<double>> result_points;
vector<double> tmppoints;
for (int i = 0; i < TrailMatrix.size(); i++)
{
tmppoints = SolvePTR(TrailMatrix[i], trailpoints[i + 1]);
result_points.push_back(tmppoints);
tmppoints.clear();
}

FILE *wf = fopen(savefilename, "wt");
for (int i = 0; i < trailpoints.size(); i++)
{
if (i == 0 || i == 1 || i == trailpoints.size() - 1)
{
for (int j = 0; j < 8; j++)
{
fprintf(wf, "%.6f ", (trailpoints[i])[j]);
}
fprintf(wf, " \n");
}
else
{
for (int j = 0; j < 6; j++)
fprintf(wf, "%.6f ", (result_points[i - 2])[j]);
fprintf(wf, "%f %f \n", trailpoints[i][6], trailpoints[i][7]);
}
}
fclose(wf);
}*/