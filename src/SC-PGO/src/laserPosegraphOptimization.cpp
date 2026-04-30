#include <fstream>
#include <math.h>
#include <vector>
#include <mutex>
#include <queue>
#include <thread>
#include <iostream>
#include <string>
#include <optional>
#include <unordered_map>
#include <cstdint>
#include <iomanip>
#include <limits>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/impl/search.hpp>
#include <pcl/range_image/range_image.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/correspondence_rejection_trimmed.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/octree/octree_pointcloud_voxelcentroid.h>
#include <pcl/filters/crop_box.h> 
#include <pcl_conversions/pcl_conversions.h>

#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/NavSatFix.h>
#include <geometry_msgs/Vector3Stamped.h>
#include <cv_bridge/cv_bridge.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <std_msgs/Bool.h>

#include <eigen3/Eigen/Dense>

#include <ceres/ceres.h>

#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot2.h>
#include <gtsam/geometry/Pose2.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/ISAM2.h>

#include "aloam_velodyne/common.h"
#include "aloam_velodyne/tic_toc.h"

#include "scancontext/Scancontext.h"

using namespace gtsam;

using std::cout;
using std::endl;

double keyframeMeterGap;
double keyframeDegGap, keyframeRadGap;
double translationAccumulated = 1000000.0; // large value means must add the first given frame.
double rotaionAccumulated = 1000000.0; // large value means must add the first given frame.

bool isNowKeyFrame = false; 

Pose6D odom_pose_prev {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}; // init 
Pose6D odom_pose_curr {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}; // init pose is zero 

std::queue<nav_msgs::Odometry::ConstPtr> odometryBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> fullResBuf;
std::queue<sensor_msgs::ImageConstPtr> imageBuf;
std::queue<sensor_msgs::NavSatFix::ConstPtr> gpsBuf;
std::queue<geometry_msgs::Vector3Stamped::ConstPtr> kfGravityBuf;
std::queue<std_msgs::Bool::ConstPtr> elevatorSceneBuf;
std::queue<std::pair<int, int> > scLoopICPBuf;

std::mutex mBuf;
std::mutex mKF;
std::mutex mGravity;

double timeLaserOdometry = 0.0;
double timeLaser = 0.0;

pcl::PointCloud<PointType>::Ptr laserCloudFullRes(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr laserCloudMapAfterPGO(new pcl::PointCloud<PointType>());

std::vector<pcl::PointCloud<PointType>::Ptr> keyframeLaserClouds; 
std::vector<Pose6D> keyframePoses;
std::vector<Pose6D> keyframePosesUpdated;
std::vector<double> keyframeTimes;
std::vector<uint64_t> keyframeSourceFrameIds;
std::vector<int> keyframeSessions;
int currentSession = 1;
uint64_t allFrameCounter = 0;
int recentIdxUpdated = 0;
int lastLoopDetectCurrIdx = -1;

gtsam::NonlinearFactorGraph gtSAMgraph;
bool gtSAMgraphMade = false;
gtsam::Values initialEstimate;
gtsam::ISAM2 *isam;
gtsam::Values isamCurrentEstimate;

noiseModel::Diagonal::shared_ptr priorNoise;
noiseModel::Diagonal::shared_ptr odomNoise;
noiseModel::Diagonal::shared_ptr degenerateNoise;
noiseModel::Base::shared_ptr floorHeightNoise;
noiseModel::Base::shared_ptr robustLoopNoise;
noiseModel::Base::shared_ptr robustGPSNoise;

bool latestElevatorScene = false;
bool hasElevatorSceneMsg = false;
std::string elevatorSceneTopic = "/lio/elevator_scene";
std::string imageTopic = "/rgb_img";

bool enableNonElevatorFloorHeightTracking = true;
double nonElevatorFloorHeightAlpha = 0.1;
bool hasTrackedFloorHeight = false;
double trackedFloorHeight = 0.0;
std::mutex mFloorHeight;
bool useTrackedFloorHeightFactorInElevator = true;
double floorHeightFactorSigma = 0.2;
double elevatorFloorHeight = 3.0;
int elevatorFloorCount = 1;
int elevatorCurrentFloor = 0;
int elevatorTargetFloor = 1;
bool hasPrevKeyframeElevatorScene = false;
bool prevKeyframeElevatorScene = false;
int elevatorEnterNodeIdx = -1;
Eigen::Vector3d elevatorEnterUpAxis(0.0, 0.0, 1.0);
bool hasElevatorEnterUpAxis = false;

pcl::VoxelGrid<PointType> downSizeFilterScancontext;
SCManager scManager;
double scDistThres, scMaximumRadius;
double loopFitnessScoreThreshold = 0.3;
double icpTrimmedOverlapRatio = 0.4;

pcl::VoxelGrid<PointType> downSizeFilterICP;
double icpSubmapLeafSize = 0.4;
std::mutex mtxICP;
std::mutex mtxPosegraph;
std::mutex mtxRecentPose;

pcl::PointCloud<PointType>::Ptr laserCloudMapPGO(new pcl::PointCloud<PointType>());
pcl::VoxelGrid<PointType> downSizeFilterMapPGO;
double mapVizLeafSize = 0.4;
bool laserCloudMapPGORedraw = true;

bool useGPS = false;
// bool useGPS = false;
sensor_msgs::NavSatFix::ConstPtr currGPS;
bool hasGPSforThisKF = false;
bool gpsOffsetInitialized = false; 
double gpsAltitudeInitOffset = 0.0;
double recentOptimizedX = 0.0;
double recentOptimizedY = 0.0;

ros::Publisher pubMapAftPGO, pubOdomAftPGO, pubPathAftPGO;
ros::Publisher pubLoopScanLocal, pubLoopSubmapLocal;
ros::Publisher pubOdomRepubVerifier;

Eigen::Vector3d latestKfGravity(0.0, 0.0, -9.81);
bool hasKfGravity = false;
bool useKfGravityAlignForSC = false;
std::string kfGravityTopic = "/kf_gravity";

std::string save_directory;
std::string pgKITTIformat, pgScansDirectory;
std::string pgLoopCloudDirectory;
std::string odomKITTIformat;
std::fstream pgTimeSaveStream;
std::fstream loopTimePairSaveStream;
std::fstream scDetectAllPairsSaveStream;
std::fstream keyframeFrameIdSaveStream;
std::fstream subscribedOdomSaveStream;
std::fstream subscribedOdomMetaSaveStream;
std::fstream subscribedKfGravitySaveStream;
bool saveMapToDisk = true;
bool saveSCKeyframeToDisk = false;
bool saveKeyframeImageToDisk = true;
double mapSaveIntervalSec = 30.0;
std::string keyframeImageDirectory;

sensor_msgs::ImageConstPtr latestImageMsg;

std::string padZeros(int val, int num_digits = 6) {
  std::ostringstream out;
  out << std::internal << std::setfill('0') << std::setw(num_digits) << val;
  return out.str();
}

gtsam::Pose3 Pose6DtoGTSAMPose3(const Pose6D& p)
{
    return gtsam::Pose3( gtsam::Rot3::RzRyRx(p.roll, p.pitch, p.yaw), gtsam::Point3(p.x, p.y, p.z) );
} // Pose6DtoGTSAMPose3

void saveOdometryVerticesKITTIformat(std::string _filename)
{
    // ref from gtsam's original code "dataset.cpp"
    std::fstream stream(_filename.c_str(), std::fstream::out);

    // Snapshot under lock to avoid concurrent vector reallocation while iterating.
    std::vector<Pose6D> keyframePosesSnapshot;
    mKF.lock();
    keyframePosesSnapshot = keyframePoses;
    mKF.unlock();

    for(const auto& _pose6d: keyframePosesSnapshot) {
        gtsam::Pose3 pose = Pose6DtoGTSAMPose3(_pose6d);
        Point3 t = pose.translation();
        Rot3 R = pose.rotation();
        auto col1 = R.column(1); // Point3
        auto col2 = R.column(2); // Point3
        auto col3 = R.column(3); // Point3

        stream << col1.x() << " " << col2.x() << " " << col3.x() << " " << t.x() << " "
               << col1.y() << " " << col2.y() << " " << col3.y() << " " << t.y() << " "
               << col1.z() << " " << col2.z() << " " << col3.z() << " " << t.z() << std::endl;
    }
}

void saveOptimizedVerticesKITTIformat(gtsam::Values _estimates, std::string _filename)
{
    using namespace gtsam;

    // ref from gtsam's original code "dataset.cpp"
    std::fstream stream(_filename.c_str(), std::fstream::out);

    for(const auto& key_value: _estimates) {
        auto p = dynamic_cast<const GenericValue<Pose3>*>(&key_value.value);
        if (!p) continue;

        const Pose3& pose = p->value();

        Point3 t = pose.translation();
        Rot3 R = pose.rotation();
        auto col1 = R.column(1); // Point3
        auto col2 = R.column(2); // Point3
        auto col3 = R.column(3); // Point3

        stream << col1.x() << " " << col2.x() << " " << col3.x() << " " << t.x() << " "
               << col1.y() << " " << col2.y() << " " << col3.y() << " " << t.y() << " "
               << col1.z() << " " << col2.z() << " " << col3.z() << " " << t.z() << std::endl;
    }
}

void laserOdometryHandler(const nav_msgs::Odometry::ConstPtr &_laserOdometry)
{
	mBuf.lock();
	odometryBuf.push(_laserOdometry);
	mBuf.unlock();
} // laserOdometryHandler

void laserCloudFullResHandler(const sensor_msgs::PointCloud2ConstPtr &_laserCloudFullRes)
{
	mBuf.lock();
	fullResBuf.push(_laserCloudFullRes);
	mBuf.unlock();
} // laserCloudFullResHandler

void imageHandler(const sensor_msgs::ImageConstPtr &msg)
{
    mBuf.lock();
    imageBuf.push(msg);
    mBuf.unlock();
}

void gpsHandler(const sensor_msgs::NavSatFix::ConstPtr &_gps)
{
    if(useGPS) {
        mBuf.lock();
        gpsBuf.push(_gps);
        mBuf.unlock();
    }
} // gpsHandler

void elevatorSceneHandler(const std_msgs::Bool::ConstPtr& msg)
{
    mBuf.lock();
    elevatorSceneBuf.push(msg);
    latestElevatorScene = msg->data;
    hasElevatorSceneMsg = true;
    mBuf.unlock();
}

void kfGravityHandler(const geometry_msgs::Vector3Stamped::ConstPtr& msg)
{
    std::lock_guard<std::mutex> lock(mGravity);
    latestKfGravity = Eigen::Vector3d(msg->vector.x, msg->vector.y, msg->vector.z);
    hasKfGravity = true;

    mBuf.lock();
    kfGravityBuf.push(msg);
    mBuf.unlock();
}

bool getLatestKfGravity(Eigen::Vector3d& gravityOut)
{
    std::lock_guard<std::mutex> lock(mGravity);
    if (!hasKfGravity) {
        return false;
    }

    gravityOut = latestKfGravity;
    return true;
}

bool getHeightAlongGravityAxis(const Pose6D& pose, double& heightOut)
{
    Eigen::Vector3d gravityVec;
    if (!getLatestKfGravity(gravityVec)) {
        return false;
    }

    const double gNorm = gravityVec.norm();
    if (gNorm < 1e-6) {
        return false;
    }

    const Eigen::Vector3d upAxis = -gravityVec / gNorm;
    const Eigen::Vector3d position(pose.x, pose.y, pose.z);
    heightOut = upAxis.dot(position);
    return true;
}

bool getCloudHeightSpanAlongGravityAxis(const pcl::PointCloud<PointType>::Ptr& cloud,
                                        double& heightSpanOut)
{
    if (!cloud || cloud->empty()) {
        return false;
    }

    Eigen::Vector3d gravityVec;
    if (!getLatestKfGravity(gravityVec)) {
        return false;
    }

    const double gNorm = gravityVec.norm();
    if (gNorm < 1e-6) {
        return false;
    }

    const Eigen::Vector3d upAxis = -gravityVec / gNorm;
    double minProj = std::numeric_limits<double>::infinity();
    double maxProj = -std::numeric_limits<double>::infinity();

    for (const auto& point : cloud->points) {
        const Eigen::Vector3d p(point.x, point.y, point.z);
        const double projection = upAxis.dot(p);
        minProj = std::min(minProj, projection);
        maxProj = std::max(maxProj, projection);
    }

    if (!std::isfinite(minProj) || !std::isfinite(maxProj)) {
        return false;
    }

    heightSpanOut = maxProj - minProj;
    return true;
}

bool getCloudHeightSpanAlongZ(const pcl::PointCloud<PointType>::Ptr& cloud,
                              double& heightSpanOut)
{
    if (!cloud || cloud->empty()) {
        return false;
    }

    double minZ = std::numeric_limits<double>::infinity();
    double maxZ = -std::numeric_limits<double>::infinity();
    for (const auto& point : cloud->points) {
        minZ = std::min(minZ, static_cast<double>(point.z));
        maxZ = std::max(maxZ, static_cast<double>(point.z));
    }

    if (!std::isfinite(minZ) || !std::isfinite(maxZ)) {
        return false;
    }

    heightSpanOut = maxZ - minZ;
    return true;
}

void updateTrackedFloorHeight(const double zMeasurement)
{
    std::lock_guard<std::mutex> lock(mFloorHeight);
    if (!hasTrackedFloorHeight) {
        trackedFloorHeight = zMeasurement;
        hasTrackedFloorHeight = true;
        return;
    }

    trackedFloorHeight = (1.0 - nonElevatorFloorHeightAlpha) * trackedFloorHeight
                      + nonElevatorFloorHeightAlpha * zMeasurement;
}

bool getTrackedFloorHeight(double& zOut)
{
    std::lock_guard<std::mutex> lock(mFloorHeight);
    if (!hasTrackedFloorHeight) {
        return false;
    }

    zOut = trackedFloorHeight;
    return true;
}

class GravityAxisHeightFactor : public gtsam::NoiseModelFactor1<gtsam::Pose3>
{
public:
    GravityAxisHeightFactor(gtsam::Key key,
                            const Eigen::Vector3d& upAxis,
                            const double targetHeight,
                            const gtsam::noiseModel::Base::shared_ptr& model)
        : gtsam::NoiseModelFactor1<gtsam::Pose3>(model, key), upAxis_(upAxis), targetHeight_(targetHeight) {}

    gtsam::Vector evaluateError(const gtsam::Pose3& pose,
                                gtsam::OptionalMatrixType H = nullptr) const override
    {
        const Eigen::Vector3d position = pose.translation();
        gtsam::Vector1 error;
        error << upAxis_.dot(position) - targetHeight_;

        if (H) {
            H->setZero(1, 6);
            H->block<1, 3>(0, 3) = upAxis_.transpose();
        }

        return error;
    }

private:
    Eigen::Vector3d upAxis_;
    double targetHeight_;
};

class GravityAxisHeightDifferenceFactor : public gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Pose3>
{
public:
    GravityAxisHeightDifferenceFactor(gtsam::Key keyFrom,
                                      gtsam::Key keyTo,
                                      const Eigen::Vector3d& upAxis,
                                      const double targetHeightDelta,
                                      const gtsam::noiseModel::Base::shared_ptr& model)
        : gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Pose3>(model, keyFrom, keyTo),
          upAxis_(upAxis),
          targetHeightDelta_(targetHeightDelta) {}

    gtsam::Vector evaluateError(const gtsam::Pose3& poseFrom,
                                const gtsam::Pose3& poseTo,
                                gtsam::OptionalMatrixType H1 = nullptr,
                                gtsam::OptionalMatrixType H2 = nullptr) const override
    {
        const Eigen::Vector3d positionFrom = poseFrom.translation();
        const Eigen::Vector3d positionTo = poseTo.translation();
        gtsam::Vector1 error;
        error << upAxis_.dot(positionTo - positionFrom) - targetHeightDelta_;

        if (H1) {
            H1->setZero(1, 6);
            H1->block<1, 3>(0, 3) = -upAxis_.transpose();
        }

        if (H2) {
            H2->setZero(1, 6);
            H2->block<1, 3>(0, 3) = upAxis_.transpose();
        }

        return error;
    }

private:
    Eigen::Vector3d upAxis_;
    double targetHeightDelta_;
};

void alignPointCloudToGravityZ(const Eigen::Vector3d& gravityVec, pcl::PointCloud<PointType>::Ptr& cloud)
{
    if (!cloud || cloud->empty()) {
        return;
    }

    const double gNorm = gravityVec.norm();
    if (gNorm < 1e-6) {
        return;
    }

    const Eigen::Vector3d from = gravityVec / gNorm;
    const Eigen::Vector3d to(0.0, 0.0, -1.0);
    Eigen::Quaterniond q = Eigen::Quaterniond::FromTwoVectors(from, to);
    Eigen::Matrix3d R = q.toRotationMatrix();

    for (auto& p : cloud->points) {
        Eigen::Vector3d v(p.x, p.y, p.z);
        const Eigen::Vector3d vr = R * v;
        p.x = static_cast<float>(vr.x());
        p.y = static_cast<float>(vr.y());
        p.z = static_cast<float>(vr.z());
    }
}

void initNoises( void )
{
    gtsam::Vector priorNoiseVector6(6);
    priorNoiseVector6 << 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12;
    priorNoise = noiseModel::Diagonal::Variances(priorNoiseVector6);

    gtsam::Vector odomNoiseVector6(6);
    // odomNoiseVector6 << 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4;
    odomNoiseVector6 << 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6;
    odomNoise = noiseModel::Diagonal::Variances(odomNoiseVector6);

    gtsam::Vector degenerateNoiseVector6(6);
    // Degraded-elevator scene: use larger variances to distrust front-end odometry factors.
    // Translation is unreliable in degenerate scenes, but rotation is relatively reliable.
    // Order: [x, y, z, roll, pitch, yaw]
    degenerateNoiseVector6 << 1e-4, 1e-4, 1e-4, 1e-2, 1e-2, 1e-2;
    degenerateNoise = noiseModel::Diagonal::Variances(degenerateNoiseVector6);

    floorHeightNoise = gtsam::noiseModel::Isotropic::Sigma(1, floorHeightFactorSigma);

    /* double loopNoiseScore = 0.5; // constant is ok...
    gtsam::Vector robustNoiseVector6(6); // gtsam::Pose3 factor has 6 elements (6D)
    robustNoiseVector6 << loopNoiseScore, loopNoiseScore, loopNoiseScore, loopNoiseScore, loopNoiseScore, loopNoiseScore;
    robustLoopNoise = gtsam::noiseModel::Robust::Create(
                    gtsam::noiseModel::mEstimator::Cauchy::Create(1), // optional: replacing Cauchy by DCS or GemanMcClure is okay but Cauchy is empirically good.
                    gtsam::noiseModel::Diagonal::Variances(robustNoiseVector6) ); */
    gtsam::Vector loopNoiseVector6(6);
    loopNoiseVector6 << 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6;
    robustLoopNoise = noiseModel::Diagonal::Variances(loopNoiseVector6);

    double bigNoiseTolerentToXY = 1000000000.0; // 1e9
    double gpsAltitudeNoiseScore = 250.0; // if height is misaligned after loop clsosing, use this value bigger
    gtsam::Vector robustNoiseVector3(3); // gps factor has 3 elements (xyz)
    robustNoiseVector3 << bigNoiseTolerentToXY, bigNoiseTolerentToXY, gpsAltitudeNoiseScore; // means only caring altitude here. (because LOAM-like-methods tends to be asymptotically flyging)
    robustGPSNoise = gtsam::noiseModel::Robust::Create(
                    gtsam::noiseModel::mEstimator::Cauchy::Create(1), // optional: replacing Cauchy by DCS or GemanMcClure is okay but Cauchy is empirically good.
                    gtsam::noiseModel::Diagonal::Variances(robustNoiseVector3) );

} // initNoises

Pose6D getOdom(nav_msgs::Odometry::ConstPtr _odom)
{
    auto tx = _odom->pose.pose.position.x;
    auto ty = _odom->pose.pose.position.y;
    auto tz = _odom->pose.pose.position.z;

    double roll, pitch, yaw;
    geometry_msgs::Quaternion quat = _odom->pose.pose.orientation;
    tf::Matrix3x3(tf::Quaternion(quat.x, quat.y, quat.z, quat.w)).getRPY(roll, pitch, yaw);

    return Pose6D{tx, ty, tz, roll, pitch, yaw}; 
} // getOdom

Pose6D diffTransformation(const Pose6D& _p1, const Pose6D& _p2)
{
    Eigen::Affine3f SE3_p1 = pcl::getTransformation(_p1.x, _p1.y, _p1.z, _p1.roll, _p1.pitch, _p1.yaw);
    Eigen::Affine3f SE3_p2 = pcl::getTransformation(_p2.x, _p2.y, _p2.z, _p2.roll, _p2.pitch, _p2.yaw);
    Eigen::Matrix4f SE3_delta0 = SE3_p1.matrix().inverse() * SE3_p2.matrix();
    Eigen::Affine3f SE3_delta; SE3_delta.matrix() = SE3_delta0;
    float dx, dy, dz, droll, dpitch, dyaw;
    pcl::getTranslationAndEulerAngles (SE3_delta, dx, dy, dz, droll, dpitch, dyaw);
    // std::cout << "delta : " << dx << ", " << dy << ", " << dz << ", " << droll << ", " << dpitch << ", " << dyaw << std::endl;

    return Pose6D{double(abs(dx)), double(abs(dy)), double(abs(dz)), double(abs(droll)), double(abs(dpitch)), double(abs(dyaw))};
} // SE3Diff

pcl::PointCloud<PointType>::Ptr local2global(const pcl::PointCloud<PointType>::Ptr &cloudIn, const Pose6D& tf)
{
    pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

    const int cloudSize = cloudIn->size();
    cloudOut->resize(cloudSize);

    Eigen::Affine3f transCur = pcl::getTransformation(tf.x, tf.y, tf.z, tf.roll, tf.pitch, tf.yaw);
    
    int numberOfCores = 16;
    #pragma omp parallel for num_threads(numberOfCores)
    for (int i = 0; i < cloudSize; ++i)
    {
        const auto& pointFrom = cloudIn->points[i];

        cloudOut->points[i].x = transCur(0,0) * pointFrom.x + transCur(0,1) * pointFrom.y + transCur(0,2) * pointFrom.z + transCur(0,3);
        cloudOut->points[i].y = transCur(1,0) * pointFrom.x + transCur(1,1) * pointFrom.y + transCur(1,2) * pointFrom.z + transCur(1,3);
        cloudOut->points[i].z = transCur(2,0) * pointFrom.x + transCur(2,1) * pointFrom.y + transCur(2,2) * pointFrom.z + transCur(2,3);
        cloudOut->points[i].rgb = pointFrom.rgb;
    }

    return cloudOut;
}

pcl::PointCloud<PointType>::Ptr local2reference(const pcl::PointCloud<PointType>::Ptr &cloudIn,
                                                const Pose6D& pose_k,
                                                const Pose6D& pose_ref)
{
    pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

    const int cloudSize = cloudIn->size();
    cloudOut->resize(cloudSize);

    Eigen::Affine3f transK = pcl::getTransformation(pose_k.x, pose_k.y, pose_k.z, pose_k.roll, pose_k.pitch, pose_k.yaw);
    Eigen::Affine3f transRef = pcl::getTransformation(pose_ref.x, pose_ref.y, pose_ref.z, pose_ref.roll, pose_ref.pitch, pose_ref.yaw);
    const Eigen::Matrix4f transKToRef = transRef.matrix().inverse() * transK.matrix();

    int numberOfCores = 16;
    #pragma omp parallel for num_threads(numberOfCores)
    for (int i = 0; i < cloudSize; ++i)
    {
        const auto& pointFrom = cloudIn->points[i];

        cloudOut->points[i].x = transKToRef(0,0) * pointFrom.x + transKToRef(0,1) * pointFrom.y + transKToRef(0,2) * pointFrom.z + transKToRef(0,3);
        cloudOut->points[i].y = transKToRef(1,0) * pointFrom.x + transKToRef(1,1) * pointFrom.y + transKToRef(1,2) * pointFrom.z + transKToRef(1,3);
        cloudOut->points[i].z = transKToRef(2,0) * pointFrom.x + transKToRef(2,1) * pointFrom.y + transKToRef(2,2) * pointFrom.z + transKToRef(2,3);
        cloudOut->points[i].rgb = pointFrom.rgb;
    }

    return cloudOut;
}

void pubPath( void )
{
    // pub odom and path 
    nav_msgs::Odometry odomAftPGO;
    nav_msgs::Path pathAftPGO;
    pathAftPGO.header.frame_id = "camera_init";
    mKF.lock(); 
    // for (int node_idx=0; node_idx < int(keyframePosesUpdated.size()) - 1; node_idx++) // -1 is just delayed visualization (because sometimes mutexed while adding(push_back) a new one)
    for (int node_idx=0; node_idx < recentIdxUpdated; node_idx++) // -1 is just delayed visualization (because sometimes mutexed while adding(push_back) a new one)
    {
        const Pose6D& pose_est = keyframePosesUpdated.at(node_idx); // upodated poses
        // const gtsam::Pose3& pose_est = isamCurrentEstimate.at<gtsam::Pose3>(node_idx);

        nav_msgs::Odometry odomAftPGOthis;
        odomAftPGOthis.header.frame_id = "camera_init";
        odomAftPGOthis.child_frame_id = "/aft_pgo";
        odomAftPGOthis.header.stamp = ros::Time().fromSec(keyframeTimes.at(node_idx));
        odomAftPGOthis.pose.pose.position.x = pose_est.x;
        odomAftPGOthis.pose.pose.position.y = pose_est.y;
        odomAftPGOthis.pose.pose.position.z = pose_est.z;
        odomAftPGOthis.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(pose_est.roll, pose_est.pitch, pose_est.yaw);
        odomAftPGO = odomAftPGOthis;

        geometry_msgs::PoseStamped poseStampAftPGO;
        poseStampAftPGO.header = odomAftPGOthis.header;
        poseStampAftPGO.pose = odomAftPGOthis.pose.pose;

        pathAftPGO.header.stamp = odomAftPGOthis.header.stamp;
        pathAftPGO.header.frame_id = "camera_init";
        pathAftPGO.poses.push_back(poseStampAftPGO);
    }
    mKF.unlock(); 
    pubOdomAftPGO.publish(odomAftPGO); // last pose 
    pubPathAftPGO.publish(pathAftPGO); // poses 

    static tf::TransformBroadcaster br;
    static ros::Time lastTfStamp(0);
    tf::Transform transform;
    tf::Quaternion q;
    transform.setOrigin(tf::Vector3(odomAftPGO.pose.pose.position.x, odomAftPGO.pose.pose.position.y, odomAftPGO.pose.pose.position.z));
    q.setW(odomAftPGO.pose.pose.orientation.w);
    q.setX(odomAftPGO.pose.pose.orientation.x);
    q.setY(odomAftPGO.pose.pose.orientation.y);
    q.setZ(odomAftPGO.pose.pose.orientation.z);
    transform.setRotation(q);
    // Avoid repeatedly broadcasting the same stamp, which causes TF_REPEATED_DATA warnings.
    if (odomAftPGO.header.stamp > lastTfStamp) {
        br.sendTransform(tf::StampedTransform(transform, odomAftPGO.header.stamp, "camera_init", "/aft_pgo"));
        lastTfStamp = odomAftPGO.header.stamp;
    }
} // pubPath

void updatePoses(void)
{
    mKF.lock(); 
    const int poseCount = std::min(int(isamCurrentEstimate.size()), int(keyframePosesUpdated.size()));
    for (int node_idx = 0; node_idx < poseCount; node_idx++)
    {
        Pose6D& p =keyframePosesUpdated[node_idx];
        p.x = isamCurrentEstimate.at<gtsam::Pose3>(node_idx).translation().x();
        p.y = isamCurrentEstimate.at<gtsam::Pose3>(node_idx).translation().y();
        p.z = isamCurrentEstimate.at<gtsam::Pose3>(node_idx).translation().z();
        p.roll = isamCurrentEstimate.at<gtsam::Pose3>(node_idx).rotation().roll();
        p.pitch = isamCurrentEstimate.at<gtsam::Pose3>(node_idx).rotation().pitch();
        p.yaw = isamCurrentEstimate.at<gtsam::Pose3>(node_idx).rotation().yaw();
    }
    const int updatedSize = int(keyframePosesUpdated.size());
    mKF.unlock();

    mtxRecentPose.lock();
    const gtsam::Pose3& lastOptimizedPose = isamCurrentEstimate.at<gtsam::Pose3>(int(isamCurrentEstimate.size())-1);
    recentOptimizedX = lastOptimizedPose.translation().x();
    recentOptimizedY = lastOptimizedPose.translation().y();

    recentIdxUpdated = updatedSize - 1;

    mtxRecentPose.unlock();
} // updatePoses

void runISAM2opt(void)
{
    // called when a variable added 
    isam->update(gtSAMgraph, initialEstimate);
    isam->update();
    
    gtSAMgraph.resize(0);
    initialEstimate.clear();

    isamCurrentEstimate = isam->calculateEstimate();
    updatePoses();
}

pcl::PointCloud<PointType>::Ptr transformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn, gtsam::Pose3 transformIn)
{
    pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

    const int cloudSize = cloudIn->size();
    cloudOut->resize(cloudSize);

    Eigen::Affine3f transCur = pcl::getTransformation(
                                    transformIn.translation().x(), transformIn.translation().y(), transformIn.translation().z(), 
                                    transformIn.rotation().roll(), transformIn.rotation().pitch(), transformIn.rotation().yaw() );
    
    int numberOfCores = 8; // TODO move to yaml 
    #pragma omp parallel for num_threads(numberOfCores)
    for (int i = 0; i < cloudSize; ++i)
    {
        const auto& pointFrom = cloudIn->points[i];
        cloudOut->points[i].x = transCur(0,0) * pointFrom.x + transCur(0,1) * pointFrom.y + transCur(0,2) * pointFrom.z + transCur(0,3);
        cloudOut->points[i].y = transCur(1,0) * pointFrom.x + transCur(1,1) * pointFrom.y + transCur(1,2) * pointFrom.z + transCur(1,3);
        cloudOut->points[i].z = transCur(2,0) * pointFrom.x + transCur(2,1) * pointFrom.y + transCur(2,2) * pointFrom.z + transCur(2,3);
        cloudOut->points[i].rgb = pointFrom.rgb;
    }
    return cloudOut;
} // transformPointCloud

namespace {

struct VoxelKey {
    int x;
    int y;
    int z;

    bool operator==(const VoxelKey& other) const {
        return x == other.x && y == other.y && z == other.z;
    }
};

struct VoxelKeyHasher {
    std::size_t operator()(const VoxelKey& key) const {
        std::size_t h1 = std::hash<int>{}(key.x);
        std::size_t h2 = std::hash<int>{}(key.y);
        std::size_t h3 = std::hash<int>{}(key.z);
        return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
};

// Keep the first point in each voxel. This avoids using PCL filter allocations in the loop-closure thread.
pcl::PointCloud<PointType>::Ptr downsampleByVoxelFirstPoint(
    const pcl::PointCloud<PointType>::Ptr& cloudIn,
    const float leafSize)
{
    pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());
    if (!cloudIn || cloudIn->empty()) {
        return cloudOut;
    }

    if (leafSize <= 0.0f) {
        *cloudOut = *cloudIn;
        return cloudOut;
    }

    std::unordered_map<VoxelKey, std::size_t, VoxelKeyHasher> voxelToIndex;
    voxelToIndex.reserve(cloudIn->size());
    cloudOut->points.reserve(cloudIn->size());

    const float invLeaf = 1.0f / leafSize;
    for (const auto& p : cloudIn->points) {
        VoxelKey key{
            static_cast<int>(std::floor(p.x * invLeaf)),
            static_cast<int>(std::floor(p.y * invLeaf)),
            static_cast<int>(std::floor(p.z * invLeaf))
        };

        if (voxelToIndex.find(key) == voxelToIndex.end()) {
            voxelToIndex.emplace(key, cloudOut->points.size());
            cloudOut->points.push_back(p);
        }
    }

    cloudOut->width = static_cast<std::uint32_t>(cloudOut->points.size());
    cloudOut->height = 1;
    cloudOut->is_dense = cloudIn->is_dense;
    return cloudOut;
}

} // namespace

void loopFindNearKeyframesCloud( pcl::PointCloud<PointType>::Ptr& nearKeyframes, const int& key, const int& submap_size, const int& root_idx)
{
    if (!nearKeyframes) {
        return;
    }

    // submap_size == 0 is valid (single keyframe), only reject negative values.
    if (submap_size < 0) {
        ROS_WARN_STREAM("[SC loop] invalid submap_size: " << submap_size);
        return;
    }

    // extract and stacking near keyframes (in global coord)
    nearKeyframes->clear();

    struct StagedCloud {
        pcl::PointCloud<PointType>::Ptr cloud;
        Pose6D pose;
    };
    std::vector<StagedCloud> localClouds;
    Pose6D rootPose;
    bool hasRootPose = false;

    // Snapshot keyframe clouds under the same mutex used for push_back writes.
    {
        std::lock_guard<std::mutex> lock(mKF);

        if (root_idx >= 0 && root_idx < int(keyframePosesUpdated.size())) {
            rootPose = keyframePosesUpdated[root_idx];
            hasRootPose = true;
        }

        localClouds.reserve(2 * submap_size + 1);
        for (int i = -submap_size; i <= submap_size; ++i) {
            const int keyNear = key + i;
            if (keyNear < 0 || keyNear >= int(keyframeLaserClouds.size())) {
                continue;
            }
            if (keyNear >= int(keyframePosesUpdated.size())) {
                continue;
            }
            const auto& cloudPtr = keyframeLaserClouds[keyNear];
            if (!cloudPtr || cloudPtr->empty()) {
                continue;
            }

            // Deep copy avoids dangling/shared ownership issues when other threads append keyframes.
            localClouds.push_back(StagedCloud{pcl::PointCloud<PointType>::Ptr(new pcl::PointCloud<PointType>(*cloudPtr)),
                                             keyframePosesUpdated[keyNear]});
        }
    }

    if (!hasRootPose) {
        ROS_WARN_STREAM("[SC loop] invalid root_idx for local submap transform: " << root_idx);
        return;
    }

    for (const auto& staged : localClouds) {
        if (!staged.cloud || staged.cloud->empty()) {
            continue;
        }
        *nearKeyframes += *local2reference(staged.cloud, staged.pose, rootPose);
    }

    if (nearKeyframes->empty())
        return;

    pcl::PointCloud<PointType>::Ptr cloudFiltered =
        downsampleByVoxelFirstPoint(nearKeyframes, static_cast<float>(icpSubmapLeafSize));

    if (!cloudFiltered->empty()) {
        nearKeyframes.swap(cloudFiltered);
    }
}


struct LoopIcpResult {
    gtsam::Pose3 relative_pose;
    double fitness_score;
};

std::optional<LoopIcpResult> doICPVirtualRelative( int _loop_kf_idx, int _curr_kf_idx )
{
    cout << "[SC loop] doICPVirtualRelative: " << _loop_kf_idx << " and " << _curr_kf_idx << endl;
    int cloudSize = 0;
    int poseSize = 0;
    Pose6D loop_pose, curr_pose;
    {
        std::lock_guard<std::mutex> lock(mKF);
        cloudSize = int(keyframeLaserClouds.size());
        poseSize = int(keyframePosesUpdated.size());
        if (_loop_kf_idx >= 0 && _loop_kf_idx < poseSize) {
            loop_pose = keyframePosesUpdated[_loop_kf_idx];
        }
        if (_curr_kf_idx >= 0 && _curr_kf_idx < poseSize) {
            curr_pose = keyframePosesUpdated[_curr_kf_idx];
        }
    }

    if (_loop_kf_idx < 0 || _curr_kf_idx < 0 ||
        _loop_kf_idx >= cloudSize || _curr_kf_idx >= cloudSize ||
        _loop_kf_idx >= poseSize || _curr_kf_idx >= poseSize) {
        std::cout << "[SC loop] invalid keyframe index pair: "
                  << _loop_kf_idx << " and " << _curr_kf_idx
                  << ", cloudSize=" << cloudSize << ", poseSize=" << poseSize << std::endl;
        return std::nullopt;
    }

    // parse pointclouds
    // Transform source cloud to curr frame's local coordinate
    // Transform target cloud to loop frame's local coordinate
    // ICP then finds the true relative pose between them
    int historyKeyframeSearchNum = 25; // enough. ex. [-25, 25] covers submap length of 50x1 = 50m if every kf gap is 1m
    pcl::PointCloud<PointType>::Ptr cureKeyframeCloud(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr targetKeyframeCloud(new pcl::PointCloud<PointType>());
    loopFindNearKeyframesCloud(cureKeyframeCloud, _curr_kf_idx, historyKeyframeSearchNum, _curr_kf_idx);
    loopFindNearKeyframesCloud(targetKeyframeCloud, _loop_kf_idx, historyKeyframeSearchNum, _loop_kf_idx); 

    if (cureKeyframeCloud->empty() || targetKeyframeCloud->empty()) {
        std::cout << "[SC loop] empty cloud pair: src=" << cureKeyframeCloud->size()
                  << ", tgt=" << targetKeyframeCloud->size() << ". Skip ICP." << std::endl;
        return std::nullopt;
    }

    const std::string srcLoopCloudPath = pgLoopCloudDirectory + "loop_src_" + std::to_string(_loop_kf_idx) + "_" + std::to_string(_curr_kf_idx) + ".pcd";
    const std::string tgtLoopCloudPath = pgLoopCloudDirectory + "loop_tgt_" + std::to_string(_loop_kf_idx) + "_" + std::to_string(_curr_kf_idx) + ".pcd";
    pcl::io::savePCDFileBinary(srcLoopCloudPath, *cureKeyframeCloud);
    pcl::io::savePCDFileBinary(tgtLoopCloudPath, *targetKeyframeCloud);

    // loop verification 
    sensor_msgs::PointCloud2 cureKeyframeCloudMsg;
    pcl::toROSMsg(*cureKeyframeCloud, cureKeyframeCloudMsg);
    cureKeyframeCloudMsg.header.frame_id = "camera_init";
    pubLoopScanLocal.publish(cureKeyframeCloudMsg);

    sensor_msgs::PointCloud2 targetKeyframeCloudMsg;
    pcl::toROSMsg(*targetKeyframeCloud, targetKeyframeCloudMsg);
    targetKeyframeCloudMsg.header.frame_id = "camera_init";
    pubLoopSubmapLocal.publish(targetKeyframeCloudMsg);

    // Compute initial guess from odometry estimate
    // T_loop_curr (odom) = T_loop^{-1} * T_curr
    Eigen::Affine3f transCurr = pcl::getTransformation(curr_pose.x, curr_pose.y, curr_pose.z,
                                                        curr_pose.roll, curr_pose.pitch, curr_pose.yaw);
    Eigen::Affine3f transLoop = pcl::getTransformation(loop_pose.x, loop_pose.y, loop_pose.z,
                                                        loop_pose.roll, loop_pose.pitch, loop_pose.yaw);
    Eigen::Matrix4f initialGuess = transLoop.matrix().inverse() * transCurr.matrix();

    // ICP Settings
    pcl::IterativeClosestPoint<PointType, PointType> icp;
    icp.setMaxCorrespondenceDistance(150); // giseop , use a value can cover 2*historyKeyframeSearchNum range in meter 
    icp.setMaximumIterations(100);
    icp.setTransformationEpsilon(1e-6);
    icp.setEuclideanFitnessEpsilon(1e-6);
    icp.setRANSACIterations(0);

    // Enforce overlap during ICP by trimming correspondences.
    const double overlapRatioClamped = std::max(0.05, std::min(1.0, icpTrimmedOverlapRatio));
    pcl::registration::CorrespondenceRejectorTrimmed::Ptr trimRejector(
        new pcl::registration::CorrespondenceRejectorTrimmed());
    trimRejector->setOverlapRatio(static_cast<float>(overlapRatioClamped));
    icp.addCorrespondenceRejector(trimRejector);

    // Align pointclouds with initial guess
    icp.setInputSource(cureKeyframeCloud);
    icp.setInputTarget(targetKeyframeCloud);
    pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
    icp.align(*unused_result, initialGuess);
 
    const double fitnessScore = icp.getFitnessScore();
    if (icp.hasConverged() == false || fitnessScore > loopFitnessScoreThreshold) {
        std::cout << "[SC loop] ICP fitness test failed (" << fitnessScore << " > " << loopFitnessScoreThreshold << "). Reject this SC loop." << std::endl;
        return std::nullopt;
    } else {
        std::cout << "[SC loop] ICP fitness test passed (" << fitnessScore << " < " << loopFitnessScoreThreshold << "). Add this SC loop." << std::endl;
    }

    // Get pose transformation
    float x, y, z, roll, pitch, yaw;
    Eigen::Affine3f correctionLidarFrame;
    correctionLidarFrame = icp.getFinalTransformation();
    pcl::getTranslationAndEulerAngles (correctionLidarFrame, x, y, z, roll, pitch, yaw);
    gtsam::Pose3 poseFrom = Pose3(Rot3::RzRyRx(roll, pitch, yaw), Point3(x, y, z));

    return LoopIcpResult{poseFrom, fitnessScore};
}

void process_pg()
{
    while (ros::ok())
    {
		while (ros::ok())
        {
            sensor_msgs::ImageConstPtr imageMsgForFrame;
            bool hasImageForFrame = false;

            mBuf.lock();
            if (odometryBuf.empty() || fullResBuf.empty()) {
                mBuf.unlock();
                break;
            }

            // Intentionally pair by FIFO order without timestamp gating.
            timeLaserOdometry = odometryBuf.front()->header.stamp.toSec();
            timeLaser = fullResBuf.front()->header.stamp.toSec();

            laserCloudFullRes->clear();
            pcl::PointCloud<PointType>::Ptr thisKeyFrame(new pcl::PointCloud<PointType>());
            pcl::fromROSMsg(*fullResBuf.front(), *thisKeyFrame);
            fullResBuf.pop();

            const nav_msgs::Odometry::ConstPtr odomMsg = odometryBuf.front();
            Pose6D pose_curr = getOdom(odomMsg);
            odometryBuf.pop();
            const uint64_t currentAllFrameId = allFrameCounter++;

            bool elevatorSceneThisFrame = false;
            if (!elevatorSceneBuf.empty()) {
                elevatorSceneThisFrame = elevatorSceneBuf.front()->data;
                elevatorSceneBuf.pop();
                latestElevatorScene = elevatorSceneThisFrame;
                hasElevatorSceneMsg = true;
            } else if (hasElevatorSceneMsg) {
                // Fallback to latest state if the queue is temporarily empty.
                elevatorSceneThisFrame = latestElevatorScene;
            }
            double currentFloorHeight = 0.0;
            bool useGravityAxisHeightForLog = false;
            if (enableNonElevatorFloorHeightTracking) {
                if (getCloudHeightSpanAlongGravityAxis(thisKeyFrame, currentFloorHeight)) {
                    useGravityAxisHeightForLog = true;
                } else {
                    getCloudHeightSpanAlongZ(thisKeyFrame, currentFloorHeight);
                }
                ROS_INFO_STREAM_THROTTLE(1.0,
                                         "[SC-PGO] current cloud height span=" << currentFloorHeight
                                         << (useGravityAxisHeightForLog ? " (gravity-axis max-min)" : " (z-axis max-min fallback)"));
            }

            if (enableNonElevatorFloorHeightTracking && !elevatorSceneThisFrame) {
                double gravityAxisHeight = 0.0;
                if (getHeightAlongGravityAxis(pose_curr, gravityAxisHeight)) {
                    updateTrackedFloorHeight(gravityAxisHeight);
                } else {
                    // Fallback when gravity is unavailable: keep legacy z-based tracking.
                    updateTrackedFloorHeight(pose_curr.z);
                }
            }

            // Determine elevator session transitions for this keyframe
            bool isElevatorEnterTransition = false;
            bool isElevatorExitTransition = false;
            if (hasPrevKeyframeElevatorScene) {
                isElevatorEnterTransition = (!prevKeyframeElevatorScene && elevatorSceneThisFrame);
                isElevatorExitTransition = (prevKeyframeElevatorScene && !elevatorSceneThisFrame);
            }
            // If this keyframe is right after exiting elevator, advance session
            if (isElevatorExitTransition) {
                currentSession++;
            }

            if (subscribedOdomSaveStream.is_open()) {
                gtsam::Pose3 pose = Pose6DtoGTSAMPose3(pose_curr);
                Point3 t = pose.translation();
                Rot3 R = pose.rotation();
                auto col1 = R.column(1);
                auto col2 = R.column(2);
                auto col3 = R.column(3);

                subscribedOdomSaveStream << std::fixed << std::setprecision(9)
                                         << col1.x() << " " << col2.x() << " " << col3.x() << " " << t.x() << " "
                                         << col1.y() << " " << col2.y() << " " << col3.y() << " " << t.y() << " "
                                         << col1.z() << " " << col2.z() << " " << col3.z() << " " << t.z()
                                         << std::endl;
            }

            if (subscribedOdomMetaSaveStream.is_open()) {
                const double syncDiff = timeLaserOdometry - timeLaser;
                subscribedOdomMetaSaveStream << currentAllFrameId << " "
                                             << std::fixed << std::setprecision(6)
                                             << timeLaserOdometry << " "
                                             << timeLaser << " "
                                             << syncDiff << std::endl;
            }

            if (!imageBuf.empty()) {
                latestImageMsg = imageBuf.front();
                imageBuf.pop();
            }
            if (latestImageMsg) {
                imageMsgForFrame = latestImageMsg;
                hasImageForFrame = true;
            }

            // find nearest gps 
            double eps = 0.1; // find a gps topioc arrived within eps second 
            while (!gpsBuf.empty()) {
                auto thisGPS = gpsBuf.front();
                auto thisGPSTime = thisGPS->header.stamp.toSec();
                if( abs(thisGPSTime - timeLaserOdometry) < eps ) {
                    currGPS = thisGPS;
                    hasGPSforThisKF = true; 
                    break;
                } else {
                    hasGPSforThisKF = false;
                }
                gpsBuf.pop();
            }
            mBuf.unlock(); 

            //
            // Early reject by counting local delta movement (for equi-spereated kf drop)
            // 
            odom_pose_prev = odom_pose_curr;
            odom_pose_curr = pose_curr;
            Pose6D dtf = diffTransformation(odom_pose_prev, odom_pose_curr); // dtf means delta_transform

            double delta_translation = sqrt(dtf.x*dtf.x + dtf.y*dtf.y + dtf.z*dtf.z); // note: absolute value. 
            translationAccumulated += delta_translation;
            rotaionAccumulated += (dtf.roll + dtf.pitch + dtf.yaw); // sum just naive approach.  

            if( translationAccumulated > keyframeMeterGap || rotaionAccumulated > keyframeRadGap ) {
                isNowKeyFrame = true;
                translationAccumulated = 0.0; // reset 
                rotaionAccumulated = 0.0; // reset 
            } else {
                isNowKeyFrame = false;
            }

            if( ! isNowKeyFrame ) 
                continue; 

            if( !gpsOffsetInitialized ) {
                if(hasGPSforThisKF) { // if the very first frame 
                    gpsAltitudeInitOffset = currGPS->altitude;
                    gpsOffsetInitialized = true;
                } 
            }

            //
            // Save data and Add consecutive node 
            //
            pcl::PointCloud<PointType>::Ptr thisKeyFrameDS(new pcl::PointCloud<PointType>());
            // downSizeFilterScancontext.setInputCloud(thisKeyFrame);
            // downSizeFilterScancontext.filter(*thisKeyFrameDS);
            *thisKeyFrameDS = *thisKeyFrame;

            // Keep map keyframe cloud unchanged; gravity alignment is only for Scan Context input.
            pcl::PointCloud<PointType>::Ptr thisKeyFrameForSC = thisKeyFrameDS;

            if (useKfGravityAlignForSC) {
                Eigen::Vector3d gravityNow;
                if (getLatestKfGravity(gravityNow)) {
                    thisKeyFrameForSC.reset(new pcl::PointCloud<PointType>(*thisKeyFrameDS));
                    alignPointCloudToGravityZ(gravityNow, thisKeyFrameForSC);
                }
            }

            int prev_node_idx = -1;
            int curr_node_idx = -1;
            Pose6D pose_prev_snapshot;
            Pose6D pose_curr_snapshot;
            {
                std::lock_guard<std::mutex> lock(mKF);
                keyframeLaserClouds.push_back(thisKeyFrameDS);
                keyframePoses.push_back(pose_curr);
                keyframePosesUpdated.push_back(pose_curr); // init
                keyframeTimes.push_back(timeLaserOdometry);
                keyframeSourceFrameIds.push_back(currentAllFrameId);
                keyframeSessions.push_back(currentSession);

                scManager.makeAndSaveScancontextAndKeys(*thisKeyFrameForSC);

                laserCloudMapPGORedraw = true;
                curr_node_idx = int(keyframePoses.size()) - 1;
                prev_node_idx = curr_node_idx - 1;
                pose_curr_snapshot = keyframePoses.at(curr_node_idx);
                if (prev_node_idx >= 0) {
                    pose_prev_snapshot = keyframePoses.at(prev_node_idx);
                }
            }

            if (isElevatorEnterTransition) {
                elevatorEnterNodeIdx = curr_node_idx;
                Eigen::Vector3d gravityVec;
                if (getLatestKfGravity(gravityVec)) {
                    const double gNorm = gravityVec.norm();
                    if (gNorm > 1e-6) {
                        elevatorEnterUpAxis = -gravityVec / gNorm;
                        hasElevatorEnterUpAxis = true;
                    } else {
                        hasElevatorEnterUpAxis = false;
                    }
                } else {
                    hasElevatorEnterUpAxis = false;
                }
            }

            if( ! gtSAMgraphMade /* prior node */) {
                const int init_node_idx = 0; 
                gtsam::Pose3 poseOrigin = Pose6DtoGTSAMPose3(pose_curr_snapshot);
                // auto poseOrigin = gtsam::Pose3(gtsam::Rot3::RzRyRx(0.0, 0.0, 0.0), gtsam::Point3(0.0, 0.0, 0.0));

                mtxPosegraph.lock();
                {
                    // prior factor 
                    gtSAMgraph.add(gtsam::PriorFactor<gtsam::Pose3>(init_node_idx, poseOrigin, priorNoise));
                    initialEstimate.insert(init_node_idx, poseOrigin);
                    // runISAM2opt();          
                }   
                mtxPosegraph.unlock();

                gtSAMgraphMade = true; 

                cout << "posegraph prior node " << init_node_idx << " added" << endl;
            } else /* consecutive node (and odom factor) after the prior added */ { // == keyframePoses.size() > 1 
                gtsam::Pose3 poseFrom = Pose6DtoGTSAMPose3(pose_prev_snapshot);
                gtsam::Pose3 poseTo = Pose6DtoGTSAMPose3(pose_curr_snapshot);

                mtxPosegraph.lock();
                {
                    // odom factor
                    const auto odomFactorNoise = elevatorSceneThisFrame ? degenerateNoise : odomNoise;
                    gtSAMgraph.add(gtsam::BetweenFactor<gtsam::Pose3>(prev_node_idx, curr_node_idx, poseFrom.between(poseTo), odomFactorNoise));

                    if (useTrackedFloorHeightFactorInElevator && isElevatorExitTransition && elevatorEnterNodeIdx >= 0) {
                        Eigen::Vector3d upAxis;
                        bool hasUpAxis = false;
                        if (hasElevatorEnterUpAxis) {
                            upAxis = elevatorEnterUpAxis;
                            hasUpAxis = true;
                        } else {
                            Eigen::Vector3d gravityVec;
                            if (getLatestKfGravity(gravityVec)) {
                                const double gNorm = gravityVec.norm();
                                if (gNorm > 1e-6) {
                                    upAxis = -gravityVec / gNorm;
                                    hasUpAxis = true;
                                }
                            }
                        }

                        if (hasUpAxis) {
                            const int floorDelta = elevatorTargetFloor - elevatorCurrentFloor;
                            const double targetHeightDelta = elevatorFloorHeight * static_cast<double>(floorDelta);
                            gtSAMgraph.add(std::shared_ptr<GravityAxisHeightDifferenceFactor>(
                                new GravityAxisHeightDifferenceFactor(elevatorEnterNodeIdx,
                                                                      curr_node_idx,
                                                                      upAxis,
                                                                      targetHeightDelta,
                                                                      floorHeightNoise)));
                            cout << "Elevator height-delta factor added between nodes "
                                 << elevatorEnterNodeIdx << " and " << curr_node_idx
                                 << ", floor from " << elevatorCurrentFloor
                                 << " to " << elevatorTargetFloor
                                 << ", target delta = " << targetHeightDelta << endl;
                        }
                    }

                    // gps factor 
                    if(hasGPSforThisKF) {
                        double curr_altitude_offseted = currGPS->altitude - gpsAltitudeInitOffset;
                        mtxRecentPose.lock();
                        gtsam::Point3 gpsConstraint(recentOptimizedX, recentOptimizedY, curr_altitude_offseted); // in this example, only adjusting altitude (for x and y, very big noises are set) 
                        mtxRecentPose.unlock();
                        gtSAMgraph.add(gtsam::GPSFactor(curr_node_idx, gpsConstraint, robustGPSNoise));
                        cout << "GPS factor added at node " << curr_node_idx << endl;
                    }
                    initialEstimate.insert(curr_node_idx, poseTo);                
                    // runISAM2opt();
                }
                mtxPosegraph.unlock();

                if(curr_node_idx % 100 == 0)
                    cout << "posegraph odom node " << curr_node_idx << " added." << endl;
            }

            if (isElevatorExitTransition) {
                elevatorEnterNodeIdx = -1;
                hasElevatorEnterUpAxis = false;
            }
            prevKeyframeElevatorScene = elevatorSceneThisFrame;
            hasPrevKeyframeElevatorScene = true;

            // if want to print the current graph, use gtSAMgraph.print("\nFactor Graph:\n");

            // save utility 
            std::string curr_node_idx_str = padZeros(curr_node_idx);
            // pcl::io::savePCDFileBinary(pgScansDirectory + curr_node_idx_str + ".pcd", *thisKeyFrame); // raw scan
            if (saveSCKeyframeToDisk) {
                pcl::io::savePCDFileBinary(pgScansDirectory + curr_node_idx_str + "_sc.pcd", *thisKeyFrameForSC);
            }
            if (saveKeyframeImageToDisk && hasImageForFrame && imageMsgForFrame && !imageMsgForFrame->data.empty()) {
                const std::string imageSavePath = keyframeImageDirectory + curr_node_idx_str + ".png";
                try {
                    cv_bridge::CvImageConstPtr cvPtr;
                    if (imageMsgForFrame->encoding == sensor_msgs::image_encodings::BGR8) {
                        cvPtr = cv_bridge::toCvShare(imageMsgForFrame, sensor_msgs::image_encodings::BGR8);
                    } else {
                        cvPtr = cv_bridge::toCvCopy(imageMsgForFrame, sensor_msgs::image_encodings::BGR8);
                    }
                    if (!cv::imwrite(imageSavePath, cvPtr->image)) {
                        ROS_WARN_STREAM("[SC-PGO] cv::imwrite failed for keyframe image " << imageSavePath);
                    }
                } catch (const cv_bridge::Exception &e) {
                    ROS_WARN_STREAM("[SC-PGO] cv_bridge conversion failed for keyframe image: " << e.what());
                }
            }
            pgTimeSaveStream << timeLaser << std::endl; // path 
            if (keyframeFrameIdSaveStream.is_open()) {
                keyframeFrameIdSaveStream << curr_node_idx << " "
                                          << currentAllFrameId << " "
                                          << std::fixed << std::setprecision(6) << timeLaserOdometry << std::endl;
            }
        }

        // ps. 
        // scan context detector is running in another thread (in constant Hz, e.g., 1 Hz)
        // pub path and point cloud in another thread

        // wait (must required for running the while loop)
        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
    }
} // process_pg

void performSCLoopClosure(void)
{
    int prev_node_idx = -1;
    int curr_node_idx = -1;
    double prev_node_time_relative = -1.0;
    double curr_node_time_relative = -1.0;
    bool hasRelativeTimes = false;

    // Protect ScanContext internals and keyframe metadata from concurrent updates.
    mKF.lock();
    const int currentKeyframeIdx = int(keyframePoses.size()) - 1;
    if (currentKeyframeIdx < 0 || currentKeyframeIdx == lastLoopDetectCurrIdx) {
        mKF.unlock();
        return;
    }
    lastLoopDetectCurrIdx = currentKeyframeIdx;

    if( int(keyframePoses.size()) >= scManager.NUM_EXCLUDE_RECENT) {
        auto detectResult = scManager.detectLoopClosureID(); // first: nn index, second: yaw diff
        int SCclosestHistoryFrameID = detectResult.first;
        if( SCclosestHistoryFrameID != -1 ) {
            prev_node_idx = SCclosestHistoryFrameID;
            curr_node_idx = currentKeyframeIdx; // because cpp starts 0 and ends n-1

            if (prev_node_idx >= 0 && curr_node_idx >= 0 &&
                prev_node_idx < int(keyframeTimes.size()) && curr_node_idx < int(keyframeTimes.size())) {
                const double start_time = keyframeTimes.front();
                prev_node_time_relative = keyframeTimes.at(prev_node_idx) - start_time;
                curr_node_time_relative = keyframeTimes.at(curr_node_idx) - start_time;
                hasRelativeTimes = true;
            }
        }
    }
    mKF.unlock();

    

    if(prev_node_idx != -1 && curr_node_idx != -1) {
        /* const double time_interval = curr_node_time_relative - prev_node_time_relative; */
        const double time_interval = curr_node_time_relative - prev_node_time_relative;

        if (scDetectAllPairsSaveStream.is_open() && hasRelativeTimes) {
            scDetectAllPairsSaveStream << prev_node_idx << " "
                                       << curr_node_idx << " "
                                       << std::fixed << std::setprecision(6)
                                       << prev_node_time_relative << " "
                                       << curr_node_time_relative << " "
                                       << time_interval << std::endl;
        } else if (!hasRelativeTimes) {
            ROS_WARN_STREAM("[SC loop] skip writing sc_detect_all_pairs due to invalid relative time for pair "
                            << prev_node_idx << " " << curr_node_idx);
        }

        cout << "Loop detected! - between " << prev_node_idx << " and " << curr_node_idx 
                 << " (time interval: " << time_interval << "s)" << endl;
        if (curr_node_idx - prev_node_idx > 50) {
            bool differentSession = true;
            mKF.lock();
            if (prev_node_idx >= 0 && prev_node_idx < int(keyframeSessions.size()) &&
                curr_node_idx >= 0 && curr_node_idx < int(keyframeSessions.size())) {
                differentSession = (keyframeSessions.at(prev_node_idx) != keyframeSessions.at(curr_node_idx));
            }
            mKF.unlock();

            if (differentSession) {
                mBuf.lock();
                scLoopICPBuf.push(std::pair<int, int>(prev_node_idx, curr_node_idx));
                // adding actual 6D constraints in the other thread, icp_calculation.
                mBuf.unlock();
            } else {
                cout << "Skip loop: same session between " << prev_node_idx << " and " << curr_node_idx << endl;
            }
        }
        else {
            cout << "node index gap: " << curr_node_idx - prev_node_idx << endl;
        }
    }
} // performSCLoopClosure

void process_lcd(void)
{
    float loopClosureFrequency = 1.0; // can change 
    ros::Rate rate(loopClosureFrequency);
    while (ros::ok())
    {
        rate.sleep();
        performSCLoopClosure();
        // performRSLoopClosure(); // TODO
    }
} // process_lcd

void process_icp(void)
{
    while (ros::ok())
    {
		while (ros::ok())
        {
            mBuf.lock();
            if (scLoopICPBuf.empty()) {
                mBuf.unlock();
                break;
            }

            if( scLoopICPBuf.size() > 30 ) {
                ROS_WARN("Too many loop clousre candidates to be ICPed is waiting ... Do process_lcd less frequently (adjust loopClosureFrequency)");
            }

            std::pair<int, int> loop_idx_pair = scLoopICPBuf.front();
            scLoopICPBuf.pop();
            mBuf.unlock(); 

            const int prev_node_idx = loop_idx_pair.first;
            const int curr_node_idx = loop_idx_pair.second;
            // Double-check sessions differ before expensive ICP
            {
                bool differentSession = true;
                std::lock_guard<std::mutex> lock(mKF);
                if (prev_node_idx >= 0 && prev_node_idx < int(keyframeSessions.size()) &&
                    curr_node_idx >= 0 && curr_node_idx < int(keyframeSessions.size())) {
                    differentSession = (keyframeSessions.at(prev_node_idx) != keyframeSessions.at(curr_node_idx));
                }
                if (!differentSession) {
                    cout << "Process ICP: skip candidate because same session " << prev_node_idx << " " << curr_node_idx << endl;
                    continue;
                }
            }
            auto relative_pose_optional = doICPVirtualRelative(prev_node_idx, curr_node_idx);
            if(relative_pose_optional) {
                const LoopIcpResult& loopIcp = relative_pose_optional.value();
                const gtsam::Pose3& relative_pose = loopIcp.relative_pose;
                mtxPosegraph.lock();
                gtSAMgraph.add(gtsam::BetweenFactor<gtsam::Pose3>(prev_node_idx, curr_node_idx, relative_pose, robustLoopNoise));
                // runISAM2opt();
                mtxPosegraph.unlock();

                double prev_node_time_relative = -1.0;
                double curr_node_time_relative = -1.0;
                {
                    std::lock_guard<std::mutex> lock(mKF);
                    if (prev_node_idx >= 0 && curr_node_idx >= 0 &&
                        prev_node_idx < int(keyframeTimes.size()) && curr_node_idx < int(keyframeTimes.size())) {
                        const double start_time = keyframeTimes.front();
                        prev_node_time_relative = keyframeTimes.at(prev_node_idx) - start_time;
                        curr_node_time_relative = keyframeTimes.at(curr_node_idx) - start_time;
                    }
                }

                if (loopTimePairSaveStream.is_open()) {
                    constexpr int kLoopLogFieldWidth = 12;
                    const gtsam::Matrix4 relativePoseMat = relative_pose.matrix();
                    loopTimePairSaveStream << std::setw(kLoopLogFieldWidth) << prev_node_idx
                                           << std::setw(kLoopLogFieldWidth) << curr_node_idx
                                           << std::setw(kLoopLogFieldWidth) << std::fixed << std::setprecision(1) << prev_node_time_relative
                                           << std::setw(kLoopLogFieldWidth) << std::fixed << std::setprecision(1) << curr_node_time_relative
                                           << std::setw(kLoopLogFieldWidth) << std::defaultfloat << std::setprecision(6) << loopIcp.fitness_score
                                           << std::setw(kLoopLogFieldWidth) << std::fixed << std::setprecision(9) << relativePoseMat(0, 0)
                                           << std::setw(kLoopLogFieldWidth) << std::fixed << std::setprecision(9) << relativePoseMat(0, 1)
                                           << std::setw(kLoopLogFieldWidth) << std::fixed << std::setprecision(9) << relativePoseMat(0, 2)
                                           << std::setw(kLoopLogFieldWidth) << std::fixed << std::setprecision(9) << relativePoseMat(0, 3)
                                           << std::setw(kLoopLogFieldWidth) << std::fixed << std::setprecision(9) << relativePoseMat(1, 0)
                                           << std::setw(kLoopLogFieldWidth) << std::fixed << std::setprecision(9) << relativePoseMat(1, 1)
                                           << std::setw(kLoopLogFieldWidth) << std::fixed << std::setprecision(9) << relativePoseMat(1, 2)
                                           << std::setw(kLoopLogFieldWidth) << std::fixed << std::setprecision(9) << relativePoseMat(1, 3)
                                           << std::setw(kLoopLogFieldWidth) << std::fixed << std::setprecision(9) << relativePoseMat(2, 0)
                                           << std::setw(kLoopLogFieldWidth) << std::fixed << std::setprecision(9) << relativePoseMat(2, 1)
                                           << std::setw(kLoopLogFieldWidth) << std::fixed << std::setprecision(9) << relativePoseMat(2, 2)
                                           << std::setw(kLoopLogFieldWidth) << std::fixed << std::setprecision(9) << relativePoseMat(2, 3)
                                           << std::endl;
                }
            } 
        }

        // wait (must required for running the while loop)
        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
    }
} // process_icp

void process_viz_path(void)
{
    float hz = 10.0; 
    ros::Rate rate(hz);
    while (ros::ok()) {
        rate.sleep();
        if(recentIdxUpdated > 1) {
            pubPath();
        }
    }
}

void process_isam(void)
{
    float hz = 1; 
    ros::Rate rate(hz);
    while (ros::ok()) {
        rate.sleep();
        if( gtSAMgraphMade ) {
            bool hasPendingUpdates = false;
            mtxPosegraph.lock();
            hasPendingUpdates = (!gtSAMgraph.empty() || !initialEstimate.empty());
            if (hasPendingUpdates) {
                runISAM2opt();
                cout << "running isam2 optimization ..." << endl;
            }
            mtxPosegraph.unlock();

            if (hasPendingUpdates) {
                saveOptimizedVerticesKITTIformat(isamCurrentEstimate, pgKITTIformat); // pose
                saveOdometryVerticesKITTIformat(odomKITTIformat); // pose
            }
        }
    }
}

void pubMap(void)
{
    std::vector<std::pair<pcl::PointCloud<PointType>::Ptr, Pose6D>> mapFrames;
    mapFrames.reserve(512);

    int recentIdxSnapshot = 0;
    mtxRecentPose.lock();
    recentIdxSnapshot = recentIdxUpdated;
    mtxRecentPose.unlock();

    mKF.lock();
    const int mapFrameCount = std::min(recentIdxSnapshot,
                                       std::min(int(keyframeLaserClouds.size()), int(keyframePosesUpdated.size())));
    for (int node_idx = 0; node_idx < mapFrameCount; node_idx++) {
        mapFrames.emplace_back(keyframeLaserClouds[node_idx], keyframePosesUpdated[node_idx]);
    }
    mKF.unlock();

    pcl::PointCloud<PointType>::Ptr mapCloudRaw(new pcl::PointCloud<PointType>());
    for (const auto& frame : mapFrames) {
        const auto& cloudPtr = frame.first;
        if (!cloudPtr || cloudPtr->empty()) {
            continue;
        }
        pcl::PointCloud<PointType>::Ptr transformedCloud = local2global(cloudPtr, frame.second);
        if (!transformedCloud || transformedCloud->empty()) {
            continue;
        }
        *mapCloudRaw += *transformedCloud;
    }

    *laserCloudMapPGO = *mapCloudRaw;

    sensor_msgs::PointCloud2 laserCloudMapPGOMsg;
    pcl::toROSMsg(*laserCloudMapPGO, laserCloudMapPGOMsg);
    laserCloudMapPGOMsg.header.frame_id = "camera_init";
    pubMapAftPGO.publish(laserCloudMapPGOMsg);

    if (saveMapToDisk && !laserCloudMapPGO->empty()) {
        static ros::Time lastMapSaveStamp(0);
        const ros::Time now = ros::Time::now();
        if (lastMapSaveStamp.isZero() || (now - lastMapSaveStamp).toSec() >= mapSaveIntervalSec) {
            std::string mapSavePath = save_directory;
            if (!mapSavePath.empty() && mapSavePath.back() != '/') {
                mapSavePath += "/";
            }
            mapSavePath += "aft_pgo_map_final.pcd";

            const int saveResult = pcl::io::savePCDFileBinary(mapSavePath, *laserCloudMapPGO);
            if (saveResult == 0) {
                ROS_INFO_STREAM("[SC-PGO] Saved map to " << mapSavePath
                                 << " (points=" << laserCloudMapPGO->size() << ")");
                lastMapSaveStamp = now;
            } else {
                ROS_WARN_STREAM("[SC-PGO] Failed to save map to " << mapSavePath
                                 << ", error code=" << saveResult);
            }
        }
    }
}

void process_viz_map(void)
{
    float vizmapFrequency = 0.1; // 0.1 means run onces every 10s
    ros::Rate rate(vizmapFrequency);
    while (ros::ok()) {
        rate.sleep();
        int recentIdxSnapshot = 0;
        mtxRecentPose.lock();
        recentIdxSnapshot = recentIdxUpdated;
        mtxRecentPose.unlock();
        if(recentIdxSnapshot > 1) {
            pubMap();
        }
    }
} // pointcloud_viz

void process_kf_gravity_save(void)
{
    while (ros::ok()) {
        while (ros::ok()) {
            geometry_msgs::Vector3Stamped::ConstPtr gravityMsg;

            mBuf.lock();
            if (kfGravityBuf.empty()) {
                mBuf.unlock();
                break;
            }

            gravityMsg = kfGravityBuf.front();
            kfGravityBuf.pop();
            mBuf.unlock();

            if (subscribedKfGravitySaveStream.is_open() && gravityMsg) {
                subscribedKfGravitySaveStream << std::fixed << std::setprecision(9)
                                              << gravityMsg->header.stamp.toSec() << " "
                                              << gravityMsg->vector.x << " "
                                              << gravityMsg->vector.y << " "
                                              << gravityMsg->vector.z << std::endl;
            }
        }

        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
    }
}

int main(int argc, char **argv)
{
	ros::init(argc, argv, "laserPGO");
	ros::NodeHandle nh;

	nh.param<std::string>("save_directory", save_directory, "/"); // pose assignment every k m move 
    nh.param<bool>("save_map_to_disk", saveMapToDisk, true);
    nh.param<bool>("save_sc_keyframe_to_disk", saveSCKeyframeToDisk, true);
    nh.param<double>("map_save_interval_sec", mapSaveIntervalSec, 10.0);
    pgKITTIformat = save_directory + "optimized_poses.txt";
    odomKITTIformat = save_directory + "odom_poses.txt";
    pgTimeSaveStream = std::fstream(save_directory + "times.txt", std::fstream::out); 
    pgTimeSaveStream.precision(std::numeric_limits<double>::max_digits10);
    loopTimePairSaveStream = std::fstream(save_directory + "loop_times.txt", std::fstream::out);
    loopTimePairSaveStream.precision(std::numeric_limits<double>::max_digits10);
    if (loopTimePairSaveStream.is_open()) {
        loopTimePairSaveStream << "# prev_idx curr_idx prev_time curr_time fitness "
                               << "r00 r01 r02 tx r10 r11 r12 ty r20 r21 r22 tz" << std::endl;
    }
    scDetectAllPairsSaveStream = std::fstream(save_directory + "sc_detect_all_pairs.txt", std::fstream::out);
    scDetectAllPairsSaveStream.precision(std::numeric_limits<double>::max_digits10);
    if (scDetectAllPairsSaveStream.is_open()) {
        scDetectAllPairsSaveStream << "# prev_idx curr_idx prev_time curr_time time_interval" << std::endl;
    }
    keyframeFrameIdSaveStream = std::fstream(save_directory + "keyframe_allframe_ids.txt", std::fstream::out);
    if (keyframeFrameIdSaveStream.is_open()) {
        keyframeFrameIdSaveStream << "# keyframe_idx all_frame_id stamp" << std::endl;
    }
    subscribedOdomSaveStream = std::fstream(save_directory + "subscribed_odom_allframes_kitti.txt", std::fstream::out);
    if (subscribedOdomSaveStream.is_open()) {
        subscribedOdomSaveStream << "# KITTI 3x4: r00 r01 r02 tx r10 r11 r12 ty r20 r21 r22 tz" << std::endl;
    }
    subscribedOdomMetaSaveStream = std::fstream(save_directory + "subscribed_odom_allframes_meta.txt", std::fstream::out);
    if (subscribedOdomMetaSaveStream.is_open()) {
        subscribedOdomMetaSaveStream << "# all_frame_id odom_stamp cloud_stamp sync_diff" << std::endl;
    }
    subscribedKfGravitySaveStream = std::fstream(save_directory + "subscribed_kf_gravity_allframes.txt", std::fstream::out);
    if (subscribedKfGravitySaveStream.is_open()) {
        subscribedKfGravitySaveStream << "# gravity_stamp gx gy gz" << std::endl;
    }
    pgScansDirectory = save_directory + "Scans/";
    auto unused = system((std::string("exec rm -r ") + pgScansDirectory).c_str());
    unused = system((std::string("mkdir -p ") + pgScansDirectory).c_str());

    pgLoopCloudDirectory = save_directory + "LoopClouds/";
    unused = system((std::string("exec rm -r ") + pgLoopCloudDirectory).c_str());
    unused = system((std::string("mkdir -p ") + pgLoopCloudDirectory).c_str());

	nh.param<double>("keyframe_meter_gap", keyframeMeterGap, 1.0); // pose assignment every k m move 
	nh.param<double>("keyframe_deg_gap", keyframeDegGap, 10.0); // pose assignment every k deg rot 
    keyframeRadGap = deg2rad(keyframeDegGap);

	nh.param<double>("sc_dist_thres", scDistThres, 0.2);  
	nh.param<double>("sc_max_radius", scMaximumRadius, 80.0); // 80 is recommended for outdoor, and lower (ex, 20, 40) values are recommended for indoor 
    nh.param<double>("loop_fitness_score_threshold", loopFitnessScoreThreshold, 0.3);
    nh.param<double>("icp_trimmed_overlap_ratio", icpTrimmedOverlapRatio, 0.8);
    nh.param<bool>("use_kf_gravity_align_for_sc", useKfGravityAlignForSC, true);
    nh.param<std::string>("kf_gravity_topic", kfGravityTopic, std::string("/kf_gravity"));
    nh.param<std::string>("elevator_scene_topic", elevatorSceneTopic, std::string("/lio/elevator_scene"));
    nh.param<bool>("enable_non_elevator_floor_height_tracking", enableNonElevatorFloorHeightTracking, false);
    nh.param<double>("non_elevator_floor_height_alpha", nonElevatorFloorHeightAlpha, 0.1);
    nh.param<bool>("use_tracked_floor_height_factor_in_elevator", useTrackedFloorHeightFactorInElevator, false);
    nh.param<double>("floor_height_factor_sigma", floorHeightFactorSigma, 0.2);
    nh.param<double>("elevator_floor_height", elevatorFloorHeight, 3.0);
    // Backward compatibility: use elevator_floor_count to fill missing floor IDs.
    nh.param<int>("elevator_floor_count", elevatorFloorCount, 1);
    const bool hasElevatorCurrentFloorParam = nh.getParam("elevator_current_floor", elevatorCurrentFloor);
    const bool hasElevatorTargetFloorParam = nh.getParam("elevator_target_floor", elevatorTargetFloor);
    const bool hasBothFloorParams = hasElevatorCurrentFloorParam && hasElevatorTargetFloorParam;
    if (!hasElevatorCurrentFloorParam) {
        elevatorCurrentFloor = 0;
    }
    if (!hasElevatorTargetFloorParam) {
        elevatorTargetFloor = elevatorCurrentFloor + elevatorFloorCount;
    }
    if (!hasBothFloorParams) {
        // Require both floor IDs to enable this factor. Otherwise disable to avoid ambiguous sign/magnitude.
        useTrackedFloorHeightFactorInElevator = false;
        ROS_WARN_STREAM("[SC-PGO] elevator_current_floor and elevator_target_floor must both be provided. "
                        "Floor-height factor is disabled.");
    }
    nonElevatorFloorHeightAlpha = std::max(0.0, std::min(1.0, nonElevatorFloorHeightAlpha));
    floorHeightFactorSigma = std::max(1e-4, floorHeightFactorSigma);
    ROS_INFO_STREAM("[SC-PGO] Elevator floor config: height=" << elevatorFloorHeight
                    << ", current=" << elevatorCurrentFloor
                    << ", target=" << elevatorTargetFloor
                    << ", floor_delta=" << (elevatorTargetFloor - elevatorCurrentFloor)
                    << ", height_factor_enabled=" << (useTrackedFloorHeightFactorInElevator ? "true" : "false"));
    nh.param<bool>("save_keyframe_image_to_disk", saveKeyframeImageToDisk, false);
    nh.param<std::string>("image_topic", imageTopic, std::string("/rgb_img"));
    // Backward compatibility: allow old parameter name to override image_topic.
    nh.param<std::string>("compressed_image_topic", imageTopic, imageTopic);

    ISAM2Params parameters;
    parameters.relinearizeThreshold = 0.01;
    parameters.relinearizeSkip = 1;
    isam = new ISAM2(parameters);
    initNoises();

    scManager.setSCdistThres(scDistThres);
    scManager.setMaximumRadius(scMaximumRadius);

    float filter_size = 0.4; 
    downSizeFilterScancontext.setLeafSize(filter_size, filter_size, filter_size);
    downSizeFilterICP.setLeafSize(filter_size, filter_size, filter_size);
    icpSubmapLeafSize = filter_size;
    nh.param<double>("icp_submap_leaf_size", icpSubmapLeafSize, double(filter_size));

    double mapVizFilterSize;
	nh.param<double>("mapviz_filter_size", mapVizFilterSize, 0.4); // pose assignment every k frames 
    mapVizLeafSize = mapVizFilterSize;
    downSizeFilterMapPGO.setLeafSize(mapVizFilterSize, mapVizFilterSize, mapVizFilterSize);

	ros::Subscriber subLaserCloudFullRes = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_cloud_registered_local", 200000, laserCloudFullResHandler);
	ros::Subscriber subLaserOdometry = nh.subscribe<nav_msgs::Odometry>("/aft_mapped_to_init", 20000, laserOdometryHandler);
    ros::Subscriber subImage = nh.subscribe<sensor_msgs::Image>(imageTopic, 200000, imageHandler);
	ros::Subscriber subGPS = nh.subscribe<sensor_msgs::NavSatFix>("/gps/fix", 20000, gpsHandler);
    ros::Subscriber subElevatorScene = nh.subscribe<std_msgs::Bool>(elevatorSceneTopic, 20000, elevatorSceneHandler);
    ros::Subscriber subKfGravity = nh.subscribe<geometry_msgs::Vector3Stamped>(kfGravityTopic, 20000, kfGravityHandler);

	pubOdomAftPGO = nh.advertise<nav_msgs::Odometry>("/aft_pgo_odom", 100);
	pubOdomRepubVerifier = nh.advertise<nav_msgs::Odometry>("/repub_odom", 100);
	pubPathAftPGO = nh.advertise<nav_msgs::Path>("/aft_pgo_path", 100);
	pubMapAftPGO = nh.advertise<sensor_msgs::PointCloud2>("/aft_pgo_map", 100);

	pubLoopScanLocal = nh.advertise<sensor_msgs::PointCloud2>("/loop_scan_local", 100);
	pubLoopSubmapLocal = nh.advertise<sensor_msgs::PointCloud2>("/loop_submap_local", 100);

    keyframeImageDirectory = save_directory + "KeyframeImages/";
    if (saveKeyframeImageToDisk) {
        unused = system((std::string("exec rm -r ") + keyframeImageDirectory).c_str());
        unused = system((std::string("mkdir -p ") + keyframeImageDirectory).c_str());
    }

	std::thread posegraph_slam {process_pg}; // pose graph construction
	std::thread lc_detection {process_lcd}; // loop closure detection 
	std::thread icp_calculation {process_icp}; // loop constraint calculation via icp 
	std::thread isam_update {process_isam}; // if you want to call less isam2 run (for saving redundant computations and no real-time visulization is required), uncommment this and comment all the above runisam2opt when node is added. 
    std::thread gravity_save {process_kf_gravity_save}; // persist all subscribed gravity messages to file

    std::thread viz_map {process_viz_map}; // Disabled for debugging potential race in map visualization.
	std::thread viz_path {process_viz_path}; // visualization - path (high frequency)

 	ros::spin();

    if (posegraph_slam.joinable()) posegraph_slam.join();
    if (lc_detection.joinable()) lc_detection.join();
    if (icp_calculation.joinable()) icp_calculation.join();
    if (isam_update.joinable()) isam_update.join();
    if (gravity_save.joinable()) gravity_save.join();
    if (viz_map.joinable()) viz_map.join();
    if (viz_path.joinable()) viz_path.join();

	return 0;
}
