#include "HMD_AbstractTracker.hpp"
#include "HMD_TrackerTM.hpp"
//#include "HMD_TrackerPF.hpp"

using namespace cv;
using namespace std;

/// global variables used in this file
static set<T_HANDLE> g_active_trakcers;

T_HANDLE CreateVideoTracker( int type )
{
	AbstractTracker* pTracker = NULL;

	switch( type )
	{
	case 0:

		pTracker = new TrackerTM();
		break;

	case 1:

		//pTracker = new TrackerPF();
		//break;

	default:
		break;
	}

	if (pTracker)
	{
		g_active_trakcers.insert( (T_HANDLE) pTracker );

	}
	return (T_HANDLE) pTracker;
}



bool DeleteVideoTracker( T_HANDLE handle )
{
	set<T_HANDLE>::iterator itr = g_active_trakcers.find( handle );

	if (itr == g_active_trakcers.end())
		return false;

	delete (AbstractTracker*) (*itr);
	g_active_trakcers.erase( itr );

	return true;
}



int SetTrackingTarget( T_HANDLE handle, const cv::Mat& image, const cv::Rect& target )
{
	set<T_HANDLE>::iterator itr = g_active_trakcers.find( handle );

	if (itr == g_active_trakcers.end())
		return false;

	AbstractTracker* pTracker = (AbstractTracker*) (*itr);

	return pTracker->setTrackingObject(image, target);
}




int AddTrackingTarget( T_HANDLE handle, const cv::Mat& image, const cv::Rect& target )
{
	set<T_HANDLE>::iterator itr = g_active_trakcers.find( handle );

	if (itr == g_active_trakcers.end())
		return false;

	AbstractTracker* pTracker = (AbstractTracker*) (*itr);

	return pTracker->addTrackingObject(image, target);
}



bool RemoveTrackingTarget( T_HANDLE handle, const int& obj_id )
{
	set<T_HANDLE>::iterator itr = g_active_trakcers.find( handle );

	if (itr == g_active_trakcers.end())
		return false;

	AbstractTracker* pTracker = (AbstractTracker*) (*itr);

	return pTracker->remTrackingObject(obj_id);
}



bool RunTargetTracking( T_HANDLE handle, const cv::Mat& image, std::map<int,cv::Rect>& targets )
{
	set<T_HANDLE>::iterator itr = g_active_trakcers.find( handle );

	if (itr == g_active_trakcers.end())
		return false;

	AbstractTracker* pTracker = (AbstractTracker*) (*itr);

	return pTracker->runObjectTrackingAlgorithm(image, targets);
}

