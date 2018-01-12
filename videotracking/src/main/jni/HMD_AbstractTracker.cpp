#include "HMD_AbstractTracker.hpp"
#include "HMD_TrackerTM.hpp"
#include "HMD_TrackerKCF.hpp"
#include "HMD_TrackerCMT.hpp"
#include "HMD_TrackerCSK.hpp"
#include "HMD_TrackerDAT.hpp"
#include "HMD_TrackerSKCF.hpp"
#include "HMD_TrackerCT.hpp"
#include "HMD_TrackerSTC.hpp"
/*#include "HMD_TrackerBACF.hpp"*/

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

	// coded by jimchen
	case 1: //2015 CVPR
		pTracker = new HMD_TrackerKCF(); 
		break;
	
    case 2: //2015 CVPR
    	pTracker = new TrackerCMT();
    	break;

    case 3: //2015 CVPR
    	pTracker = new TrackerDAT();
    	break;

	case 5: //2012 ECCV
		pTracker = new TrackerCSK();
		break;

	case 6: //2015 SAMF's simple version with 3 scales 
		pTracker = new TrackerSKCF();
		break;

	case 7: //2012 ECCV 
		pTracker = new TrackerCT();
		break;

	case 8: //20xx 
		pTracker = new TrackerSTC();
		break;

// 	case 9: //2017 cvpr 
// 		pTracker = new TrackerBACF();
// 		break;

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

