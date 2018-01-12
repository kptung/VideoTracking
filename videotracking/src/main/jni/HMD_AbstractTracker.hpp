#ifndef __HMD_AbstractTracker_hpp__
#define __HMD_AbstractTracker_hpp__

#include <map>
#include <set>
#include <vector>
#include <opencv2/opencv.hpp>

typedef long long T_HANDLE;

#ifdef __cplusplus
extern "C"
{
#endif

T_HANDLE CreateVideoTracker( int type = 0 );

bool DeleteVideoTracker( T_HANDLE handle );

int SetTrackingTarget( T_HANDLE handle, const cv::Mat& image, const cv::Rect& target );

int AddTrackingTarget( T_HANDLE handle, const cv::Mat& image, const cv::Rect& target );

bool RemoveTrackingTarget( T_HANDLE handle, const int& object_id );

bool RunTargetTracking( T_HANDLE handle, const cv::Mat &image, std::map<int, cv::Rect>& results );

//
//bool GetActiveTrackers( std::vector<T_HANDLE>& active_tracker_handles );
//
//bool GetTrackingTargets( T_HANDLE handle, std::vector<int>& tracking_object_ids );
//

#ifdef __cplusplus
};
#endif



class AbstractTracker
{
public:
	/**
	 * Constructor
	 */
	AbstractTracker (int algo = 0) : m_algorithm_type(algo), m_largest_object_id(0)
	{
	}


	virtual ~AbstractTracker ()
	{
	}


	virtual int setTrackingObject (const cv::Mat& image, const cv::Rect& obj_roi)
	{
		if ( image.empty() )
			return -2;

		m_frame_resolution = image.size();
		m_active_objects.clear();
		m_largest_object_id = 0;

		return addTrackingObject(image, obj_roi);
	}


	virtual int addTrackingObject (const cv::Mat& image, const cv::Rect& obj_roi)
	{
		if ( image.empty() || m_frame_resolution != image.size() )
			return -2;

		int obj_id = -1;

		// verify if the roi is inside the image
		cv::Rect image_rect(0, 0, image.cols, image.rows);
		if ( (obj_roi & image_rect) == obj_roi )
		{
			obj_id = m_largest_object_id++;
			m_active_objects.insert( std::make_pair(obj_id, image(obj_roi) ) );
		}

		return obj_id;
	}


	virtual bool remTrackingObject (int obj_id)
	{
		auto search = m_active_objects.find( obj_id );
		if( search == m_active_objects.end() )
			return false;
		m_active_objects.erase( search );

		return true;
	}


	virtual bool runObjectTrackingAlgorithm (const cv::Mat& image, std::map<int, cv::Rect>& objects) = 0;


protected:

	int m_algorithm_type;
	cv::Size m_frame_resolution;
	int m_largest_object_id;
	std::map<int, cv::Mat> m_active_objects;

};


#endif
