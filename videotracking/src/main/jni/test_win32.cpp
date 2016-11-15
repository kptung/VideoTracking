#include "common.hpp"
#include "HMD_AbstractTracker.hpp"

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
    std::string seq("./Data/seq0001/");
	std::string tmp("D:/workprojs/out/");
    std::vector<std::string> files;
    get_files_in_directory(seq, files);
    if ( files.size() <= 0 )
        return 0;

	int init_frame_index = 224;
	int w = 25, h = 25;
	std::vector<cv::Rect> objs = make_vector<cv::Rect>() << cv::Rect(120, 150, 52, 45);
	std::vector<int> tr_obj_ids;

	bool track_multiobj_flag = true;

	if (track_multiobj_flag)
	{
		cv::Rect r1(22, 117, w, h);
		cv::Rect r2(48, 117, w, h);
		cv::Rect r3(78, 117, w, h);
		cv::Rect r4(30, 184, w, h);
		cv::Rect r5(68, 187, w, h);
		std::vector<cv::Rect> others = make_vector<cv::Rect>() << r1 << r2 << r3 << r4 << r5;
		objs.insert(objs.end(), others.begin(), others.end());
	}
	int NumOfObjs = (int) objs.size();


	T_HANDLE tracker = CreateVideoTracker();

	cv::Mat frame = imread( seq + files[init_frame_index-1]);

	for(int kk=0;kk<NumOfObjs;kk++)
	{ 
		if (kk == 0)
		{
			int obj_id = SetTrackingTarget(tracker, frame, objs.at(kk));
			tr_obj_ids.push_back(obj_id);
			rectangle(frame, objs.at(kk), Scalar(255 * (1 - (obj_id & 1)), 255 * (obj_id & 2), 255 * (1 - (obj_id & 4))), 1, 8);
			if (!track_multiobj_flag)
				break;
		}
		else
		{
			int obj_id = AddTrackingTarget(tracker, frame, objs.at(kk));
			tr_obj_ids.push_back(obj_id);
			rectangle(frame, objs.at(kk), Scalar(255 * (1 - (obj_id & 1)), 255 * (obj_id & 2), 255 * (1 - (obj_id & 4))), 1, 8);
		}
	}

	cv::imwrite( tmp + files[init_frame_index-1], frame );

	printf("Tracking...\n");
	std::map<int, cv::Rect> results;
	for (int next = 0; next < 700; ++next)
	{
		int index = init_frame_index + next  ;
		printf("Frame: %d\n", index);
		frame = imread( seq + files[index] );
		
		double t = (double)getTickCount();
		bool success = RunTargetTracking( tracker, frame, results );
		t = ((double)getTickCount() - t) / getTickFrequency();

		auto itr= results.begin();
		for ( ; itr != results.end(); ++itr)
		{
			int obj_id = itr->first;
			rectangle(frame, itr->second, Scalar(255*(1-obj_id&1),255*(obj_id&2),255*(1-(obj_id&4))), 1, 8);
		}
		imwrite(tmp+files[index], frame);
		//
		ofstream out1;
		char *ptimefile_name = "D:/workprojs/out/ptime.csv";
		out1.open(ptimefile_name, ios::app);
		out1 << index+1 << "," << t *1000 << "," << endl;
	}
	printf("Tracking Done!");
	DeleteVideoTracker(tracker);
	return 0;
}
