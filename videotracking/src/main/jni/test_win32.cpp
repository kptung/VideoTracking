#include "common.hpp"
#include "HMD_AbstractTracker.hpp"

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
	int datach = 0;//0:data-sequence, 1: china-steer sequence
	std::string seq;
	int init_frame_index = -1;
	int w = -1, h = -1;
	int stick = -1;
	std::vector<cv::Rect> objs;

	if (datach == 0)
	{
		seq = std::string("D:/workprojs/III.Projs/DataSet/Real_Data/Lab_seq0001/");
		init_frame_index = 93; //data sequence
		w = 60;
		h = 50;//data sequence
		objs = make_vector<cv::Rect>() << cv::Rect(130, 120, w, h);//data-sequence
		stick = 1;
	}
	else
	{
		seq = std::string("D:/workprojs/III.Projs/DataSet/Real_Data/Real_seq0001/");
		init_frame_index = 103; //china-steel sequence
		w = 60;
		h = 82;//china-steel sequence
		objs = make_vector<cv::Rect>() << cv::Rect(292, 367, w, h);//china-steel sequence
		stick = 5;
	}

    //std::string seq("D:/workprojs/III.Projs/DataSet/Real_Data/Lab_seq0001/");//data-sequence
	//std::string seq("D:/workprojs/III.Projs/DataSet/Real_Data/Real_seq0001/");//china-steel sequence
	std::string tmp("D:/workprojs/III.Projs/out/trackout/");
    std::vector<std::string> files;
    get_files_in_directory(seq, files);
    if ( files.size() <= 0 )
        return 0;

	//int init_frame_index = 103; //china-steel sequence
	//int 
	//int 
	//int w = 60, h = 50; //data sequence
	//std::vector<cv::Rect> objs = make_vector<cv::Rect>() << cv::Rect(292, 367, w, h);//china-steel sequence
	//std::vector<cv::Rect> objs = make_vector<cv::Rect>() << cv::Rect(130, 120, w, h);//data-sequence
	std::vector<int> tr_obj_ids;

	bool track_multiobj_flag = 0;

	if (track_multiobj_flag)
	{
		cv::Rect r1(304, 461, w, h);//china-steel sequence
		cv::Rect r2(254, 539, w, h);//china-steel sequence
		//cv::Rect r3(78, 117, w, h);
		//cv::Rect r4(30, 184, w, h);
		//cv::Rect r5(68, 187, w, h);
		std::vector<cv::Rect> others = make_vector<cv::Rect>() << r1 << r2;
		objs.insert(objs.end(), others.begin(), others.end());
	}
	int NumOfObjs = (int) objs.size();

	/************************************************************************/
	/*     Tracking method                                                  */
	/* 0: TM                                                                */
	/* 1: KCF by jimchen transfered on OpenCV  // work on long-term ok                            */
	/* 2: CMT // it's ok on short-term and long-term but slow                                                             */
	/* 3: DAT // it's ok but not good on short-term and long-term                                                             */
	/* 5: CSK // it's ok on short-term but not good on long-term                                                               */
	/* 6: SKCF // it's ok on short-term but not excellent on long-term and similiar objects (SAMF's simple version with 3 scales)                                                          */
	/* 7: CT //too slow                                                     */
	/* 8: STC //not good on short-term and long-term                                                    */

	/***********************************************************************/
	T_HANDLE tracker = CreateVideoTracker(6);
	/************************************************************************/

	cv::Mat frame = imread( seq + files[init_frame_index-1]);

	for(int kk=0;kk<NumOfObjs;kk++)
	{ 
		if (kk == 0)
		{
			int obj_id = SetTrackingTarget(tracker, frame, objs.at(kk));
			tr_obj_ids.push_back(obj_id);
			rectangle(frame, objs.at(kk), Scalar(255 * (1 - (obj_id & 1)), 255 * (obj_id & 2), 255 * (1 - (obj_id & 4))), stick, 8);
			if (!track_multiobj_flag)
				break;
		}
		else
		{
			int obj_id = AddTrackingTarget(tracker, frame, objs.at(kk));
			tr_obj_ids.push_back(obj_id);
			rectangle(frame, objs.at(kk), Scalar(255 * (1 - (obj_id & 1)), 255 * (obj_id & 2), 255 * (1 - (obj_id & 4))), stick, 8);
		}
	}

	cv::imwrite( tmp + files[init_frame_index-1], frame );

	printf("Tracking...\n");
	std::map<int, cv::Rect> results;
	for (int next = 0; next < files.size()-1; ++next)
	//for (int next = 0; next < 700; ++next)
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
			rectangle(frame, itr->second, Scalar(255*(1-obj_id&1),255*(obj_id&2),255*(1-(obj_id&4))), stick, 8);
		}
		imwrite(tmp+files[index], frame);
		//
		ofstream out1;
		char *ptimefile_name = "D:/workprojs/III.Projs/out/ptime.csv";
		out1.open(ptimefile_name, ios::app);
		out1 << index+1 << "," << t *1000 << "," << endl;
	}
	printf("Tracking Done!");
	DeleteVideoTracker(tracker);
	return 0;
}
