#ifndef __common_hpp__
#define __common_hpp__

#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
#include <iomanip> 

#ifndef SEP
#define SEP (std::string("/"))
#endif
#ifndef TMP
#define TMP (std::string("tmp"))
#endif
#ifndef JPG
#define JPG (std::string(".jpg"))
#endif
#ifndef PNG
#define PNG (std::string(".png"))
#endif
#ifndef _RAD2DEG_
#define _RAD2DEG_ (57.2957795786)
#endif

template <typename T>
std::string ToString(const T& value)
{
    std::ostringstream stream;
    stream << value;
    return stream.str();
}


template <typename T>
class make_vector
{
public:
	typedef make_vector<T> my_type;
	my_type& operator<< (const T& val)
	{
		data_.push_back(val);
		return *this;
	}
	operator std::vector<T>() const
	{
		return data_;
	}

private:
	std::vector<T> data_;
};


#ifdef _WIN32
#include "win32\dirent.h"
#else
#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>
#endif


inline bool read_line_by_line(const std::string &filename, std::vector<std::string> &lines)
{
	lines.clear();

	std::fstream fin;

	fin.open(filename.c_str(), std::ios::in);

	if( !fin )
	{
		std::cerr << "Can't open file: " << filename << std::endl;
		return false;
	}

	std::string str;

	while( getline(fin,str) )
	{
		lines.push_back( std::string(str) );
	}

	fin.close();

	return true;
}


/* Returns a list of files in a directory (except the ones that begin with a dot) */
inline void get_files_in_directory(const std::string &directory, std::vector<std::string> &out)
{
    DIR *dir;
    struct dirent *ent;
    struct stat st;

    dir = opendir( directory.c_str() );

	if ( dir == NULL )
	{
		std::cerr << "Can't find folder: " << directory << std::endl;
		return;
	}

    while ( (ent = readdir(dir)) != NULL )
	{
        const std::string file_name = ent->d_name;
        const std::string full_file_name = directory + "/" + file_name;

        if (file_name[0] == '.')
            continue;

        if (stat(full_file_name.c_str(), &st) == -1)
            continue;

        const bool is_directory = (st.st_mode & S_IFDIR) != 0;

        if (is_directory)
            continue;

		out.push_back(file_name);
    }

    closedir(dir);
}

#endif
