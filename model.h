#ifndef __MODEL_H__
#define __MODEL_H__

#include <vector>
#include "geometry.h"

class Model 
{
private:
	std::vector<Vec3f> verts_;
	std::vector<std::vector<int> > faces_;
public:

	//构造函数从文件路径读取obj文件
	Model(const char* filename);
	~Model();

	//返回顶点数量
	int nverts();
	//返回面数
	int nfaces();

	//返回索引为i的顶点数据
	Vec3f vert(int i);

	//返回当前面的三个索引
	std::vector<int> face(int idx);
};

#endif //__MODEL_H__
