#ifndef __MODEL_H__
#define __MODEL_H__

#include <vector>
#include"tgaimage.h"
#include "geometry.h"

class Model 
{
private:
	std::vector<Vec3f> verts_;
	std::vector<std::vector<Vec3i> > faces_;



	std::vector<Vec3f>norms_;
	std::vector<Vec2f>uv_;
	TGAImage diffusemap_;
	void load_texture(std::string filename, const char* suffix, TGAImage& img);

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

	Vec2i uv(int iface, int nvert);
	TGAColor diffuse(Vec2i uv);
};

#endif //__MODEL_H__
