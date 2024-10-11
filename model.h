#ifndef __MODEL_H__
#define __MODEL_H__

#include <vector>
#include"tgaimage.h"
#include "geometry.h"

class Model 
{
private:
	//存储顶点的模型空间坐标
	std::vector<Vec3f> verts_;
	//存储所有面的索引
	std::vector<std::vector<Vec3i> > faces_;



	std::vector<Vec3f>norms_;

	//存储当前顶点的纹理值
	std::vector<Vec2f>uv_;
	//存储纹理图的信息
	TGAImage diffusemap_;
	//读取纹理图的信息
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

	//根据面索引和顶点索引获取UV值
	Vec2i uv(int iface, int nvert);
	//根据UV坐标读取纹理图信息
	TGAColor diffuse(Vec2i uv);
	//根据面索引和顶点索引获取法线
	Vec3f norm(int iface, int nvert);
};

#endif //__MODEL_H__
