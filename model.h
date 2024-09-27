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

	//���캯�����ļ�·����ȡobj�ļ�
	Model(const char* filename);
	~Model();

	//���ض�������
	int nverts();
	//��������
	int nfaces();

	//��������Ϊi�Ķ�������
	Vec3f vert(int i);

	//���ص�ǰ�����������
	std::vector<int> face(int idx);

	Vec2i uv(int iface, int nvert);
	TGAColor diffuse(Vec2i uv);
};

#endif //__MODEL_H__
