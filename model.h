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
};

#endif //__MODEL_H__
