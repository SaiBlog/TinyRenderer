#ifndef __MODEL_H__
#define __MODEL_H__

#include <vector>
#include"tgaimage.h"
#include "geometry.h"

class Model 
{
private:
	//�洢�����ģ�Ϳռ�����
	std::vector<Vec3f> verts_;
	//�洢�����������
	std::vector<std::vector<Vec3i> > faces_;



	std::vector<Vec3f>norms_;

	//�洢��ǰ���������ֵ
	std::vector<Vec2f>uv_;
	//�洢����ͼ����Ϣ
	TGAImage diffusemap_;
	//��ȡ����ͼ����Ϣ
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

	//�����������Ͷ���������ȡUVֵ
	Vec2i uv(int iface, int nvert);
	//����UV�����ȡ����ͼ��Ϣ
	TGAColor diffuse(Vec2i uv);
	//�����������Ͷ���������ȡ����
	Vec3f norm(int iface, int nvert);
};

#endif //__MODEL_H__
