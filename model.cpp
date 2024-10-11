#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include "model.h"


Model::Model(const char* filename) : verts_(), faces_(), norms_(), uv_(), diffusemap_()
{
    std::ifstream in;
    //找到指定路径
    in.open(filename, std::ifstream::in);

    if (in.fail()) return;

    std::string line;

    while (!in.eof()) //文件是否到达结尾
    {
        //将当前行写入line
        std::getline(in, line);

        //将line转换为const char*类型，并传给iss
        std::istringstream iss(line.c_str());
        char trash;

        //如果当前行的前两个字符为v (后面有一个空格)
        if (!line.compare(0, 2, "v ")) 
        {
            iss >> trash;   //现在我们已经不再需要v、f这样的标识符了
            Vec3f v;
            for (int i = 0; i < 3; i++) iss >> v[i];
            verts_.push_back(v);
        }
        else if (!line.compare(0, 2, "f ")) //如果是面信息
        {
            std::vector<Vec3i> f;
            Vec3i tmp;
            iss >> trash;
            while (iss >> tmp[0] >> trash >> tmp[1] >> trash >> tmp[2]) 
            {
                for (int i = 0; i < 3; i++) tmp[i]--; 
                f.push_back(tmp);
            }
            faces_.push_back(f);
        }
        else if (!line.compare(0,3,"vn "))
        {
            iss >> trash >> trash;
            Vec3f n;
            for (int i = 0; i < 3; i++)iss >> n[i];
            norms_.push_back(n);
        }
        else if (!line.compare(0, 3, "vt "))
        {
            iss >> trash >> trash;
            Vec2f uv;
            for (int i = 0; i < 2; i++)iss >> uv[i];
            uv_.push_back(uv);
        }
    }
    std::cerr << "# v# " << verts_.size() << " f# " << faces_.size() << std::endl;

    load_texture(filename, "_diffuse.tga", diffusemap_);


}

Model::~Model() 
{
}

int Model::nverts() 
{
    return (int)verts_.size();
}

int Model::nfaces() 
{
    return (int)faces_.size();
}

std::vector<int> Model::face(int idx) 
{
    std::vector<int>face;
    for (int i = 0; i < (int)faces_[idx].size(); i++)face.push_back(faces_[idx][i][0]);
    return face;
}

Vec3f Model::vert(int i) 
{
    return verts_[i];
}


void Model::load_texture(std::string filename, const char* suffix, TGAImage& img)
{
    std::string texfile(filename);
    size_t dot = texfile.find_last_of(".");
    if (dot != std::string::npos) 
    {
        texfile = texfile.substr(0, dot) + std::string(suffix);
        std::cerr << "texture file " << texfile << " loading " << (img.read_tga_file(texfile.c_str()) ? "ok" : "failed") << std::endl;
        img.flip_vertically();
    }
}


TGAColor Model::diffuse(Vec2i uv) 
{
    return diffusemap_.get(uv.x, uv.y);
}

Vec2i Model::uv(int iface, int nvert) 
{
    int idx = faces_[iface][nvert][1];//读取vt 
    return Vec2i(uv_[idx].x * diffusemap_.get_width(), uv_[idx].y * diffusemap_.get_height());
}

Vec3f Model::norm(int iface, int nvert) 
{
    int idx = faces_[iface][nvert][2];
    return norms_[idx].normalize();
}