#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include "model.h"

Model::Model(const char* filename) : verts_(), faces_() 
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
            std::vector<int> f;
            int itrash, idx;
            iss >> trash;
            while (iss >> idx >> trash >> itrash >> trash >> itrash) 
            {
                idx--; // in wavefront obj all indices start at 1, not zero
                f.push_back(idx);//现在我们只管面的索引，不理会法线、切线
            }
            faces_.push_back(f);
        }
    }
    std::cerr << "# v# " << verts_.size() << " f# " << faces_.size() << std::endl;
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
    return faces_[idx];
}

Vec3f Model::vert(int i) 
{
    return verts_[i];
}

