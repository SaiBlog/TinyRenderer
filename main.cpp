#include <vector>
#include <cmath>
#include <cstdlib>
#include <limits>
#include<iostream>
#include "tgaimage.h"
#include "model.h"
#include "geometry.h"


const TGAColor white = TGAColor(255, 255, 255, 255);
const TGAColor red   = TGAColor(255, 0,   0,   255);
const TGAColor green  = TGAColor(0 , 255,   0,   255);

const int width = 800;
const int height = 800;
const int depth = 255;


#pragma region 全局变量
Model* model = NULL;

Vec3f light_dir1 = Vec3f(0, 0, -1).normalize();
Vec3f light_dir2 = Vec3f(0.5, 1, 1).normalize();
//Vec3f light_color(255, 170, 0);//落山的太阳
Vec3f light_color(255, 255, 255);//白光

Vec3f camera(0, 0, 10);
Vec3f eye(-5, -5, 10);
Vec3f center(0, 0, 0);

TGAImage image(width, height, TGAImage::RGB);
TGAImage zbimage(width, height, TGAImage::GRAYSCALE);

float* fzbuffer = new float[width * height];
int* izbuffer = new int[width * height];

#pragma endregion



Vec2i t0[3] = { Vec2i(10, 70),   Vec2i(50, 160),  Vec2i(70, 80) };
Vec2i t1[3] = { Vec2i(180, 50),  Vec2i(150, 1),   Vec2i(70, 180) };
Vec2i t2[3] = { Vec2i(180, 150), Vec2i(120, 160), Vec2i(130, 180) };

//Vec3i pts[3] = { Vec3i(10,10,0), Vec3i(100, 10,0), Vec3i(190, 10,0) };//退化的三角形
Vec3i pts[3] = { Vec3i(10,10,0), Vec3i(100, 30,0), Vec3i(190, 160,0) };



#pragma region Helper
Vec3f world2screen(Vec3f v) 
{
    return Vec3f(int((v.x + 1.) * width / 2. + .5), int((v.y + 1.) * height / 2. + .5), v.z);
}

Vec3f cross(Vec3f v1, Vec3f v2)
{
    return Vec3f(v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x);
}

//将点从齐次空间变回来
Vec3f m2v(Matrix m)
{
    return Vec3f(m[0][0] / m[3][0], m[1][0] / m[3][0], m[2][0] / m[3][0]);
}

//将向量变为列向量的
Matrix v2m(Vec3f v)
{
    Matrix m(4, 1);
    m[0][0] = v.x;
    m[1][0] = v.y;
    m[2][0] = v.z;
    m[3][0] = 1.f;
    return m;
}
#pragma endregion


#pragma region Transformation
//构建视口矩阵
Matrix viewport(int x, int y, int w, int h) 
{
    //定义一个单位矩阵
    Matrix m = Matrix::identity(4);

    //平移
    m[0][3] = x + w / 2.f;
    m[1][3] = y + h / 2.f;
    m[2][3] = depth / 2.f;

    //缩放
    m[0][0] = w / 2.f;
    m[1][1] = h / 2.f;
    m[2][2] = depth / 2.f;
    return m;
}

//构建视图矩阵
Matrix lookat(Vec3f eye, Vec3f center, Vec3f up) 
{
    Vec3f z = (eye - center).normalize();
    Vec3f x = (up ^ z).normalize();
    Vec3f y = (z ^ x).normalize();

    Matrix res = Matrix::identity(4);

    for (int i = 0; i < 3; i++) 
    {
        res[0][i] = x[i];
        res[1][i] = y[i];
        res[2][i] = z[i];
        res[i][3] = -center[i];
    }
    return res;
}

#pragma endregion




//----------------------------------------------------------------------------


#pragma region 绘制直线

void Bresenham_DrawLine(int x0, int y0, int x1, int y1, TGAImage& image, TGAColor color) 
{
    bool steep = false;

    //判断k值
    if (std::abs(x0 - x1) < std::abs(y0 - y1)) 
    {
        std::swap(x0, y0);
        std::swap(x1, y1);
        steep = true;
    }

    //从左往右绘制
    if (x0 > x1) 
    {
        std::swap(x0, x1);
        std::swap(y0, y1);
    }


    int dx = x1 - x0;
    int dy = y1 - y0;
    int derror2 = std::abs(dy) * 2;
    int error2 = 0;
    int y = y0;


    for (int x = x0; x <= x1; x++)
    {
        if (steep) 
        {
            image.set(y, x, color);
        }
        else 
        {
            image.set(x, y, color);
        }

        error2 += derror2;

        if (error2 > dx)
        {
            y += (y1 > y0 ? 1 : -1);
            error2 -= dx * 2;
        }
    }
}

void line(Vec2i p0, Vec2i p1, TGAImage& image, TGAColor color)
{
    Bresenham_DrawLine(p0.x, p0.y, p1.x, p1.y, image, color);
}

#pragma endregion


//----------------------------------------------------------------------------


#pragma region CPU

void fillTriangle_EdgeWalking_Basic(Vec2i t0, Vec2i t1, Vec2i t2, TGAImage& image, TGAColor color)
{
    if (t0.y == t1.y && t0.y == t2.y) return; // I dont care about degenerate triangles 
    // sort the vertices, t0, t1, t2 lower−to−upper (bubblesort yay!) 
    if (t0.y > t1.y) std::swap(t0, t1);
    if (t0.y > t2.y) std::swap(t0, t2);
    if (t1.y > t2.y) std::swap(t1, t2);

    int total_height = t2.y - t0.y;
    for (int i = 0; i < total_height; i++)
    {
        bool second_half = i > t1.y - t0.y || t1.y == t0.y; //判断有没有到达第二种三角形或者就是第二种三角形

        int segment_height = second_half ? t2.y - t1.y : t1.y - t0.y;//两种形态的总长度

        float alpha = (float)i / total_height;  //红色线的比例
        //绿线的比例，由于存在两种状态，所以需要判断
        float beta = (float)(i - (second_half ? t1.y - t0.y : 0)) / segment_height; // be careful: with above conditions no division by zero here 

        Vec2i A = t0 + (t2 - t0) * alpha;
        Vec2i B = second_half ? t1 + (t2 - t1) * beta : t0 + (t1 - t0) * beta;
        if (A.x > B.x) std::swap(A, B);
        for (int j = A.x; j <= B.x; j++) 
        {
            image.set(j, t0.y + i, color); // attention, due to int casts t0.y+i != A.y 
        }
    }
}

void drawObjWireFrame_CPU(int& argc, char**& argv)
{
    if (2 == argc) 
    {
        model = new Model(argv[1]);
    }
    else {
        model = new Model("obj/african_head.obj");
        //model = new Model("obj/1.obj");
    }


    for (int i = 0; i < model->nfaces(); i++)
    {
        std::vector<int> face = model->face(i);

        for (int j = 0; j < 3; j++)
        {
            Vec3f v0 = model->vert(face[j]);
            Vec3f v1 = model->vert(face[(j + 1) % 3]);

            int x0 = (v0.x + 1.) * width / 2.;
            int y0 = (v0.y + 1.) * height / 2.;
            int x1 = (v1.x + 1.) * width / 2.;
            int y1 = (v1.y + 1.) * height / 2.;

            Bresenham_DrawLine(x0, y0, x1, y1, image, white);
        }
    }
}

void fillObj_FlatShading_EdgeWalking_Basic(int& argc, char**& argv)
{
    if (2 == argc)
    {
        model = new Model(argv[1]);
    }
    else
    {
        model = new Model("obj/african_head.obj");
    }


    for (int i = 0; i < model->nfaces(); i++)
    {
        std::vector<int> face = model->face(i);

        Vec2i screen_coords[3];
        Vec3f world_coords[3];

        for (int j = 0; j < 3; j++)
        {
            Vec3f v = model->vert(face[j]);

            screen_coords[j] = Vec2i((v.x + 1.) * width / 2., (v.y + 1.) * height / 2.);
            world_coords[j] = v;
        }

        Vec3f n = cross((world_coords[2] - world_coords[0]), (world_coords[1] - world_coords[0])); //计算三角形的法向量
        n.normalize();

        float intensity = n * light_dir1;//喜闻乐见

        if (intensity > 0)
        {
            fillTriangle_EdgeWalking_Basic(screen_coords[0], screen_coords[1], screen_coords[2], image,
                TGAColor(intensity * light_color.x, intensity * light_color.y, intensity * light_color.z, 255));
        }
    }

}

void fillTriangle_EdgeWalking_Z_UV(Vec3i t0, Vec3i t1, Vec3i t2, Vec2i uv0, Vec2i uv1, Vec2i uv2, TGAImage& image, float intensity, float* fzbuffer)
{
    if (t0.y == t1.y && t0.y == t2.y) return; // i dont care about degenerate triangles

    if (t0.y > t1.y) { std::swap(t0, t1); std::swap(uv0, uv1); }
    if (t0.y > t2.y) { std::swap(t0, t2); std::swap(uv0, uv2); }
    if (t1.y > t2.y) { std::swap(t1, t2); std::swap(uv1, uv2); }

    int total_height = t2.y - t0.y;
    for (int i = 0; i < total_height; i++)
    {
        //Edge Walking
        bool second_half = i > t1.y - t0.y || t1.y == t0.y;
        int segment_height = second_half ? t2.y - t1.y : t1.y - t0.y;
        float alpha = (float)i / total_height;
        float beta = (float)(i - (second_half ? t1.y - t0.y : 0)) / segment_height; // be careful: with above conditions no division by zero here
        Vec3i A = t0 + Vec3f(t2 - t0) * alpha;
        Vec3i B = second_half ? t1 + Vec3f(t2 - t1) * beta : t0 + Vec3f(t1 - t0) * beta;

        //获取水平线上左右两个点的UV值
        Vec2i uvA = uv0 + (uv2 - uv0) * alpha;
        Vec2i uvB = second_half ? uv1 + (uv2 - uv1) * beta : uv0 + (uv1 - uv0) * beta;

        if (A.x > B.x) { std::swap(A, B); std::swap(uvA, uvB); }

        for (int j = A.x; j <= B.x; j++)
        {
            //计算比例
            float phi = B.x == A.x ? 1. : (float)(j - A.x) / (float)(B.x - A.x);

            //以下P为当前点
            //插值时我们要连同深度值一起计算
            Vec3i   P = Vec3f(A) + Vec3f(B - A) * phi;

            //插值出的纹理坐标
            Vec2i uvP = uvA + (uvB - uvA) * phi;

            //计算当前点的索引
            int idx = P.x + P.y * width;
            if (fzbuffer[idx] < P.z)
            {
                fzbuffer[idx] = P.z;
                //根据纹理坐标获取纹理的颜色值
                TGAColor color = model->diffuse(uvP);
                image.set(P.x, P.y, TGAColor(color.r * intensity, color.g * intensity, color.b * intensity));
            }
        }
    }
}

void fillObj_EdgeWalking_Z_UV_Projection(int argc, char** argv)
{
    if (2 == argc) {
        model = new Model(argv[1]);
    }
    else {
        model = new Model("obj/african_head.obj");
    }


    for (int i = 0; i < width * height; i++)
    {
        fzbuffer[i] = std::numeric_limits<float>::min();
    }

    { // draw the model
        Matrix Projection = Matrix::identity(4);
        Projection[3][2] = -1.f / camera.z;

        Matrix ViewPort = viewport(width / 8, height / 8, width * 3 / 4, height * 3 / 4);


        for (int i = 0; i < model->nfaces(); i++)
        {
            std::vector<int> face = model->face(i);
            Vec3i screen_coords[3];
            Vec3f world_coords[3];


            for (int j = 0; j < 3; j++)
            {
                Vec3f v = model->vert(face[j]);
                //将模型空间投影到屏幕空间
                screen_coords[j] = m2v(ViewPort * Projection * v2m(v));
                world_coords[j] = v;
            }

            Vec3f n = (world_coords[2] - world_coords[0]) ^ (world_coords[1] - world_coords[0]);
            n.normalize();

            float intensity = n * light_dir1;

            if (intensity > 0)
            {
                Vec2i uv[3];
                for (int k = 0; k < 3; k++)
                {
                    //获取每一个点的UV值
                    uv[k] = model->uv(i, k);
                }
                //重头戏，插值出每一个点的信息
                fillTriangle_EdgeWalking_Z_UV(screen_coords[0], screen_coords[1], screen_coords[2], uv[0], uv[1], uv[2], image, intensity, fzbuffer);
            }
        }

    }

    { // dump z-buffer (debugging purposes only)
        TGAImage zbimage(width, height, TGAImage::GRAYSCALE);
        for (int i = 0; i < width; i++)
        {
            for (int j = 0; j < height; j++)
            {
                zbimage.set(i, j, TGAColor(fzbuffer[i + j * width], 1));
            }
        }
        zbimage.flip_vertically(); // i want to have the origin at the left bottom corner of the image
        zbimage.write_tga_file("zbuffer.tga");
    }
}

void filTriangle_EdgeWalking_Z_UV_MVP(Vec3i t0, Vec3i t1, Vec3i t2, float ity0, float ity1, float ity2, Vec2i uv0, Vec2i uv1, Vec2i uv2, TGAImage& image, int* zbuffer)
{
    if (t0.y == t1.y && t0.y == t2.y) return; // i dont care about degenerate triangles
    if (t0.y > t1.y) { std::swap(t0, t1); std::swap(ity0, ity1); std::swap(uv0, uv1); }
    if (t0.y > t2.y) { std::swap(t0, t2); std::swap(ity0, ity2); std::swap(uv0, uv2); }
    if (t1.y > t2.y) { std::swap(t1, t2); std::swap(ity1, ity2); std::swap(uv1, uv2); }

    int total_height = t2.y - t0.y;
    for (int i = 0; i < total_height; i++)
    {
        bool second_half = i > t1.y - t0.y || t1.y == t0.y;
        int segment_height = second_half ? t2.y - t1.y : t1.y - t0.y;

        float alpha = (float)i / total_height;
        float beta = (float)(i - (second_half ? t1.y - t0.y : 0)) / segment_height; // be careful: with above conditions no division by zero here

        Vec3i A = t0 + Vec3f(t2 - t0) * alpha;
        Vec3i B = second_half ? t1 + Vec3f(t2 - t1) * beta : t0 + Vec3f(t1 - t0) * beta;

        //获取水平线上左右两个点的UV值
        Vec2i uvA = uv0 + (uv2 - uv0) * alpha;
        Vec2i uvB = second_half ? uv1 + (uv2 - uv1) * beta : uv0 + (uv1 - uv0) * beta;

        //光强求比例
        float ityA = ity0 + (ity2 - ity0) * alpha;
        float ityB = second_half ? ity1 + (ity2 - ity1) * beta : ity0 + (ity1 - ity0) * beta;

        if (A.x > B.x) { std::swap(A, B); std::swap(ityA, ityB); std::swap(uvA, uvB); }

        for (int j = A.x; j <= B.x; j++)
        {
            //水平比例
            float phi = B.x == A.x ? 1. : (float)(j - A.x) / (B.x - A.x);

            Vec3i    P = Vec3f(A) + Vec3f(B - A) * phi;

            //插值出的纹理坐标
            Vec2i uvP = uvA + (uvB - uvA) * phi;

            //光强的插值
            float ityP = ityA + (ityB - ityA) * phi;
            //防止溢出
            ityP = ityP > 1.f ? 1.f : (ityP < 0.f ? 0.f : ityP);

            int idx = P.x + P.y * width;

            if (P.x >= width || P.y >= height || P.x < 0 || P.y < 0) continue;
            if (zbuffer[idx] < P.z)
            {
                zbuffer[idx] = P.z;
                TGAColor color = model->diffuse(uvP);
                image.set(P.x, P.y, TGAColor(light_color.x * ityP * color.r / 255, light_color.y * ityP * color.g / 255, light_color.z * ityP * color.b / 255));
                //image.set(P.x, P.y, TGAColor(ityP * color.r, ityP * color.g, ityP * color.b));
            }
        }
    }
}

void fillObj_EdgeWalking_Z_UV_MVP(int argc, char** argv)
{
    if (2 == argc)
    {
        model = new Model(argv[1]);
    }
    else
    {
        model = new Model("obj/african_head.obj");
    }

    for (int i = 0; i < width * height; i++)
    {
        izbuffer[i] = std::numeric_limits<int>::min();
    }

    { // draw the model
        Matrix ModelView = lookat(eye, center, Vec3f(0, 1, 0));

        Matrix Projection = Matrix::identity(4);
        Projection[3][2] = -1.f / (eye - center).norm();

        Matrix ViewPort = viewport(width / 8, height / 8, width * 3 / 4, height * 3 / 4);

        std::cerr << ModelView << std::endl;
        std::cerr << Projection << std::endl;
        std::cerr << ViewPort << std::endl;
        Matrix z = (ViewPort * Projection * ModelView);
        std::cerr << z << std::endl;

        for (int i = 0; i < model->nfaces(); i++)
        {
            std::vector<int> face = model->face(i);
            Vec3i screen_coords[3];
            Vec3f world_coords[3];
            float intensity[3];

            for (int j = 0; j < 3; j++)
            {
                Vec3f v = model->vert(face[j]);
                screen_coords[j] = Vec3f(ViewPort * Projection * ModelView * Matrix(v));
                world_coords[j] = v;

                intensity[j] = model->norm(i, j) * light_dir2;
            }

            Vec2i uv[3];
            for (int k = 0; k < 3; k++)
            {
                //获取每一个点的UV值
                uv[k] = model->uv(i, k);
            }

            filTriangle_EdgeWalking_Z_UV_MVP(screen_coords[0], screen_coords[1], screen_coords[2], intensity[0], intensity[1], intensity[2], uv[0], uv[1], uv[2], image, izbuffer);
        }
    }
}

#pragma endregion


//----------------------------------------------------------------------------


#pragma region  GPU

Vec3f barycentric(Vec3f A, Vec3f B, Vec3f C, Vec3f P)
{
    Vec3f s[2];
    for (int i = 2; i--;)
    {
        s[i][0] = C[i] - A[i];
        s[i][1] = B[i] - A[i];
        s[i][2] = A[i] - P[i];
    }
    Vec3f u = cross(s[0], s[1]);

    //如果三角形的三点共线，那么u.z为0
    if (std::abs(u[2]) < 1e-2) return Vec3f(-1, 1, 1);

    return Vec3f(1.f - (u.x + u.y) / u.z, u.y / u.z, u.x / u.z);//得到最终的比例
}

Vec3f barycentric(Vec3i A, Vec3i B, Vec3i C, Vec3i P)
{
    Vec3f s[2];
    for (int i = 2; i--;)
    {
        s[i][0] = C[i] - A[i];
        s[i][1] = B[i] - A[i];
        s[i][2] = A[i] - P[i];
    }
    Vec3f u = cross(s[0], s[1]);

    //如果三角形的三点共线，那么u.z为0
    if (std::abs(u[2]) < 1e-2) return Vec3f(-1, 1, 1);

    return Vec3f(1.f - (u.x + u.y) / u.z, u.y / u.z, u.x / u.z);//得到最终的比例
}

void fillTriangle_EdgeEquation_Basic(Vec3i* pts, TGAImage& image, TGAColor color) 
{
    Vec2i bboxmin(image.get_width() - 1, image.get_height() - 1);
    Vec2i bboxmax(0, 0);
    Vec2i clamp(image.get_width() - 1, image.get_height() - 1);

    //计算三个点的包围盒
    for (int i = 0; i < 3; i++) 
    {
        bboxmin.x = std::max(0, std::min(bboxmin.x, pts[i].x));
        bboxmin.y = std::max(0, std::min(bboxmin.y, pts[i].y));

        bboxmax.x = std::min(clamp.x, std::max(bboxmax.x, pts[i].x));
        bboxmax.y = std::min(clamp.y, std::max(bboxmax.y, pts[i].y));
    }

    //根据重心坐标判断如果包围盒中的点在三角形中，则绘制该点
    Vec3i P;
    for (P.x = bboxmin.x; P.x <= bboxmax.x; P.x++) 
    {
        for (P.y = bboxmin.y; P.y <= bboxmax.y; P.y++) 
        {
            Vec3f bc_screen = barycentric(pts[0], pts[1], pts[2], P);
            if (bc_screen.x < 0 || bc_screen.y < 0 || bc_screen.z < 0) continue;//如果权值为负数
            image.set(P.x, P.y, color);
        }
    }
}

void fillTriangle_EdgeEquation_Z(Vec3f* pts, float* fzbuffer, TGAImage& image, TGAColor color)
{
    Vec2f bboxmin(std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
    Vec2f bboxmax(-std::numeric_limits<float>::max(), -std::numeric_limits<float>::max());
    Vec2f clamp(image.get_width() - 1, image.get_height() - 1);

    for (int i = 0; i < 3; i++)
    {
        bboxmin.x = std::max(0.f, std::min(bboxmin.x, pts[i][0]));
        bboxmax.x = std::min(clamp[0], std::max(bboxmax.x, pts[i][0]));

        bboxmin.y = std::max(0.f, std::min(bboxmin.y, pts[i][1]));
        bboxmax.y = std::min(clamp[1], std::max(bboxmax.y, pts[i][1]));
    }


    Vec3f P;


    for (P.x = bboxmin.x; P.x <= bboxmax.x; P.x++)
    {
        for (P.y = bboxmin.y; P.y <= bboxmax.y; P.y++)
        {

            Vec3f bc_screen = barycentric(pts[0], pts[1], pts[2], P);
            if (bc_screen.x < 0 || bc_screen.y < 0 || bc_screen.z < 0) continue;

            P.z = 0;

            for (int i = 0; i < 3; i++) P.z += pts[i][2] * bc_screen[i];

            if (fzbuffer[int(P.x + P.y * width)] < P.z)
            {
                fzbuffer[int(P.x + P.y * width)] = P.z;
                image.set(P.x, P.y, color);
            }
        }
    }
}

void fillObj_FlatShading_Equation_Z(int& argc, char**& argv)
{
    if (2 == argc)
    {
        model = new Model(argv[1]);
    }
    else
    {
        model = new Model("obj/african_head.obj");
    }

    for (int i = width * height; i--; fzbuffer[i] = -std::numeric_limits<float>::max());//初始化为负无穷

    for (int i = 0; i < model->nfaces(); i++)
    {

        Vec3f screen_coords[3];
        Vec3f world_coords[3];
        std::vector<int> face = model->face(i);

        for (int j = 0; j < 3; j++)
        {
            Vec3f v = model->vert(face[j]);
            world_coords[j] = v;

            screen_coords[j] = world2screen(v);//z轴不变，将x、y转换为屏幕坐标
        }

        Vec3f n = cross((world_coords[2] - world_coords[0]), (world_coords[1] - world_coords[0])); //计算三角形的法向量
        n.normalize();

        float intensity = n * light_dir1;//喜闻乐见

        //for (int i = 0; i < 3; i++)
        //{
        //    for (int j = 0; j < 3;j++)std::cout << screen_coords[i][j] << ' ';
        //    std::cout << std::endl;
        //}
        //std::cout << std::endl;

        if (intensity > 0)
        {
            fillTriangle_EdgeEquation_Z(screen_coords, fzbuffer, image,
                TGAColor(intensity * light_color.x, intensity * light_color.y, intensity * light_color.z, 255));
        }
    }
}

#pragma endregion



//----------------------------------------------------------------------------

//测试drawline的漏洞
void lineTest(TGAImage& image)
{
    Bresenham_DrawLine(10, 1, 20, 50, image, red);
    Bresenham_DrawLine(10, 101, 61, 110, image, red);
    Bresenham_DrawLine(10, 201, 61, 191, image, red);
    Bresenham_DrawLine(10, 301, 20, 251, image, red);

    Bresenham_DrawLine(790, 1, 780, 50, image, red);
    Bresenham_DrawLine(790, 101, 739, 110, image, red);
    Bresenham_DrawLine(790, 201, 739, 191, image, red);
    Bresenham_DrawLine(790, 301, 780, 251, image, red);
}

// 默认使用CPU方式绘制颜色
void fillObj_randomColor(int& argc, char**& argv,int mode = 1)
{
    if (2 == argc)
    {
        model = new Model(argv[1]);
    }
    else
    {
        model = new Model("obj/african_head.obj");
    }

    for (int i = 0; i < model->nfaces(); i++)
    {
        std::vector<int> face = model->face(i);

        Vec2i screen_coord[3];
        Vec3i world_icoords[3];
        for (int j = 0; j < 3; j++)
        {
            Vec3f world_coords = model->vert(face[j]);

            screen_coord[j] = Vec2i((world_coords.x + 1.) * width / 2., (world_coords.y + 1.) * height / 2.);
            world_icoords[j] = Vec3i((world_coords.x + 1.) * width / 2., (world_coords.y + 1.) * height / 2., world_coords.z);
        }

        if (mode == 1)fillTriangle_EdgeWalking_Basic(screen_coord[0], screen_coord[1], screen_coord[2], image, TGAColor(rand() % 255, rand() % 255, rand() % 255, 255));
        else fillTriangle_EdgeEquation_Basic(world_icoords, image, TGAColor(rand() % 255, rand() % 255, rand() % 255, 255));
    }



}


int main(int argc, char** argv) 
{
    //画线的样例
    //lineTest(image);
    //使用EdgeWalking绘制三角形
    //fillTriangle_EdgeWalking_Basic(t0[0], t0[1], t0[2], image, red);
    //fillTriangle_EdgeWalking_Basic(t1[0], t1[1], t1[2], image, white);
    //fillTriangle_EdgeWalking_Basic(t2[0], t2[1], t2[2], image, green);
    //EdgeEquation绘制三角形
    //fillTriangle_EdgeEquation_Basic(pts, image, white);
    //绘制wireframe
    //drawObjWireFrame_CPU(argc, argv);
    
    //随机颜色的模型
    //fillObj_randomColor(argc, argv);
    
    //基础光照模型
    //fillObj_FlatShading_EdgeWalking_Basic(argc, argv);
    //加上ZBuffer
    //fillObj_FlatShading_Equation_Z(argc, argv);
    //透视投影+UV
    //fillObj_EdgeWalking_Z_UV_Projection(argc, argv);
    //相机自由位置+UV
    //fillObj_EdgeWalking_Z_UV_MVP(argc, argv);


    image.flip_vertically(); // i want to have the origin at the left bottom corner of the image
    image.write_tga_file("output.tga");


    delete model;
    delete[] fzbuffer;
    return 0;
}



