#include "common_structs.h"
#include "spline.h"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <array>
#include <unordered_map>
#include <cereal/archives/binary.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/array.hpp>
#include <cereal/types/unordered_map.hpp>
#include <cereal/types/map.hpp>


void Serialize(std::unordered_map<char, std::vector<Spline> > fun, char* filename)
{
    std::ofstream os(filename, std::ios::binary);
    cereal::BinaryOutputArchive archive(os);
    archive(fun); 
    std::cout << "Saved model to " << filename << std::endl;
}


void Serialize_array(std::array<Spline, 3> arr, char* filename)
{
    std::ofstream os(filename, std::ios::binary);
    cereal::BinaryOutputArchive archive(os);
    archive(arr); 
    std::cout << "Saved model to " << filename << std::endl;
}


void Serialize_vector(std::vector<int> v, char* filename)
{
    std::ofstream os(filename, std::ios::binary);
    cereal::BinaryOutputArchive archive(os);
    archive(v);
    std::cout << "Saved model to " << filename << std::endl;
}


auto Deserialize(char* filename)
{
    Spline model;
    std::ifstream is(filename, std::ios::binary);
    cereal::BinaryInputArchive archive(is);
    archive(model);
    return model;
}


int main() 
{
    Spline x = (Spline){1, 1, 1};
    
    std::vector<Spline> v;

    v.push_back(x);
    v.push_back(x);
    v.push_back(x);

    std::vector<Spline> v2;
    v2.push_back(x);
    v2.push_back(x);
    v2.push_back(x);

    std::vector<std::vector<Spline> > fun(10);

    std::unordered_map<char, std::vector<Spline> > fun2;

    fun2['a'] = v;
    fun2['b'] = v2;
    // fun2['b']
    // fun[1] = v;
    // fun.push_back(v);
    // fun.push_back(v2);
    // fun.push_back(v2);


    std::array<Spline, 3> arr = {};

    std::vector<int> vv(10);

    char model_path[] = "../saved_models/test_vector.bin";
    Serialize_vector(vv, model_path);

    // char model_path[] = "test_spline_vector.bin";
    // Serialize(v, model_path);

    // char model_path[] = "../saved_models/test_spline_array.bin";
    // Serialize(fun2, model_path);
    // Serialize_array(arr, model_path);
    return 0;
}
