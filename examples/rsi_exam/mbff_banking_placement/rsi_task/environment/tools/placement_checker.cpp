#include <iostream>
#include <fstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <sstream>
#include <boost/functional/hash.hpp>
using namespace std;

// template <> struct hash<std::pair<double, double>> {
//     inline size_t operator()(const std::pair<double, double> &v) const {
//         std::hash<double> int_hasher;
//         return int_hasher(v.first) ^ int_hasher(v.second);
//     }
// };
struct Rectangle {
    double x, y, width, height;
    Rectangle(vector<double>& in){
        x = in[0]; y = in[1];
        width = in[2]; height = in[3];
    }
};

// by Gemini
bool doOverlap(Rectangle rect1, Rectangle rect2) {
    // If one rectangle is on left side of other
    if (rect1.x + rect1.width <= rect2.x)
        return false;
    if (rect2.x + rect2.width <= rect1.x)
        return false;

    // If one rectangle is above other
    if (rect1.y + rect1.height <= rect2.y)
        return false;
    if (rect2.y + rect2.height <= rect1.y)
        return false;

    return true;
}

int main(int argc, char** argv){
    if(argc != 3){
        cout << "Usage : ./placement_checker [input file] [output file]" << endl;
        return 0;
    }
    ifstream fin(argv[1]);
    string line;
    unordered_map<string, vector<double>> FFCell, GateCell; // cell name and WH
    vector<vector<double>> gateList, FFList; // {N, 4} -> x, y, w, h
    vector<string> FFName, gateName;
    unordered_set<pair<double, double>, boost::hash<pair<double, double>>> site;

    //----------------------------
    // parse input file
    //----------------------------
    while(getline(fin, line)){
        stringstream ss(line);
        ss >> line;
        string name;
        double w, h;
        double x, y;
        if(line == "FlipFlop"){ // FF cell
            ss >> line >> name >> w >> h;
            FFCell[name] = {w, h};
        }
        else if(line == "Gate"){ // Gate cell
            ss >> name >> w >> h;
            GateCell[name] = {w, h};
        }
        else if(line == "Inst"){ // parse inst
            ss >> line >> name >> x >> y;
            if(GateCell.count(name)){ // std cell
                gateList.push_back({x, y, GateCell[name][0], GateCell[name][1]});
                gateName.push_back(line);
            }
            else{ // ff

            }
        }
        else if(line == "PlacementRows"){
            double startX, startY, width;
            size_t total;
            ss >> startX >> startY >> width >> line >> total;
            for(size_t i=0;i<total;i++)
                site.insert({startX + i*width, startY});
        }
    }
    fin.close();
    //----------------------------
    // parse output file
    //----------------------------
    fin.open(argv[2]);
    while(getline(fin, line)){
        stringstream ss(line);
        ss >> line;
        string name;
        double x, y;
        if(line == "Inst"){ // parse inst
            ss >> line >> name >> x >> y;
            if(FFCell.count(name)){ // FF
                FFList.push_back({x, y, FFCell[name][0], FFCell[name][1]});
                FFName.push_back(line);
            }
            else{
                cout << "something wrong!!!" << endl; 
            }
        }
    }

    //----------------------------
    // on site check
    //----------------------------
    bool onSite = true;
    cout << "[PLACEMENT CHECKER] start on site check ... " << endl;
    for(size_t i=0;i<FFList.size();i++){
        if(site.count({FFList[i][0], FFList[i][1]})){
            // on site
        }
        else{
            cout << "FF : " << FFName[i] << " is not on site!!!" << endl;
            onSite = false;
        }
    }
    if(onSite)
        cout << "[PLACEMENT CHECKER] All FF are on site!!!" << endl; 

    //----------------------------
    // overlap check
    //---------------------------- 
    bool nonOverlap = true;
    cout << "[PLACEMENT CHECKER] start overlap check ... " << endl;
    for(size_t i=0;i<FFList.size();i++){
        // check with std cell
        for(size_t j=0;j<gateList.size();j++){
            if(doOverlap(FFList[i], gateList[j])){
                cout << FFName[i] << " is overlap with " << gateName[j] << endl;
                nonOverlap = false;
            }
        }
        // check with FF
        for(size_t j=0;j<FFList.size();j++){
            if(i!=j && doOverlap(FFList[i], FFList[j])){
                cout << FFName[i] << " is overlap with " << FFName[j] << endl;
                nonOverlap = false;
            }
        }
    }
    if(nonOverlap)
        cout << "[PLACEMENT CHECKER] All instance are not overlapped!!!" << endl;
    if(onSite && nonOverlap)
        cout << "[PLACEMENT CHECKER] Pass placement checker!!!" << endl;
    else
        cout << "[PLACEMENT CHECKER] Fail !!! pls check your result" << endl;
    return 0;
}