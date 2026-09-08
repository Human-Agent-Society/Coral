#ifndef GLASSDEFECT_H
#define GLASSDEFECT_H

#include <ostream>

/* This class is created
 * to model a defect instance with its all attributes 
 **/
class GlassDefect {
public:
    GlassDefect();
    GlassDefect(unsigned int id, unsigned int pid, unsigned int x, unsigned int y, unsigned int w, unsigned int h);
    virtual ~GlassDefect();

    unsigned int Getdefect_id() {
        return defect_id;
    }

    unsigned int Getplate_id() {
        return plate_id;
    }

    unsigned int Getpos_x() {
        return pos_x;
    }

    unsigned int Getpos_y() {
        return pos_y;
    }

    unsigned int Getwidth() {
        return width;
    }

    unsigned int Getheight() {
        return height;
    }

    void Setdefect_id(unsigned int val) {
        defect_id = val;
    }

    void Setplate_id(unsigned int val) {
        plate_id = val;
    }

    void Setpos_x(unsigned int val) {
        pos_x = val;
    }

    void Setpos_y(unsigned int val) {
        pos_y = val;
    }

    void Setwidth(unsigned int val) {
        width = val;
    }

    void Setheight(unsigned int val) {
        height = val;
    }

    friend std::ostream & operator<<(std::ostream& os, GlassDefect defect) {
        os << "Defect (id,p,x,y,w,h) = (" << defect.defect_id << "," << defect.plate_id << "," << defect.pos_x << "," << defect.pos_y << "," << defect.width << "," << defect.height << ")";
        return os;
    }
    
private:
    unsigned int defect_id; // Defect's Id.
    unsigned int plate_id; // Defect's plate Id
    unsigned int pos_x; // x position.
    unsigned int pos_y; // y position.
    unsigned int width; // Defect's width.
    unsigned int height; // Defect's height.
};

#endif // GLASSDEFECT_H
