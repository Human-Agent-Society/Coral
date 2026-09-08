#ifndef GLASSITEM_H
#define GLASSITEM_H

#include <ostream>

/* This class is created
 * to model a batch item instance with its all attributes 
 **/

class GlassItem {
public:
    GlassItem();

    virtual ~GlassItem();

    int id() {
        return _id;
    }

    unsigned int w() {
        return _w;
    }

    unsigned int h() {
        return _h;
    }

    unsigned int stackId() {
        return _stackId;
    }

    unsigned int position() {
        return _position;
    }

    void id(int val) {
        _id = val;
    }

    void w(const unsigned int& val) {
        _w = val;
    }

    void h(const unsigned int& val) {
        _h = val;
    }

    void stackId(unsigned int val) {
        _stackId = val;
    }

    void position(unsigned int val) {
        _position = val;
    }

    friend std::ostream & operator<<(std::ostream& os, GlassItem item) {
        os << "Item (id,w,h,s,p) = (" << item._id << "," << item._w << "," << item._h << "," << item._stackId << "," << item._position << ")";
        return os;
    }

private:
    int _id; // Item's Id.
    unsigned int _w; // Item's width.
    unsigned int _h; // Item's height.
    unsigned int _stackId; // Item's stack Id
    unsigned int _position; // Item's stack sequence.
};

#endif // GLASSITEM_H
