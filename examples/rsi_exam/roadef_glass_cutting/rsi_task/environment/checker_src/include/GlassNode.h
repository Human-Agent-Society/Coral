#ifndef GLASSNODE_H
#define GLASSNODE_H

#include <ostream>

/* This class is created
 * to model a solution node instance with its all attributes
 **/
class GlassNode {
public:
    GlassNode();
    GlassNode(unsigned int plate, unsigned int node, unsigned int x,
            unsigned int y, unsigned int w, unsigned int h,
            int t, unsigned int c, unsigned int p, unsigned int *c_array, unsigned int n);
    virtual ~GlassNode();

    unsigned int plateId() {
        return _plateId;
    }

    unsigned int id() {
        return _id;
    }

    unsigned int x() {
        return _x;
    }

    unsigned int y() {
        return _y;
    }

    unsigned int w() {
        return _w;
    }

    unsigned int h() {
        return _h;
    }

    int nodeType() {
        return _nodeType;
    }

    unsigned int Getcut() {
        return cut;
    }

    unsigned int Getparent() {
        return parent;
    }

    unsigned int Getsuccessor_nbr() {
        return successor_nbr;
    }

    GlassNode Getsuccessor(unsigned int idx) {
        return successor[idx];
    }

    void plateId(unsigned int id) {
        _plateId = id;
    }

    void Setnode_id(unsigned int id) {
        _id = id;
    }

    void x(unsigned int x) {
        _x = x;
    }

    void y(unsigned int y) {
        _y = y;
    }

    void w(unsigned int w) {
        _w = w;
    }

    void h(unsigned int h) {
        _h = h;
    }

    void nodeType(int t) {
        _nodeType = t;
    }

    void Setcut(unsigned int c) {
        cut = c;
    }

    void Setparent(unsigned int p) {
        parent = p;
    }

    void Setchild_nbr(unsigned int nbr) {
        successor_nbr = nbr;
    }

    unsigned int Setsuccessor(GlassNode c) {
        if (successor_nbr >= 2 && cut > 2) {
            return 1;
        }
        successor[successor_nbr] = c;
        successor_nbr++;
        return 0;
    }

    friend std::ostream & operator<<(std::ostream& os, GlassNode node) {
        os << "Node (p,id,x,y,w,h,t,s) = (" << node._plateId << "," << node._id << "," << node._x << "," << node._y << "," << node._w << "," << node._h  << "," << node._nodeType << "," << node.cut << ")";
        return os;
    }

public:
    GlassNode * successor; // Successor nodes list.
protected:
private:
    unsigned int _plateId; // Plate Id.
    unsigned int _id; // Node Id.
    unsigned int _x; // x position.
    unsigned int _y; // y position.
    unsigned int _w; // Plate's width.
    unsigned int _h; // Plate's height.
    int _nodeType; /* node type
                   * > 0 glass piece index in batch file.
                   * = -1 wasted glass.
                   * = -2 branch.
                   * = -3 residual.
                   */
    unsigned int cut; // cut level (there are 1, 2, 3 and 4-cut level, could not be 0-cut because GlassNode class instance can't be an entire plate).
    unsigned int parent; // parent node Id.
    unsigned int successor_nbr; // number of children for this node instance.
};

#endif // GLASSNODE_H
