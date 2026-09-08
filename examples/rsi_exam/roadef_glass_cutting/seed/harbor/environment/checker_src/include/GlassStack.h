#ifndef GLASSSTACK_H
#define GLASSSTACK_H

#include "GlassItem.h"

/* This class is created
 * to model a batch stack instance with its all attributes
 **/
class GlassStack {
public:
    GlassStack();
    virtual ~GlassStack();

    unsigned int id() {
        return _id;
    }

    unsigned int nbOfItems() {
        return _nbOfItems;
    }

    int curItemIdx() {
        return _curItemIdx;
    }

    GlassItem getitem(unsigned int idx) {
        return item[idx];
    }

    void id(unsigned int val) {
        _id = val;
    }

    void nbOfItems(unsigned int val) {
        _nbOfItems = val;
    }

    void curItemIdx(int val) {
        _curItemIdx = val;
    }

    void Setitem(GlassItem i) {
        item[_itemIdx] = i;
        _itemIdx++;
    }

    void IncreaseNbOfItems() {
        _nbOfItems++;
    }

    void IncreaseCurItemIdx() {
        ++_curItemIdx;
    }

    friend std::ostream & operator<<(std::ostream& os, GlassStack stack) {
        os << "Stack (id,|I|,curItemIdx, itemIdx) = (" << stack._id << "," << stack._nbOfItems << "," << stack._curItemIdx << "," << stack._itemIdx << ")";
        return os;
    }

public:
    GlassItem *item; // Items list.

private:
    unsigned int _id; // Stack Id.
    unsigned int _nbOfItems; // Stack's associated items number.
    int _curItemIdx; // Current items index of stack instance.
    unsigned int _itemIdx; // Used to add item on top of items list of stack instance.
};

#endif // GLASSSTACK_H
