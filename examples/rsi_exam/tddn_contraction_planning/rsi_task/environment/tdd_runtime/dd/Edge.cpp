#include "Edge.hpp"

#include "Node.cpp"

namespace dd {

template <class Node> Edge<Node> Edge<Node>::terminal(const Complex& w) {
  return {Node::getTerminal(), w};
}

template <class Node> bool Edge<Node>::isTerminal() const {
  return Node::isTerminal(p);
}

template <class Node> bool Edge<Node>::isZeroTerminal() const {
  return isTerminal() && w == Complex::zero;
}

template <class Node> bool Edge<Node>::isOneTerminal() const {
  return isTerminal() && w == Complex::one;
}





// Explicit instantiations

template struct Edge<mNode>;


} // namespace dd
